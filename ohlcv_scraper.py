#!/usr/bin/env python3
"""
OHLCV Scraper → CSV

Fetches candlestick data from Yahoo Finance or Binance and saves to CSV.

- Python ≥ 3.10
- Deps: pandas, numpy, requests, yfinance
- CLI with retries/backoff, Binance pagination, and a selftest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# --------------------------- Logging ---------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------- Utils ---------------------------


def to_utc_ts(s: pd.Series) -> pd.Series:
    """Ensure pandas Series of datetimes are timezone-aware UTC, as ISO8601 strings.

    Handles tz-naive, tz-aware, and non-datetime inputs robustly.
    """
    from pandas.api.types import (
        is_datetime64_any_dtype,
        is_datetime64tz_dtype,
    )

    if not is_datetime64_any_dtype(s):
        # Parse any strings/numbers into datetime, force UTC
        s = pd.to_datetime(s, errors="coerce", utc=True)
    else:
        # Already datetime64; ensure UTC tz-awareness
        if is_datetime64tz_dtype(s):
            s = s.dt.tz_convert("UTC")
        else:
            s = s.dt.tz_localize("UTC")
    return s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date_utc(d: str) -> dt.datetime:
    """Parse YYYY-MM-DD into a UTC datetime at 00:00:00."""
    return dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def end_inclusive_to_exclusive(end: Optional[str]) -> Optional[dt.datetime]:
    """Turn inclusive end date into exclusive end datetime (next day 00:00:00Z)."""
    if not end:
        return None
    end_dt = parse_date_utc(end)
    return end_dt + dt.timedelta(days=1)


def epoch_ms(t: dt.datetime) -> int:
    return int(t.timestamp() * 1000)


# --------------------------- Fetchers ---------------------------


def fetch_yahoo(symbol: str, interval: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """
    Use yfinance.download to fetch OHLCV.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    df = yf.download(
        tickers=symbol,
        interval=interval,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # yfinance may return columns with capitalization and extra fields
    cols_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    # Handle multi-index columns (for tickers) by dropping level if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=cols_map)

    # Index is DatetimeIndex; move to column
    df = df.reset_index().rename(columns={df.columns[0]: "timestamp"})

    needed = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in needed[1:]:
        if c not in df.columns:
            # Some intervals may produce missing fields; create if absent.
            df[c] = np.nan
    df = df[needed]
    return df


def fetch_binance(symbol: str, interval: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance REST /api/v3/klines with pagination (limit 1000 per call).
    Returns DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    base = "https://api.binance.com/api/v3/klines"
    limit = 1000
    start_dt = parse_date_utc(start)
    end_excl = end_inclusive_to_exclusive(end)
    start_ms = epoch_ms(start_dt)
    end_ms = epoch_ms(end_excl) if end_excl else None

    rows: List[List] = []
    current = start_ms

    while True:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit, "startTime": current}
        if end_ms is not None:
            params["endTime"] = end_ms
        resp = requests.get(base, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Binance response: {data}")
        if len(data) == 0:
            break
        rows.extend(data)
        # next start from last open time + 1 ms
        last_open_time = data[-1][0]
        next_start = int(last_open_time) + 1
        if end_ms is not None and next_start >= end_ms:
            break
        current = next_start
        if len(data) < limit:
            break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Binance kline format per entry:
    # [ openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, takerBuyBase, takerBuyQuote, ignore ]
    arr = np.array(rows, dtype=object)
    open_time_ms = arr[:, 0].astype(np.int64)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(open_time_ms, unit="ms", utc=True),
            "open": arr[:, 1].astype(float),
            "high": arr[:, 2].astype(float),
            "low": arr[:, 3].astype(float),
            "close": arr[:, 4].astype(float),
            "volume": arr[:, 5].astype(float),
        }
    )
    return df


# --------------------------- Normalize & Save ---------------------------


def normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ensure UTC timestamps (ISO8601), float prices/volumes, add symbol, sort, drop duplicates.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])

    out = df.copy()
    out["timestamp"] = to_utc_ts(out["timestamp"])
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["symbol"] = symbol
    out = out[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
    out = out.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def save_csv(df: pd.DataFrame, path: Path, append: bool) -> None:
    """
    Save DataFrame to CSV. If append=True and file exists, append without header
    and avoid duplicates by discarding rows with timestamp <= existing max timestamp.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    df = df[cols]

    if append and path.exists():
        try:
            # Read existing max timestamp to prevent duplicates efficiently.
            existing = pd.read_csv(path, usecols=["timestamp"])
            max_ts = existing["timestamp"].max() if not existing.empty else None
            if max_ts is not None:
                df = df[df["timestamp"] > max_ts]
        except Exception as e:
            logging.warning("Failed checking existing CSV for duplicates: %s", e)

        header = not path.exists()  # if somehow removed between checks
        df.to_csv(path, mode="a", header=header, index=False)
    else:
        df.to_csv(path, index=False)


# --------------------------- Retry Wrapper ---------------------------


def with_retry(fn: Callable[[], pd.DataFrame], retries: int, backoff: float, desc: str) -> pd.DataFrame:
    """Retry a callable returning a DataFrame with exponential backoff."""
    attempt = 0
    delay = backoff
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                logging.error("%s failed after %d retries: %s", desc, retries, e)
                raise
            logging.warning("%s failed (attempt %d/%d): %s; retrying in %.2fs", desc, attempt, retries, e, delay)
            time.sleep(delay)
            delay *= backoff


# --------------------------- CLI ---------------------------


YAHOO_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}
BINANCE_INTERVALS = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Fetch OHLCV from Yahoo/Binance and save to CSV.")
    parser.add_argument("--source", choices=["yahoo", "binance"], required=False)
    parser.add_argument("--symbol", type=str, required=False)
    parser.add_argument("--interval", type=str, required=False)
    parser.add_argument("--start", type=str, help="YYYY-MM-DD", required=False)
    parser.add_argument("--end", type=str, help="YYYY-MM-DD (optional)", required=False)
    parser.add_argument("--outfile", type=str, default="candles.csv")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.5)
    parser.add_argument("--selftest", action="store_true", help="Download sample and save to candles_test.csv")
    args = parser.parse_args(argv)

    try:
        if args.selftest:
            # Last 30 days AAPL daily
            end_date = dt.datetime.utcnow().date()
            start_date = end_date - dt.timedelta(days=30)
            outfile = Path("candles_test.csv")

            logging.info("Selftest: fetching AAPL daily from Yahoo: %s to %s", start_date, end_date)
            df_raw = with_retry(
                lambda: fetch_yahoo("AAPL", "1d", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
                retries=args.retry,
                backoff=args.backoff,
                desc="yahoo fetch",
            )
            df = normalize(df_raw, "AAPL")
            if df.empty:
                logging.error("Selftest: no data fetched.")
                return 1
            save_csv(df, outfile, append=False)
            logging.info("Selftest saved: %s (rows=%d)", outfile, len(df))
            # Print last 3 rows
            print(df.tail(3).to_string(index=False))
            return 0

        # Non-selftest mode: validate inputs
        if not args.source or not args.symbol or not args.interval or not args.start:
            parser.error("--source, --symbol, --interval, and --start are required unless --selftest is used.")

        if args.source == "yahoo" and args.interval not in YAHOO_INTERVALS:
            parser.error(f"Invalid Yahoo interval. Allowed: {sorted(YAHOO_INTERVALS)}")
        if args.source == "binance" and args.interval not in BINANCE_INTERVALS:
            parser.error(f"Invalid Binance interval. Allowed: {sorted(BINANCE_INTERVALS)}")

        outfile = Path(args.outfile)
        logging.info(
            "Fetching %s %s %s from %s to %s → %s (append=%s)",
            args.source,
            args.symbol,
            args.interval,
            args.start,
            args.end or "today",
            outfile,
            args.append,
        )

        if args.source == "yahoo":
            df_raw = with_retry(
                lambda: fetch_yahoo(args.symbol, args.interval, args.start, args.end),
                retries=args.retry,
                backoff=args.backoff,
                desc="yahoo fetch",
            )
        else:
            df_raw = with_retry(
                lambda: fetch_binance(args.symbol, args.interval, args.start, args.end),
                retries=args.retry,
                backoff=args.backoff,
                desc="binance fetch",
            )

        df = normalize(df_raw, args.symbol)
        if df.empty:
            logging.error("No data fetched. Exiting.")
            return 1

        save_csv(df, outfile, append=args.append)

        first_ts = df["timestamp"].iloc[0]
        last_ts = df["timestamp"].iloc[-1]
        logging.info("Saved %d rows to %s (first=%s, last=%s)", len(df), outfile, first_ts, last_ts)
        return 0

    except SystemExit:
        raise
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
