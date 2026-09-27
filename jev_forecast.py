#!/usr/bin/env python3
"""On-demand SPY or BTC forecasts using Twelve Data candles and Jev via OpenRouter.

This is a local, signal-only tool. It does not place orders and does not train a
market model. One Jev request is made for each new completed 15-minute bar;
repeated runs for the same bar reuse the saved forecast unless --force is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import json
import math
import os
import sqlite3
import ssl
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


TWELVE_DATA_ENDPOINT = "https://api.twelvedata.com/time_series"
OPENROUTER_DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_OPENROUTER_MODEL = "~typesafe/jev-latest"
EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc
SUPPORTED_INTERVALS = {"15min": 15}
SIGNAL_CLASSES = ("LONG", "FLAT", "SHORT")
MIN_FEATURE_BARS = 22
TAKE_PROFIT_ATR_CHOICES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
STOP_LOSS_ATR_CHOICES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
ATR_CHOICE_CONFIG_JSON = json.dumps(
    {
        "take_profit_atr": TAKE_PROFIT_ATR_CHOICES,
        "stop_loss_atr": STOP_LOSS_ATR_CHOICES,
    },
    sort_keys=True,
    separators=(",", ":"),
)


class ForecastError(RuntimeError):
    """An actionable input, provider, or forecast error."""


@dataclass(frozen=True)
class Candle:
    """One OHLCV candle; timestamp is its opening time in UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    exchange: str | None = None

    @property
    def close_time(self) -> datetime:
        return self.timestamp + timedelta(minutes=SUPPORTED_INTERVALS["15min"])


@dataclass(frozen=True)
class JevAnswer:
    signal: str
    probabilities: dict[str, float]
    confidence: float | None
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None = None
    take_profit_atr: float | None = None
    stop_loss_atr: float | None = None


@dataclass(frozen=True)
class TradeLevels:
    atr_price: float
    take_profit_price: float | None
    stop_loss_price: float | None


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Expected a timezone-aware datetime")
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _parse_provider_time(value: str, timezone_name: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ForecastError(f"Invalid candle timestamp from Twelve Data: {value!r}") from exc
    if parsed.tzinfo is None:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
        except Exception as exc:
            raise ForecastError(f"Invalid timezone in Twelve Data response: {timezone_name!r}") from exc
    return parsed.astimezone(UTC)


def _finite_number(value: Any, field: str, *, optional: bool = False) -> float | None:
    if optional and (value is None or value == ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ForecastError(f"Twelve Data returned an invalid {field} value") from exc
    if not math.isfinite(number):
        raise ForecastError(f"Twelve Data returned a non-finite {field} value")
    return number


def parse_twelvedata_payload(
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
    interval: str = "15min",
    timezone_name: str = "America/New_York",
    symbol: str = "SPY",
) -> list[Candle]:
    """Parse a Twelve Data time-series response and discard any forming candle."""
    if interval not in SUPPORTED_INTERVALS:
        raise ForecastError(f"Unsupported interval {interval!r}; supported: {', '.join(SUPPORTED_INTERVALS)}")
    if payload.get("status") == "error" or not isinstance(payload.get("values"), list):
        message = payload.get("message") or payload.get("code") or "no candle values returned"
        raise ForecastError(f"Twelve Data returned an error: {message}")

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    interval_delta = timedelta(minutes=SUPPORTED_INTERVALS[interval])
    metadata = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    exchange = str(metadata["exchange"]) if metadata.get("exchange") else None
    deduplicated: dict[datetime, Candle] = {}
    for raw in payload["values"]:
        if not isinstance(raw, dict) or "datetime" not in raw:
            continue
        timestamp = _parse_provider_time(str(raw["datetime"]), timezone_name)
        open_price = _finite_number(raw.get("open"), "open")
        high = _finite_number(raw.get("high"), "high")
        low = _finite_number(raw.get("low"), "low")
        close = _finite_number(raw.get("close"), "close")
        volume = _finite_number(raw.get("volume"), "volume", optional=True)
        assert open_price is not None and high is not None and low is not None and close is not None

        if min(open_price, high, low, close) <= 0:
            raise ForecastError(f"Twelve Data returned a non-positive {symbol} price")
        if high < low or high < max(open_price, close) or low > min(open_price, close):
            raise ForecastError(f"Twelve Data returned inconsistent OHLC values for {symbol}")
        if volume is not None and volume < 0:
            raise ForecastError("Twelve Data returned negative volume")

        candle = Candle(timestamp, open_price, high, low, close, volume, exchange)
        if candle.timestamp + interval_delta <= current_time:
            deduplicated[timestamp] = candle

    candles = [deduplicated[key] for key in sorted(deduplicated)]
    if not candles:
        raise ForecastError("Twelve Data has not returned a completed 15-minute candle yet")
    return candles


def _decode_json_response(response_bytes: bytes, service: str) -> dict[str, Any]:
    try:
        decoded = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForecastError(f"{service} returned an invalid JSON response") from exc
    if not isinstance(decoded, dict):
        raise ForecastError(f"{service} returned an unexpected response shape")
    return decoded


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """Create a verified TLS context, using an explicit CA bundle if needed."""
    try:
        if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
            context = ssl.create_default_context()
        else:
            try:
                import certifi
            except ImportError:
                context = ssl.create_default_context()
            else:
                context = ssl.create_default_context(cafile=certifi.where())
    except (OSError, ssl.SSLError) as exc:
        raise ForecastError(
            "Could not load TLS root certificates. Install/update certifi or set SSL_CERT_FILE to a trusted CA bundle."
        ) from exc

    if not context.get_ca_certs():
        raise ForecastError(
            "Python has no trusted TLS root certificates. Install certifi or set SSL_CERT_FILE to a trusted CA bundle."
        )
    return context


def _http_json(request: Request, *, service: str, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            return _decode_json_response(response.read(), service)
    except HTTPError as exc:
        # Do not include the request URL: Twelve Data's key is in its query string.
        body = exc.read()
        detail = ""
        try:
            payload = json.loads(body.decode("utf-8"))
            if isinstance(payload, dict):
                detail = str(payload.get("message") or payload.get("detail") or payload.get("error") or "")
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        suffix = f": {detail[:240]}" if detail else ""
        raise ForecastError(f"{service} request failed with HTTP {exc.code}{suffix}") from None
    except URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise ForecastError(
                "TLS certificate verification failed. Update certifi or set SSL_CERT_FILE to a trusted CA bundle."
            ) from None
        raise ForecastError(f"Could not reach {service}: {exc.reason}") from None
    except TimeoutError:
        raise ForecastError(f"{service} request timed out") from None


def fetch_twelvedata_bars(
    api_key: str,
    *,
    symbol: str = "SPY",
    interval: str = "15min",
    outputsize: int = 128,
    timezone_name: str = "America/New_York",
    exchange: str | None = None,
    now: datetime | None = None,
    timeout: float = 20,
) -> list[Candle]:
    """Fetch completed intraday candles from Twelve Data."""
    if not api_key.strip():
        raise ForecastError("Set TWELVEDATA_API_KEY before requesting market data")
    if interval not in SUPPORTED_INTERVALS:
        raise ForecastError(f"Unsupported interval {interval!r}; supported: {', '.join(SUPPORTED_INTERVALS)}")
    if not 24 <= outputsize <= 5000:
        raise ForecastError("outputsize must be between 24 and 5000")

    query = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "order": "asc",
        "timezone": timezone_name,
        "apikey": api_key,
    }
    if exchange:
        query["exchange"] = exchange
    params = urlencode(query)
    request = Request(
        f"{TWELVE_DATA_ENDPOINT}?{params}",
        headers={"Accept": "application/json", "User-Agent": "candleml-jev-forecast/1.0"},
    )
    payload = _http_json(request, service="Twelve Data", timeout=timeout)
    return parse_twelvedata_payload(
        payload,
        now=now,
        interval=interval,
        timezone_name=timezone_name,
        symbol=symbol,
    )


def _ema(values: Sequence[float], period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _return_bps(old: float, new: float) -> float:
    return (new / old - 1.0) * 10_000.0


def _rsi(closes: Sequence[float], period: int = 14) -> float:
    changes = [closes[index] - closes[index - 1] for index in range(len(closes) - period, len(closes))]
    gains = sum(max(change, 0.0) for change in changes) / period
    losses = sum(max(-change, 0.0) for change in changes) / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + gains / losses))


def compute_features(candles: Sequence[Candle]) -> dict[str, float | None]:
    """Compute causal, scale-relative features from completed bars only."""
    if len(candles) < MIN_FEATURE_BARS:
        raise ForecastError(f"Need at least {MIN_FEATURE_BARS} completed bars; received {len(candles)}")
    closes = [candle.close for candle in candles]
    last = candles[-1]
    prior_close = candles[-2].close

    returns = [closes[index] / closes[index - 1] - 1.0 for index in range(1, len(closes))]
    one_bar_vol = statistics.pstdev(returns[-16:]) if len(returns) >= 2 else 0.0
    true_ranges = []
    for index in range(max(1, len(candles) - 14), len(candles)):
        candle = candles[index]
        previous = candles[index - 1].close
        true_ranges.append(max(candle.high - candle.low, abs(candle.high - previous), abs(candle.low - previous)))
    atr = sum(true_ranges) / len(true_ranges)

    features: dict[str, float | None] = {
        "return_15m_bps": _return_bps(prior_close, last.close),
        "momentum_1h_bps": _return_bps(closes[-5], last.close),
        "momentum_4h_bps": _return_bps(closes[-17], last.close),
        "ema_8_vs_21_bps": _return_bps(_ema(closes[-21:], 21), _ema(closes[-21:], 8)),
        "rsi_14": _rsi(closes),
        "atr_14_bps": atr / last.close * 10_000.0,
        "realized_vol_1h_bps": one_bar_vol * math.sqrt(4.0) * 10_000.0,
        "last_bar_body_bps": _return_bps(last.open, last.close),
        "last_bar_range_bps": (last.high - last.low) / prior_close * 10_000.0,
        "volume_ratio_4_to_20": None,
    }
    if len(candles) >= 24 and all(candle.volume is not None for candle in candles[-24:]):
        volumes = [float(candle.volume) for candle in candles[-24:] if candle.volume is not None]
        base = sum(volumes[-24:-4]) / 20.0
        recent = sum(volumes[-4:]) / 4.0
        features["volume_ratio_4_to_20"] = recent / base if base > 0 else None
    return {name: round(value, 5) if value is not None else None for name, value in features.items()}


def build_market_state(
    candles: Sequence[Candle],
    *,
    symbol: str = "SPY",
    interval: str = "15min",
    horizon_bars: int = 4,
    neutral_band_bps: float = 10.0,
    state_bars: int = 32,
    market: str = "equity",
) -> dict[str, Any]:
    """Create the JSON market state submitted to Jev."""
    if horizon_bars < 1:
        raise ForecastError("horizon_bars must be at least 1")
    if not 1.0 <= neutral_band_bps <= 500.0:
        raise ForecastError("neutral_band_bps must be between 1 and 500")
    latest = candles[-1]
    interval_minutes = SUPPORTED_INTERVALS[interval]
    decision_time = latest.timestamp + timedelta(minutes=interval_minutes)
    recent = candles[-state_bars:]
    if market == "crypto":
        description = (
            "Bitcoin priced in U.S. dollars from Twelve Data; this exchange-market series may differ "
            "from the FTMO BTCUSD CFD quote."
            if symbol.upper() in {"BTC/USD", "BTCUSD", "BTCUSDT"}
            else f"{symbol} cryptocurrency price series supplied by Twelve Data."
        )
        bar_timezone = "UTC"
    else:
        description = (
            "SPY ETF, used as an S&P 500 proxy; it is not the index or a CFD."
            if symbol.upper() == "SPY"
            else f"{symbol} price series supplied by Twelve Data."
        )
        bar_timezone = "America/New_York"
    return {
        "instrument": {
            "symbol": symbol,
            "description": description,
            "bar_interval": interval,
            "bar_timezone": bar_timezone,
            "market": market,
            "data_provider": "Twelve Data",
            "exchange": latest.exchange,
        },
        "decision_time_utc": _iso_utc(decision_time),
        "last_completed_bar": {
            "opened_at_utc": _iso_utc(latest.timestamp),
            "closed_at_utc": _iso_utc(decision_time),
            "close": latest.close,
        },
        "forecast_target": {
            "horizon_bars": horizon_bars,
            "horizon_minutes": horizon_bars * interval_minutes,
            "neutral_band_bps": neutral_band_bps,
            "definition": (
                "Compare the close at decision_time with the close after the next "
                f"{horizon_bars} completed {interval} bars. LONG means return > "
                f"+{neutral_band_bps} bps; FLAT means return is within +/-{neutral_band_bps} bps; "
                f"SHORT means return < -{neutral_band_bps} bps."
            ),
        },
        "features": compute_features(candles),
        "recent_completed_bars": [
            {
                "opened_at_utc": _iso_utc(candle.timestamp),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in recent
        ],
    }


def validate_forecast_window(
    candles: Sequence[Candle],
    *,
    horizon_bars: int,
    max_bar_age_minutes: int = 30,
    allow_stale: bool = False,
    now: datetime | None = None,
    market: str = "equity",
    symbol: str = "SPY",
) -> datetime:
    """Reject stale bars and, for equities, bars outside their regular session."""
    if not candles:
        raise ForecastError("No completed candles are available")
    if horizon_bars < 1:
        raise ForecastError("horizon_bars must be at least 1")
    latest = candles[-1]
    decision_time = latest.close_time
    current = (now or datetime.now(UTC)).astimezone(UTC)
    age = current - decision_time
    if age < timedelta(minutes=-1):
        raise ForecastError("Latest candle appears to be in the future; check the provider timezone")
    if not allow_stale and age > timedelta(minutes=max_bar_age_minutes):
        if market == "crypto":
            raise ForecastError(
                f"Latest completed {symbol} candle is {int(age.total_seconds() // 60)} minutes old; "
                f"limit is {max_bar_age_minutes}. Use --allow-stale only for an explicitly stale forecast."
            )
        current_eastern = current.astimezone(EASTERN)
        session_open = wall_time(9, 30)
        session_close = wall_time(16, 0)
        market_is_open = (
            current_eastern.weekday() < 5
            and session_open <= current_eastern.time() < session_close
        )
        if not market_is_open:
            last_close_eastern = decision_time.astimezone(EASTERN)
            raise ForecastError(
                f"The latest {symbol} candle closed at {last_close_eastern:%Y-%m-%d %H:%M ET} "
                f"and is {int(age.total_seconds() // 60)} minutes old. The regular U.S. "
                "session is closed, and Twelve Data Basic does not provide pre/post-market "
                "candles. Run during 09:30-16:00 ET; for a one-hour forecast, the decision "
                "bar must close by 15:00 ET."
            )
        raise ForecastError(
            f"Latest completed candle is {int(age.total_seconds() // 60)} minutes old; "
            f"limit is {max_bar_age_minutes}. Use --allow-stale only for an explicitly stale forecast."
        )
    if not allow_stale and market == "equity":
        eastern_decision = decision_time.astimezone(EASTERN)
        if eastern_decision.weekday() >= 5:
            raise ForecastError(f"{symbol} forecasts are limited to regular U.S. market sessions")
        if eastern_decision.time() < wall_time(9, 45) or eastern_decision.time() > wall_time(16, 0):
            raise ForecastError("Latest candle is outside the regular U.S. equity session")
        horizon_end = eastern_decision + timedelta(minutes=horizon_bars * 15)
        if horizon_end.date() != eastern_decision.date() or horizon_end.time() > wall_time(16, 0):
            raise ForecastError(
                "Not enough regular-session time remains for this forecast horizon; "
                "try again earlier in the session or use --allow-stale for research."
            )
    return decision_time


def build_jev_request(
    state: dict[str, Any],
    *,
    model: str = DEFAULT_OPENROUTER_MODEL,
) -> dict[str, Any]:
    neutral_band = state["forecast_target"]["neutral_band_bps"]
    horizon = state["forecast_target"]["horizon_minutes"]
    symbol = state["instrument"]["symbol"]
    criteria = {
        "LONG": f"The {symbol} close {horizon} minutes after decision_time is more than {neutral_band} basis points higher than the decision-time close.",
        "FLAT": f"The {symbol} close {horizon} minutes after decision_time is within +/-{neutral_band} basis points of the decision-time close.",
        "SHORT": f"The {symbol} close {horizon} minutes after decision_time is more than {neutral_band} basis points lower than the decision-time close.",
    }
    take_profit_criteria = {
        f"{multiple:g} ATR": f"Set the take-profit distance to {multiple:g} times the 14-bar ATR."
        for multiple in TAKE_PROFIT_ATR_CHOICES
    }
    stop_loss_criteria = {
        f"{multiple:g} ATR": f"Set the stop-loss distance to {multiple:g} times the 14-bar ATR."
        for multiple in STOP_LOSS_ATR_CHOICES
    }
    level_instructions = (
        f"Using only completed bars and features in state, choose the distance that best fits the "
        f"likely {symbol} direction and the {horizon}-minute horizon. Choose one listed ATR multiple; "
        "do not return an absolute price. If the direction is FLAT, the selected distance will be ignored."
    )
    return {
        "model": model,
        "state": state,
        "questions": {
            "direction": {
                "type": "choice",
                "instructions": (
                    f"Using only the completed market data in state, which defined return class is "
                    f"most likely for {symbol} over the stated forecast horizon? The state contains no "
                    "news or future data. Return the most likely class and its probability distribution."
                ),
                "criteria": criteria,
            },
            "take_profit_atr": {
                "type": "choice",
                "instructions": f"Choose a take-profit distance. {level_instructions}",
                "criteria": take_profit_criteria,
            },
            "stop_loss_atr": {
                "type": "choice",
                "instructions": f"Choose a stop-loss distance. {level_instructions}",
                "criteria": stop_loss_criteria,
            },
        },
    }


def _jev_answer_envelope(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Find the result object across direct, result-wrapped, or data-wrapped replies."""
    wrappers = ("result", "data", "response", "output", "payload")

    def walk(value: dict[str, Any], layers: list[dict[str, Any]], depth: int) -> list[dict[str, Any]] | None:
        current_layers = [*layers, value]
        if isinstance(value.get("answers"), dict):
            return current_layers
        if depth >= 4:
            return None
        for key in wrappers:
            nested = value.get(key)
            if isinstance(nested, dict):
                found = walk(nested, current_layers, depth + 1)
                if found is not None:
                    return found
            elif isinstance(nested, str):
                try:
                    decoded = json.loads(nested)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    found = walk(decoded, current_layers, depth + 1)
                    if found is not None:
                        return found
        return None

    layers = walk(payload, [], 0)
    if layers is None:
        return None
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer)
    return merged


def _jev_response_shape(payload: dict[str, Any]) -> str:
    """Summarize response keys only; never include response values or secrets."""
    wrappers = ("result", "data", "response", "output", "payload")
    summaries: list[str] = []
    current = payload
    for depth in range(4):
        keys = ",".join(sorted(str(key) for key in current.keys())[:12])
        summaries.append(f"level{depth}=[{keys}]")
        nested = next((current.get(key) for key in wrappers if isinstance(current.get(key), dict)), None)
        if not isinstance(nested, dict):
            break
        current = nested
    return "; ".join(summaries)


def _parse_atr_choice(
    answers: dict[str, Any],
    question_name: str,
    allowed_values: Sequence[float],
) -> float:
    answer = answers.get(question_name)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ForecastError(f"Jev returned a missing or non-Choice {question_name} answer")
    choice = answer.get("choice")
    if not isinstance(choice, str):
        raise ForecastError(f"Jev returned an invalid {question_name} choice")
    allowed = {f"{value:g} ATR".casefold(): value for value in allowed_values}
    selected = allowed.get(choice.strip().casefold())
    if selected is None:
        raise ForecastError(f"Jev returned an unsupported {question_name} choice: {choice!r}")
    return selected


def parse_jev_response(payload: dict[str, Any]) -> JevAnswer:
    """Validate direction probabilities and Jev-selected ATR distances."""
    envelope = _jev_answer_envelope(payload)
    if envelope is None:
        raise ForecastError(
            "Jev response did not contain an answers object; "
            f"response shape: {_jev_response_shape(payload)}"
        )
    payload = envelope
    try:
        model = str(payload["model"])
        answers = payload["answers"]
        answer = answers["direction"]
        choice = str(answer["choice"]).upper()
        raw_probabilities = answer["probabilities"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ForecastError(
            f"Jev response did not contain a direction Choice; response shape: {_jev_response_shape(payload)}"
        ) from exc
    if answer.get("type") != "choice":
        raise ForecastError("Jev returned a non-Choice answer for direction")
    if choice not in SIGNAL_CLASSES or not isinstance(raw_probabilities, dict):
        raise ForecastError("Jev returned an unknown signal class or invalid probabilities")

    probabilities: dict[str, float] = {}
    for signal_class in SIGNAL_CLASSES:
        probability = _finite_number(raw_probabilities.get(signal_class), f"{signal_class} probability")
        assert probability is not None
        if probability < 0.0 or probability > 1.0:
            raise ForecastError("Jev returned a probability outside [0, 1]")
        probabilities[signal_class] = probability
    probability_sum = sum(probabilities.values())
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=0.02):
        raise ForecastError(f"Jev probabilities do not sum to 1 (sum={probability_sum:.4f})")
    probabilities = {key: value / probability_sum for key, value in probabilities.items()}
    most_likely = max(SIGNAL_CLASSES, key=probabilities.__getitem__)
    if choice != most_likely:
        raise ForecastError("Jev selected class does not match the highest returned probability")

    confidence_value = answer.get("confidence")
    confidence = _finite_number(confidence_value, "confidence", optional=True)
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ForecastError("Jev returned a confidence outside [0, 1]")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cost_usd = _finite_number(usage.get("cost"), "cost", optional=True)
    take_profit_atr = _parse_atr_choice(answers, "take_profit_atr", TAKE_PROFIT_ATR_CHOICES)
    stop_loss_atr = _parse_atr_choice(answers, "stop_loss_atr", STOP_LOSS_ATR_CHOICES)
    return JevAnswer(
        signal=choice,
        probabilities=probabilities,
        confidence=confidence,
        model=model,
        input_tokens=int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
        output_tokens=int(output_tokens) if isinstance(output_tokens, (int, float)) else None,
        cost_usd=cost_usd,
        take_profit_atr=take_profit_atr,
        stop_loss_atr=stop_loss_atr,
    )


def call_jev(
    api_key: str,
    state: dict[str, Any],
    *,
    model: str = DEFAULT_OPENROUTER_MODEL,
    timeout: float = 30,
) -> JevAnswer:
    if not api_key.strip():
        raise ForecastError("Set OPENROUTER_API_KEY before requesting a Jev forecast")
    payload = json.dumps(build_jev_request(state, model=model), separators=(",", ":")).encode("utf-8")
    request = Request(
        OPENROUTER_DECISIONS_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "candleml-jev-forecast/1.0",
        },
    )
    response = _http_json(request, service="OpenRouter Jev", timeout=timeout)
    return parse_jev_response(response)


def openrouter_configuration(environ: dict[str, str] | None = None) -> tuple[str, str]:
    """Resolve OpenRouter credentials without reusing keys for other providers."""
    env = os.environ if environ is None else environ
    api_key = env.get("OPENROUTER_API_KEY", "")
    model = env.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    return api_key, model


def classify_return(return_pct: float, neutral_band_bps: float) -> str:
    band_pct = neutral_band_bps / 100.0
    if return_pct > band_pct:
        return "LONG"
    if return_pct < -band_pct:
        return "SHORT"
    return "FLAT"


def calculate_trade_levels(
    *,
    entry_price: float,
    signal: str,
    atr_14_bps: float,
    take_profit_atr: float,
    stop_loss_atr: float,
) -> TradeLevels:
    """Calculate directional price levels from the decision close and 14-bar ATR."""
    if not math.isfinite(take_profit_atr) or take_profit_atr <= 0:
        raise ForecastError("take_profit_atr must be a finite number greater than zero")
    if not math.isfinite(stop_loss_atr) or stop_loss_atr <= 0:
        raise ForecastError("stop_loss_atr must be a finite number greater than zero")
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ForecastError("entry price must be a finite number greater than zero")
    if not math.isfinite(atr_14_bps) or atr_14_bps < 0:
        raise ForecastError("14-bar ATR must be a finite non-negative number")

    atr_price = entry_price * atr_14_bps / 10_000.0
    if signal == "FLAT":
        return TradeLevels(atr_price, None, None)
    if signal not in {"LONG", "SHORT"}:
        raise ForecastError(f"Unknown signal for trade levels: {signal!r}")
    if atr_price <= 0:
        raise ForecastError("14-bar ATR must be greater than zero for a directional signal")

    target_distance = atr_price * take_profit_atr
    stop_distance = atr_price * stop_loss_atr
    if signal == "LONG":
        take_profit_price = entry_price + target_distance
        stop_loss_price = entry_price - stop_distance
    else:
        take_profit_price = entry_price - target_distance
        stop_loss_price = entry_price + stop_distance
    if take_profit_price <= 0 or stop_loss_price <= 0:
        raise ForecastError("ATR multipliers produce a non-positive price level; reduce the multiplier")
    return TradeLevels(atr_price, take_profit_price, stop_loss_price)


class ForecastStore:
    """Local SQLite ledger for forecasts and resolved forward outcomes."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    decision_time_utc TEXT NOT NULL,
                    decision_close REAL NOT NULL,
                    horizon_bars INTEGER NOT NULL,
                    neutral_band_bps REAL NOT NULL,
                    lookback_bars INTEGER NOT NULL,
                    requested_model TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    confidence REAL,
                    resolved_model TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL,
                    atr_price REAL,
                    take_profit_atr REAL,
                    stop_loss_atr REAL,
                    atr_choice_config_json TEXT,
                    take_profit_price REAL,
                    stop_loss_price REAL,
                    input_state_json TEXT NOT NULL,
                    requested_at_utc TEXT NOT NULL,
                    realized_return_pct REAL,
                    realized_class TEXT,
                    resolved_at_utc TEXT,
                    trade_outcome TEXT,
                    trade_return_pct REAL,
                    trade_exit_price REAL,
                    trade_exit_time_utc TEXT
                )
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(forecasts)").fetchall()
            }
            migrations = {
                "cost_usd": "REAL",
                "atr_price": "REAL",
                "take_profit_atr": "REAL",
                "stop_loss_atr": "REAL",
                "atr_choice_config_json": "TEXT",
                "take_profit_price": "REAL",
                "stop_loss_price": "REAL",
                "trade_outcome": "TEXT",
                "trade_return_pct": "REAL",
                "trade_exit_price": "REAL",
                "trade_exit_time_utc": "TEXT",
            }
            for column, column_type in migrations.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE forecasts ADD COLUMN {column} {column_type}")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS forecasts_lookup_idx "
                "ON forecasts(symbol, interval, decision_time_utc, horizon_bars, neutral_band_bps, lookback_bars, requested_model)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS forecasts_pending_idx ON forecasts(realized_class, decision_time_utc)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def find_cached(
        self,
        *,
        symbol: str,
        interval: str,
        decision_time: datetime,
        horizon_bars: int,
        neutral_band_bps: float,
        lookback_bars: int,
        requested_model: str,
        atr_choice_config_json: str | None = None,
    ) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM forecasts
                WHERE symbol = ? AND interval = ? AND decision_time_utc = ?
                  AND horizon_bars = ? AND neutral_band_bps = ?
                  AND lookback_bars = ? AND requested_model = ?
                  AND atr_choice_config_json IS ?
                ORDER BY id DESC LIMIT 1
                """,
                (
                    symbol,
                    interval,
                    _iso_utc(decision_time),
                    horizon_bars,
                    neutral_band_bps,
                    lookback_bars,
                    requested_model,
                    atr_choice_config_json,
                ),
            ).fetchone()

    def save_forecast(
        self,
        *,
        symbol: str,
        interval: str,
        decision_time: datetime,
        decision_close: float,
        horizon_bars: int,
        neutral_band_bps: float,
        lookback_bars: int,
        requested_model: str,
        answer: JevAnswer,
        state: dict[str, Any],
        atr_price: float | None = None,
        take_profit_atr: float | None = None,
        stop_loss_atr: float | None = None,
        atr_choice_config_json: str | None = None,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        requested_at: datetime | None = None,
    ) -> int:
        if (take_profit_atr is None) != (stop_loss_atr is None):
            raise ForecastError("take_profit_atr and stop_loss_atr must be provided together")
        requested_timestamp = requested_at or datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO forecasts (
                    symbol, interval, decision_time_utc, decision_close,
                    horizon_bars, neutral_band_bps, lookback_bars, requested_model,
                    signal, probabilities_json, confidence, resolved_model,
                    input_tokens, output_tokens, cost_usd, atr_price,
                    take_profit_atr, stop_loss_atr, atr_choice_config_json,
                    take_profit_price, stop_loss_price,
                    input_state_json, requested_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    interval,
                    _iso_utc(decision_time),
                    decision_close,
                    horizon_bars,
                    neutral_band_bps,
                    lookback_bars,
                    requested_model,
                    answer.signal,
                    json.dumps(answer.probabilities, sort_keys=True),
                    answer.confidence,
                    answer.model,
                    answer.input_tokens,
                    answer.output_tokens,
                    answer.cost_usd,
                    atr_price,
                    take_profit_atr,
                    stop_loss_atr,
                    atr_choice_config_json,
                    take_profit_price,
                    stop_loss_price,
                    json.dumps(state, sort_keys=True, separators=(",", ":")),
                    _iso_utc(requested_timestamp),
                ),
            )
            return int(cursor.lastrowid)

    def resolve_pending(
        self,
        candles: Sequence[Candle],
        *,
        symbol: str,
        interval: str,
        now: datetime | None = None,
    ) -> int:
        if interval not in SUPPORTED_INTERVALS:
            raise ForecastError(f"Unsupported interval {interval!r}")
        interval_delta = timedelta(minutes=SUPPORTED_INTERVALS[interval])
        candle_by_time = {candle.timestamp: candle for candle in candles}
        resolved_at = _iso_utc(now or datetime.now(UTC))
        resolved_count = 0
        with self._connect() as connection:
            pending = connection.execute(
                """
                SELECT * FROM forecasts
                WHERE symbol = ? AND interval = ?
                  AND (
                    realized_class IS NULL
                    OR (take_profit_atr IS NOT NULL AND stop_loss_atr IS NOT NULL AND trade_outcome IS NULL)
                  )
                ORDER BY decision_time_utc
                """,
                (symbol, interval),
            ).fetchall()
            for row in pending:
                decision_time = datetime.fromisoformat(row["decision_time_utc"]).astimezone(UTC)
                horizon_bars = int(row["horizon_bars"])
                future_bars = [
                    candle_by_time.get(decision_time + interval_delta * offset)
                    for offset in range(horizon_bars)
                ]
                levels_enabled = row["take_profit_atr"] is not None and row["stop_loss_atr"] is not None
                directional = row["signal"] in {"LONG", "SHORT"}
                entry_price = float(row["decision_close"])
                trade_outcome = row["trade_outcome"]

                if levels_enabled and directional and trade_outcome is None:
                    take_profit = row["take_profit_price"]
                    stop_loss = row["stop_loss_price"]
                    if take_profit is not None and stop_loss is not None:
                        for candle in future_bars:
                            if candle is None:
                                break
                            if row["signal"] == "LONG":
                                take_profit_hit = candle.high >= float(take_profit)
                                stop_loss_hit = candle.low <= float(stop_loss)
                            else:
                                take_profit_hit = candle.low <= float(take_profit)
                                stop_loss_hit = candle.high >= float(stop_loss)

                            # OHLC candles do not reveal which level was crossed first.
                            # Use the conservative stop-first assumption when both are hit.
                            if stop_loss_hit:
                                exit_price = float(stop_loss)
                                outcome = "STOP_LOSS"
                            elif take_profit_hit:
                                exit_price = float(take_profit)
                                outcome = "TAKE_PROFIT"
                            else:
                                continue

                            side = 1.0 if row["signal"] == "LONG" else -1.0
                            trade_return_pct = (exit_price / entry_price - 1.0) * 100.0 * side
                            connection.execute(
                                """
                                UPDATE forecasts
                                SET trade_outcome = ?, trade_return_pct = ?, trade_exit_price = ?,
                                    trade_exit_time_utc = ?
                                WHERE id = ?
                                """,
                                (
                                    outcome,
                                    trade_return_pct,
                                    exit_price,
                                    _iso_utc(candle.close_time),
                                    int(row["id"]),
                                ),
                            )
                            trade_outcome = outcome
                            break

                # Resolve the close-based classification once the horizon bar is
                # available. TP/SL evaluation also needs each intervening bar so
                # a missing candle cannot hide an earlier barrier touch.
                outcome_candle = future_bars[-1]
                if outcome_candle is None:
                    continue
                if row["realized_class"] is None:
                    realized_return_pct = (outcome_candle.close / entry_price - 1.0) * 100.0
                    realized_class = classify_return(realized_return_pct, float(row["neutral_band_bps"]))
                    connection.execute(
                        """
                        UPDATE forecasts
                        SET realized_return_pct = ?, realized_class = ?, resolved_at_utc = ?
                        WHERE id = ?
                        """,
                        (realized_return_pct, realized_class, resolved_at, int(row["id"])),
                    )
                    resolved_count += 1

                if levels_enabled and trade_outcome is None:
                    if any(candle is None for candle in future_bars):
                        continue
                    if not directional:
                        trade_outcome = "NO_TRADE"
                        trade_return_pct = None
                        trade_exit_price = None
                        trade_exit_time = None
                    elif row["take_profit_price"] is None or row["stop_loss_price"] is None:
                        trade_outcome = "NO_LEVELS"
                        trade_return_pct = None
                        trade_exit_price = None
                        trade_exit_time = None
                    else:
                        side = 1.0 if row["signal"] == "LONG" else -1.0
                        trade_outcome = "HORIZON"
                        trade_return_pct = (outcome_candle.close / entry_price - 1.0) * 100.0 * side
                        trade_exit_price = outcome_candle.close
                        trade_exit_time = _iso_utc(outcome_candle.close_time)
                    connection.execute(
                        """
                        UPDATE forecasts
                        SET trade_outcome = ?, trade_return_pct = ?, trade_exit_price = ?,
                            trade_exit_time_utc = ?
                        WHERE id = ?
                        """,
                        (
                            trade_outcome,
                            trade_return_pct,
                            trade_exit_price,
                            trade_exit_time,
                            int(row["id"]),
                        ),
                    )
        return resolved_count

    def list_forecasts(self, *, limit: int = 20) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM forecasts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()


def _default_db_path() -> Path:
    return Path.home() / ".candleml" / "jev_forecasts.sqlite3"


def _market_db_path(market: str) -> Path:
    if market == "crypto":
        return Path.home() / ".candleml" / "btc_forecasts.sqlite3"
    return _default_db_path()


def _ledger_symbol(symbol: str, exchange: str | None, candles: Sequence[Candle]) -> str:
    """Keep cached forecasts and outcomes isolated by the candle venue."""
    resolved_exchange = candles[-1].exchange if candles else None
    venue = resolved_exchange or exchange
    return f"{symbol}@{venue}" if venue else symbol


def load_env_file(path: str | Path, environ: dict[str, str] | None = None) -> None:
    """Load simple KEY=value entries without overriding non-empty environment values."""
    env = os.environ if environ is None else environ
    env_path = Path(path).expanduser()
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ForecastError(f"Could not read local environment file: {env_path}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not env.get(name):
            env[name] = value


def _print_forecast(
    *,
    symbol: str,
    interval: str,
    horizon_bars: int,
    decision_time: datetime,
    signal: str,
    probabilities: dict[str, float],
    confidence: float | None,
    model: str,
    input_tokens: int | None,
    reference_price: float,
    atr_price: float | None,
    take_profit_atr: float,
    stop_loss_atr: float,
    take_profit_price: float | None,
    stop_loss_price: float | None,
    cost_usd: float | None = None,
    cached: bool = False,
    market: str = "equity",
) -> None:
    display_timezone = UTC if market == "crypto" else EASTERN
    local_time = decision_time.astimezone(display_timezone)
    print(f"{symbol} {interval} | decision bar closed {local_time:%Y-%m-%d %H:%M:%S %Z}")
    horizon_minutes = horizon_bars * SUPPORTED_INTERVALS[interval]
    print(f"Horizon: {horizon_bars} bars ({horizon_minutes} minutes)")
    print(f"Signal: {signal}{' (cached)' if cached else ''}")
    print("Probabilities: " + " | ".join(f"{label} {probabilities[label]:.1%}" for label in SIGNAL_CLASSES))
    print(f"Reference close: {_format_price(reference_price)}")
    if atr_price is not None:
        print(f"ATR(14): {_format_price(atr_price)} price units")
    if signal == "FLAT":
        print("Take profit / stop loss: not applicable to a FLAT signal")
    elif take_profit_price is not None and stop_loss_price is not None:
        target_distance = abs(take_profit_price - reference_price)
        stop_distance = abs(stop_loss_price - reference_price)
        print(
            f"Take profit: {_format_price(take_profit_price)} "
            f"({_format_price(target_distance)} price units / {take_profit_atr:g} x ATR)"
        )
        print(
            f"Stop loss: {_format_price(stop_loss_price)} "
            f"({_format_price(stop_distance)} price units / {stop_loss_atr:g} x ATR)"
        )
    else:
        print("Take profit / stop loss: unavailable")
    if confidence is not None:
        print(f"Jev confidence: {confidence:.1%} (distribution peakedness, not proven market calibration)")
    print(f"Resolved Jev model: {model}")
    if input_tokens is not None:
        print(f"Input tokens: {input_tokens}")
    if cost_usd is not None:
        print(f"OpenRouter request cost: ${cost_usd:.8f}")


def _format_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _format_atr_level(price: float | None, multiple: float | None) -> str:
    if price is None or multiple is None:
        return "—"
    return f"{_format_price(price)} ({multiple:g} ATR)"


def _print_history(rows: Sequence[sqlite3.Row]) -> None:
    if not rows:
        print("No forecasts saved yet.")
        return
    print(
        "id | symbol | requested (UTC) | decision bar (UTC) | signal | probabilities | "
        "target (ATR) | stop (ATR) | forecast outcome | TP/SL outcome"
    )
    for row in rows:
        probabilities = json.loads(row["probabilities_json"])
        probability_text = ", ".join(f"{key}={probabilities[key]:.0%}" for key in SIGNAL_CLASSES)
        if row["realized_class"]:
            forecast_outcome = f"{row['realized_class']} ({float(row['realized_return_pct']):+.3f}%)"
        else:
            forecast_outcome = "pending"
        trade_outcome = row["trade_outcome"] or "pending"
        if row["trade_return_pct"] is not None:
            trade_outcome += f" ({float(row['trade_return_pct']):+.3f}%)"
        cost = f"${float(row['cost_usd']):.8f}" if row["cost_usd"] is not None else "n/a"
        print(
            f"{row['id']} | {row['symbol']} | {row['requested_at_utc']} | {row['decision_time_utc']} | "
            f"{row['signal']} | {probability_text} | "
            f"{_format_atr_level(row['take_profit_price'], row['take_profit_atr'])} | "
            f"{_format_atr_level(row['stop_loss_price'], row['stop_loss_atr'])} | "
            f"{forecast_outcome} | {trade_outcome} | cost {cost}"
        )


def _forecast_command(args: argparse.Namespace) -> int:
    data_key = os.environ.get("TWELVEDATA_API_KEY", "")
    market = args.market
    symbol = args.symbol or ("BTC/USD" if market == "crypto" else "SPY")
    timezone_name = "UTC" if market == "crypto" else "America/New_York"
    db_path = args.db or _market_db_path(market)
    candles = fetch_twelvedata_bars(
        data_key,
        symbol=symbol,
        interval=args.interval,
        outputsize=max(128, args.lookback_bars + args.horizon_bars + 24),
        timezone_name=timezone_name,
        exchange=args.exchange,
    )
    if len(candles) < max(MIN_FEATURE_BARS, args.lookback_bars):
        raise ForecastError(
            f"Need {max(MIN_FEATURE_BARS, args.lookback_bars)} completed candles; Twelve Data returned {len(candles)}"
        )

    ledger_symbol = _ledger_symbol(symbol, args.exchange, candles)
    store = ForecastStore(db_path)
    resolved_count = store.resolve_pending(candles, symbol=ledger_symbol, interval=args.interval)
    if resolved_count:
        print(f"Resolved {resolved_count} earlier forecast(s).")

    decision_time = validate_forecast_window(
        candles,
        horizon_bars=args.horizon_bars,
        max_bar_age_minutes=args.max_bar_age_minutes,
        allow_stale=args.allow_stale,
        market=market,
        symbol=symbol,
    )
    state = build_market_state(
        candles[-args.lookback_bars :],
        symbol=symbol,
        interval=args.interval,
        horizon_bars=args.horizon_bars,
        neutral_band_bps=args.neutral_band_bps,
        state_bars=min(args.state_bars, args.lookback_bars),
        market=market,
    )
    atr_14_bps = float(state["features"]["atr_14_bps"])
    if args.dry_run:
        print(json.dumps(state, indent=2))
        print("Dry run: Jev was not called and no forecast was saved.")
        return 0

    openrouter_api_key, requested_model = openrouter_configuration()
    cached = store.find_cached(
        symbol=ledger_symbol,
        interval=args.interval,
        decision_time=decision_time,
        horizon_bars=args.horizon_bars,
        neutral_band_bps=args.neutral_band_bps,
        lookback_bars=args.lookback_bars,
        requested_model=requested_model,
        atr_choice_config_json=ATR_CHOICE_CONFIG_JSON,
    )
    if cached is not None and not args.force:
        _print_forecast(
            symbol=symbol,
            interval=args.interval,
            horizon_bars=args.horizon_bars,
            decision_time=decision_time,
            signal=cached["signal"],
            probabilities=json.loads(cached["probabilities_json"]),
            confidence=cached["confidence"],
            model=cached["resolved_model"],
            input_tokens=cached["input_tokens"],
            reference_price=float(cached["decision_close"]),
            atr_price=cached["atr_price"],
            take_profit_atr=float(cached["take_profit_atr"]),
            stop_loss_atr=float(cached["stop_loss_atr"]),
            take_profit_price=cached["take_profit_price"],
            stop_loss_price=cached["stop_loss_price"],
            cost_usd=cached["cost_usd"],
            cached=True,
            market=market,
        )
        print(f"Saved in {store.path}")
        return 0

    answer = call_jev(openrouter_api_key, state, model=requested_model)
    if answer.take_profit_atr is None or answer.stop_loss_atr is None:
        raise ForecastError("Jev did not return both TP/SL ATR choices")
    levels = calculate_trade_levels(
        entry_price=candles[-1].close,
        signal=answer.signal,
        atr_14_bps=atr_14_bps,
        take_profit_atr=answer.take_profit_atr,
        stop_loss_atr=answer.stop_loss_atr,
    )
    forecast_id = store.save_forecast(
        symbol=ledger_symbol,
        interval=args.interval,
        decision_time=decision_time,
        decision_close=candles[-1].close,
        horizon_bars=args.horizon_bars,
        neutral_band_bps=args.neutral_band_bps,
        lookback_bars=args.lookback_bars,
        requested_model=requested_model,
        answer=answer,
        state=state,
        atr_price=levels.atr_price,
        take_profit_atr=answer.take_profit_atr,
        stop_loss_atr=answer.stop_loss_atr,
        atr_choice_config_json=ATR_CHOICE_CONFIG_JSON,
        take_profit_price=levels.take_profit_price,
        stop_loss_price=levels.stop_loss_price,
    )
    _print_forecast(
        symbol=symbol,
        interval=args.interval,
        horizon_bars=args.horizon_bars,
        decision_time=decision_time,
        signal=answer.signal,
        probabilities=answer.probabilities,
        confidence=answer.confidence,
        model=answer.model,
        input_tokens=answer.input_tokens,
        reference_price=candles[-1].close,
        atr_price=levels.atr_price,
        take_profit_atr=answer.take_profit_atr,
        stop_loss_atr=answer.stop_loss_atr,
        take_profit_price=levels.take_profit_price,
        stop_loss_price=levels.stop_loss_price,
        cost_usd=answer.cost_usd,
        market=market,
    )
    print(f"Forecast #{forecast_id} saved in {store.path}")
    if args.force:
        print("--force was used; an additional Jev request may incur additional usage.")
    return 0


def _history_command(args: argparse.Namespace) -> int:
    store = ForecastStore(args.db or _market_db_path(args.market))
    _print_history(store.list_forecasts(limit=args.limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="On-demand SPY or BTC 15-minute forecasts using Twelve Data and Jev via OpenRouter."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast = subparsers.add_parser("forecast", help="Fetch candles and generate/reuse a Jev signal")
    forecast.add_argument(
        "--market",
        choices=("equity", "crypto"),
        default="equity",
        help="Market calendar/data timezone (default: equity; crypto runs 24/7 in UTC)",
    )
    forecast.add_argument("--symbol", help="Twelve Data symbol (defaults to SPY or BTC/USD for crypto)")
    forecast.add_argument("--exchange", help="Optional Twelve Data exchange filter, e.g. Binance or Coinbase")
    forecast.add_argument("--interval", choices=tuple(SUPPORTED_INTERVALS), default="15min")
    forecast.add_argument("--horizon-bars", type=int, default=4, help="Future bars to score (default: 4)")
    forecast.add_argument("--lookback-bars", type=int, default=64, help="Completed bars used for features")
    forecast.add_argument("--state-bars", type=int, default=32, help="Recent bars sent to Jev")
    forecast.add_argument(
        "--neutral-band-bps",
        type=float,
        default=10.0,
        help="Return deadband for FLAT (10 bps = 0.10%%; default: 10)",
    )
    forecast.add_argument("--max-bar-age-minutes", type=int, default=30)
    forecast.add_argument("--allow-stale", action="store_true", help="Allow a stale bar for research")
    forecast.add_argument("--force", action="store_true", help="Make another Jev call for the same bar")
    forecast.add_argument("--dry-run", action="store_true", help="Fetch data and print state without calling Jev")
    forecast.add_argument("--db", type=Path, default=None, help="Local SQLite forecast ledger")
    forecast.set_defaults(handler=_forecast_command)

    history = subparsers.add_parser("history", help="Show saved forecasts and resolved outcomes")
    history.add_argument("--market", choices=("equity", "crypto"), default="equity")
    history.add_argument("--limit", type=int, default=20)
    history.add_argument("--db", type=Path, default=None, help="Local SQLite forecast ledger")
    history.set_defaults(handler=_history_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        load_env_file(Path(__file__).with_name(".env"))
    except ForecastError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "horizon_bars", 1) < 1:
        parser.error("--horizon-bars must be at least 1")
    if getattr(args, "lookback_bars", MIN_FEATURE_BARS) < MIN_FEATURE_BARS:
        parser.error(f"--lookback-bars must be at least {MIN_FEATURE_BARS}")
    if getattr(args, "state_bars", 1) < 1:
        parser.error("--state-bars must be at least 1")
    if getattr(args, "max_bar_age_minutes", 1) < 1:
        parser.error("--max-bar-age-minutes must be at least 1")
    if getattr(args, "limit", 1) < 1:
        parser.error("--limit must be at least 1")
    try:
        return args.handler(args)
    except ForecastError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        print(f"Database error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
