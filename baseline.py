#!/usr/bin/env python3
"""
End-to-end baseline: train a predictive model on OHLCV candlesticks and run a simple backtest.

- Dependencies: python>=3.10, pandas, numpy, scikit-learn, matplotlib (optional behind --plot).
- Implements a CLI pipeline with clear functions, minimal deps, and robust checks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression


# ------------------------- Utilities & Config -------------------------


@dataclass
class DataSummary:
    start_timestamp: str
    end_timestamp: str
    n_bars: int
    bar_seconds: float
    class_balance: float  # positive class ratio in full dataset


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


# ------------------------- 1) load_data -------------------------


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load CSV with columns: timestamp, open, high, low, close, volume.
    - Parse dtypes; sort by timestamp; drop duplicates.
    - Validate monotonic timestamps, uniform time delta, and non-negative prices/volumes.
    """
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(path)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Enforce dtypes
    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="raise")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort and drop duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Basic sanity checks
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be strictly increasing after sorting/dupe removal.")
    if (df[["open", "high", "low", "close", "volume"]] < 0).any().any():
        raise ValueError("Negative prices or volumes detected.")
    if (df["high"] < df["low"]).any():
        raise ValueError("High < Low found.")

    # Uniform time delta check (allow one delta; ignore first NaT)
    deltas = df["timestamp"].diff().iloc[1:]
    if deltas.isnull().any():
        raise ValueError("Null time deltas encountered unexpectedly.")
    uniq = deltas.dt.total_seconds().round().unique()
    if len(uniq) != 1:
        raise ValueError(f"Non-uniform time delta detected. Unique deltas (s): {uniq}")

    return df


# ------------------------- 2) make_features -------------------------


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    pb = (series - lower) / ((upper - lower) + 1e-12)
    return pb


def make_features(df: pd.DataFrame, H: int) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series, List[str]]:
    """
    Compute features from past-only information. Return:
    - X: DataFrame of features
    - y: numpy array of binary labels: 1{ close[t+H] / close[t] - 1 > 0 }
    - prices: Series of close prices aligned with X/y
    - times: Series of timestamps aligned with X/y
    - feature_names: List[str]
    """
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    ts = df["timestamp"]

    # Primary return
    ret1 = np.log(close).diff()

    feats = pd.DataFrame(index=df.index)
    feats["ret1"] = ret1

    # Rolling stats of ret1
    windows = [3, 6, 12, 24]
    for w in windows:
        r = ret1.rolling(w)
        feats[f"ret1_mean_{w}"] = r.mean()
        feats[f"ret1_std_{w}"] = r.std()
        feats[f"ret1_min_{w}"] = r.min()
        feats[f"ret1_max_{w}"] = r.max()

    # Candle shape features
    feats["body_pct"] = (close - open_) / (open_ + 1e-12)
    feats["range_pct"] = (high - low) / (close + 1e-12)
    feats["upper_wick"] = (high - close).clip(lower=0) / (close + 1e-12)
    feats["lower_wick"] = (close - low).clip(lower=0) / (close + 1e-12)

    # Volume features
    v_windows = [6, 12, 24]
    for w in v_windows:
        vmean = vol.rolling(w).mean()
        vstd = vol.rolling(w).std()
        feats[f"vol_mean_{w}"] = vmean
        feats[f"vol_std_{w}"] = vstd
        feats[f"vol_z_{w}"] = (vol - vmean) / (vstd + 1e-12)

    # RSI, MACD, Bollinger %B
    feats["rsi14"] = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close, 12, 26, 9)
    feats["macd"] = macd_line
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd_hist
    feats["pct_b"] = bollinger_percent_b(close, 20, 2.0)

    # Label: future return over horizon H
    future_ret = (close.shift(-H) / close - 1.0)
    y = (future_ret > 0).astype(int)

    # Drop warmup NaNs and align
    all_cols = feats.columns.tolist()
    valid_mask = feats.notnull().all(axis=1) & y.notnull()
    feats = feats.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int).values
    prices = close.loc[valid_mask]
    times = ts.loc[valid_mask]

    # Assert alignment
    assert len(feats) == len(y) == len(prices) == len(times), "Alignment error in features/labels."

    return feats, y, prices, times, all_cols


# ------------------------- 3) time_split -------------------------


def time_split(times: pd.Series, val_ratio: float, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-based split indices (chronological).
    Returns (train_idx, val_idx, test_idx) as integer arrays.
    """
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be in (0,1) and sum < 1.")

    n = len(times)
    i_test = int(n * (1 - test_ratio))
    i_val = int(i_test * (1 - val_ratio))

    idx = np.arange(n, dtype=int)
    train_idx = idx[:i_val]
    val_idx = idx[i_val:i_test]
    test_idx = idx[i_test:]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Time split produced empty set(s). Adjust ratios or provide more data.")
    return train_idx, val_idx, test_idx


# ------------------------- 4) train_model -------------------------


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
    *,
    learning_rate: float = 0.05,
    n_estimators: int = 300,
    max_depth: int = 3,
    subsample: float = 0.9,
    min_samples_leaf: int = 20,
) -> Tuple[GradientBoostingClassifier, Dict[str, float | None]]:
    """
    Train GradientBoostingClassifier on train; report AUCs for train/val/test.
    """
    train_idx, val_idx, test_idx = split
    model = GradientBoostingClassifier(
        random_state=seed,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        max_features=None,
    )
    model.fit(X.iloc[train_idx], y[train_idx])

    # AUCs
    p_tr = model.predict_proba(X.iloc[train_idx])[:, 1]
    p_va = model.predict_proba(X.iloc[val_idx])[:, 1]
    p_te = model.predict_proba(X.iloc[test_idx])[:, 1]

    aucs = {
        "auc_train": auc_safe(y[train_idx], p_tr),
        "auc_val": auc_safe(y[val_idx], p_va),
        "auc_test": auc_safe(y[test_idx], p_te),
    }
    return model, aucs


# ------------------------- 5) predict_all -------------------------


def predict_all(model: GradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Predict probabilities in [0,1] for all rows.
    """
    return model.predict_proba(X)[:, 1]


# ------------------------- 6) backtest -------------------------


def compute_trade_stats(
    times: pd.Series,
    rets: pd.Series,
    pos: np.ndarray,
    side: int,  # 1 for long, -1 for short
) -> Tuple[float | None, int]:
    """
    Compute trade-level hit rate (fraction of trades with positive cumulative return)
    and number of trades for the given side.
    """
    if side not in (1, -1):
        return None, 0

    pos_side = (pos == side).astype(int)
    entries = (np.roll(pos_side, 1) == 0) & (pos_side == 1)
    exits = (np.roll(pos_side, 1) == 1) & (pos_side == 0)
    # Map trades by accumulating pnl between entry and exit.
    # Ensure first entry starts after index 0
    trade_pnls = []
    in_trade = False
    pnl_acc = 0.0
    for i in range(len(rets)):
        if entries[i] and not in_trade:
            in_trade = True
            pnl_acc = 0.0
        if in_trade:
            pnl_acc += float(rets.iloc[i] * side)
        if exits[i] and in_trade:
            trade_pnls.append(pnl_acc)
            in_trade = False
            pnl_acc = 0.0
    # If trade open at end, close it
    if in_trade:
        trade_pnls.append(pnl_acc)
    if len(trade_pnls) == 0:
        return None, 0
    hits = sum(1 for x in trade_pnls if x > 0)
    return hits / len(trade_pnls), len(trade_pnls)


def backtest(
    proba: np.ndarray,
    y: np.ndarray,
    prices: pd.Series,
    test_idx: np.ndarray,
    thresh: float,
    fee_bps: float,
    allow_short: bool = False,
    thresh_exit: float | None = None,
    min_hold: int = 0,
    times: pd.Series | None = None,
) -> Tuple[Dict[str, float | None], pd.DataFrame]:
    """
    Long-only when p > thresh, optional short when p < 1 - thresh.
    - Per-bar return = close[t+1]/close[t]-1
    - Apply transaction cost on signal flips: fee_bps/1e4 times abs position change.
    - Compute equity curve on test segment, Sharpe (annualized with 252), hit rate, cumulative return, trade count.
    """
    assert 0 < thresh < 1, "thresh must be in (0,1)"
    fee = float(fee_bps) / 1e4
    if thresh_exit is None:
        thresh_exit = thresh
    thresh_exit = float(thresh_exit)
    min_hold = int(min_hold)

    # Returns aligned so r[t] is next-bar return from t to t+1
    r = prices.pct_change().shift(-1)
    # Signals at time t are used for return r[t]
    # Build positions with hysteresis and minimum hold
    pos = np.zeros_like(proba, dtype=float)
    hold = 0
    for i in range(len(proba)):
        p = proba[i]
        cur = pos[i - 1] if i > 0 else 0.0
        new = cur
        if cur == 0.0:
            if p > thresh:
                new = 1.0
                hold = 1
            elif allow_short and p < (1.0 - thresh):
                new = -1.0
                hold = 1
            else:
                hold = 0
        elif cur == 1.0:
            if hold >= min_hold and p < thresh_exit:
                new = 0.0
                hold = 0
            else:
                hold += 1
        elif cur == -1.0:
            if hold >= min_hold and p > (1.0 - thresh_exit):
                new = 0.0
                hold = 0
            else:
                hold += 1
        pos[i] = new

    # Restrict to test window
    mask = np.zeros_like(proba, dtype=bool)
    mask[test_idx] = True
    r_test = r[mask].dropna()  # last bar may be NaN due to shift
    pos_test = pos[mask][: len(r_test)]
    selected_idx = np.flatnonzero(mask)[: len(r_test)]
    if times is None:
        times_test = prices.index[selected_idx]
    else:
        times_test = pd.DatetimeIndex(times.iloc[selected_idx])

    # Transaction costs on position changes
    pos_shift = np.roll(pos_test, 1)
    pos_shift[0] = 0.0
    pos_change = np.abs(pos_test - pos_shift)  # 1 for enter/exit, 2 for flip
    costs = fee * pos_change

    pnl = (pos_test * r_test.values) - costs
    equity = (1.0 + pd.Series(pnl, index=times_test)).cumprod()
    cum_return = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0

    # Sharpe (assume 252 bars/year; adjust if needed)
    sharpe = None
    if np.std(pnl) > 0:
        sharpe = float((np.mean(pnl) / (np.std(pnl) + 1e-12)) * math.sqrt(252))

    # Trade stats
    # Entries are transitions from 0 -> nonzero
    entries = ((pos_shift == 0) & (pos_test != 0)).sum()
    n_trades = int(entries)

    # Hit rates per side
    long_hit, long_trades = compute_trade_stats(pd.Series(times_test), pd.Series(r_test.values, index=times_test), pos_test, side=1)
    short_hit, short_trades = (None, 0)
    if allow_short:
        short_hit, short_trades = compute_trade_stats(pd.Series(times_test), pd.Series(r_test.values, index=times_test), pos_test, side=-1)

    metrics = {
        "sharpe": sharpe,
        "hit_rate": long_hit if not allow_short else None,
        "hit_rate_long": long_hit if allow_short else None,
        "hit_rate_short": short_hit if allow_short else None,
        "cum_return": cum_return,
        "n_trades": n_trades,
        "n_trades_long": long_trades if allow_short else None,
        "n_trades_short": short_trades if allow_short else None,
    }

    eq_df = pd.DataFrame({"timestamp": pd.to_datetime(times_test), "equity": equity.values})
    return metrics, eq_df


def tune_threshold(
    proba: np.ndarray,
    prices: pd.Series,
    idx: np.ndarray,
    fee_bps: float,
    allow_short: bool,
    thresh_min: float,
    thresh_max: float,
    thresh_step: float,
    thresh_exit: float | None,
    min_hold: int,
    times: pd.Series | None = None,
) -> float:
    """Grid-search threshold on given index to maximize Sharpe.
    Falls back to best cumulative return if all Sharpes are None.
    """
    best_thr = None
    best_sharpe = -1e18
    best_ret = -1e18
    t = float(thresh_min)
    while t <= thresh_max + 1e-12:
        metrics, _ = backtest(
            proba=proba,
            y=np.zeros(len(proba)),
            prices=prices,
            test_idx=idx,
            thresh=t,
            fee_bps=fee_bps,
            allow_short=allow_short,
            thresh_exit=thresh_exit if thresh_exit is not None else t,
            min_hold=min_hold,
            times=times,
        )
        sh = metrics.get("sharpe")
        cr = metrics.get("cum_return") or -1e18
        if sh is not None and (best_thr is None or sh > best_sharpe):
            best_thr = t
            best_sharpe = sh
            best_ret = cr
        elif sh is None and best_thr is None:
            if cr > best_ret:
                best_thr = t
                best_ret = cr
        t = round(t + thresh_step, 10)
    return float(best_thr if best_thr is not None else thresh_min)


# ------------------------- 7) save_artifacts -------------------------


def save_artifacts(
    outdir: Path,
    metrics: Dict[str, float | None],
    params: Dict,
    eq_df: pd.DataFrame,
    model: GradientBoostingClassifier,
    feature_names: List[str],
    plot: bool,
) -> None:
    # Ensure output directory exists (handles nested paths like outdir/selftest)
    outdir = ensure_outdir(outdir)
    # metrics.json
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # params.json
    (outdir / "params.json").write_text(json.dumps(params, indent=2, default=str))

    # equity_curve.csv
    eq_df.to_csv(outdir / "equity_curve.csv", index=False)

    # feature_importance.csv if available
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_.tolist()})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(outdir / "feature_importance.csv", index=False)

    # optional plot
    if plot:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4))
            plt.plot(eq_df["timestamp"], eq_df["equity"], label="Equity")
            plt.title("Equity Curve (Test)")
            plt.xlabel("Time")
            plt.ylabel("Equity")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "equity_curve.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}", file=sys.stderr)


# ------------------------- Main -------------------------


def tiny_model_card(
    ds: DataSummary,
    aucs: Dict[str, float | None],
    metrics: Dict[str, float | None],
    feature_names: List[str],
    model: GradientBoostingClassifier,
    k: int = 10,
) -> None:
    print("\n=== Model Card ===")
    print(f"Data span: {ds.start_timestamp} -> {ds.end_timestamp}")
    print(f"Bars: {ds.n_bars} | Bar seconds: {ds.bar_seconds:.0f}")
    print(f"Class balance (positive rate): {ds.class_balance:.3f}")
    print(f"AUCs: train={aucs.get('auc_train')}, val={aucs.get('auc_val')}, test={aucs.get('auc_test')}")
    print(f"Test Sharpe: {metrics.get('sharpe')}, CumRet: {metrics.get('cum_return')}, Trades: {metrics.get('n_trades')}")
    if hasattr(model, "feature_importances_"):
        imps = np.asarray(model.feature_importances_)
        top_idx = np.argsort(-imps)[:k]
        print("Top features:")
        for i in top_idx:
            print(f"  - {feature_names[i]}: {imps[i]:.4f}")
    print("==================\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ML model on OHLCV and backtest.")
    parser.add_argument("--data", type=str, help="Path to CSV with OHLCV.", required=False)
    parser.add_argument("--horizon", dest="horizon", type=int, default=5)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--thresh", type=float, default=0.55)
    parser.add_argument("--fee_bps", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./runs")
    parser.add_argument("--calibrate", action="store_true", help="Isotonic calibration on validation set.")
    parser.add_argument("--allow_short", action="store_true", help="Allow shorting when p < 1 - thresh.")
    parser.add_argument("--selftest", action="store_true", help="Run synthetic data self-test.")
    # Trading rule controls
    parser.add_argument("--thresh_exit", type=float, default=None, help="Exit threshold (defaults to --thresh).")
    parser.add_argument("--min_hold", type=int, default=0, help="Minimum bars to hold a position before exit.")
    # Auto threshold tuning on validation
    parser.add_argument("--auto_thresh", action="store_true", help="Tune threshold on validation set.")
    parser.add_argument("--thresh_min", type=float, default=0.52)
    parser.add_argument("--thresh_max", type=float, default=0.65)
    parser.add_argument("--thresh_step", type=float, default=0.01)
    # Model hyperparameters
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    args = parser.parse_args(argv)

    set_all_seeds(args.seed)
    outdir = ensure_outdir(args.outdir)

    if args.selftest:
        print("[SelfTest] Generating synthetic OHLCV...")
        synth_path = Path(outdir) / "synthetic.csv"
        df_synth = generate_synthetic_ohlcv(n=2000, seed=args.seed)
        df_synth.to_csv(synth_path, index=False)
        print("[SelfTest] Running pipeline on synthetic data...")
        retcode = run_pipeline(
            data_path=synth_path,
            horizon=args.horizon,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            thresh=args.thresh,
            fee_bps=args.fee_bps,
            plot=args.plot,
            seed=args.seed,
            outdir=outdir / "selftest",
            calibrate=args.calibrate,
            allow_short=args.allow_short,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            subsample=args.subsample,
            min_samples_leaf=args.min_samples_leaf,
            thresh_exit=args.thresh_exit,
            min_hold=args.min_hold,
            auto_thresh=args.auto_thresh,
            thresh_min=args.thresh_min,
            thresh_max=args.thresh_max,
            thresh_step=args.thresh_step,
        )
        # Assert artifacts
        assert (outdir / "selftest" / "metrics.json").exists(), "metrics.json missing"
        assert (outdir / "selftest" / "equity_curve.csv").exists(), "equity_curve.csv missing"
        assert (outdir / "selftest" / "params.json").exists(), "params.json missing"
        print("[SelfTest] OK.")
        return retcode

    if not args.data:
        print("Error: --data PATH is required (or use --selftest).", file=sys.stderr)
        return 2

    return run_pipeline(
        data_path=Path(args.data),
        horizon=args.horizon,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        thresh=args.thresh,
        fee_bps=args.fee_bps,
        plot=args.plot,
        seed=args.seed,
        outdir=outdir,
        calibrate=args.calibrate,
        allow_short=args.allow_short,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_samples_leaf=args.min_samples_leaf,
        thresh_exit=args.thresh_exit,
        min_hold=args.min_hold,
        auto_thresh=args.auto_thresh,
        thresh_min=args.thresh_min,
        thresh_max=args.thresh_max,
        thresh_step=args.thresh_step,
    )


def run_pipeline(
    data_path: Path,
    horizon: int,
    test_ratio: float,
    val_ratio: float,
    thresh: float,
    fee_bps: float,
    plot: bool,
    seed: int,
    outdir: Path,
    calibrate: bool,
    allow_short: bool,
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
    subsample: float,
    min_samples_leaf: int,
    thresh_exit: float | None = None,
    min_hold: int = 0,
    auto_thresh: bool = False,
    thresh_min: float = 0.52,
    thresh_max: float = 0.65,
    thresh_step: float = 0.01,
) -> int:
    print("[1/7] Loading data...")
    df = load_data(data_path)

    print("[2/7] Building features...")
    X, y, prices, times, feature_names = make_features(df, horizon)

    print("[3/7] Time-based split...")
    train_idx, val_idx, test_idx = time_split(times, val_ratio=val_ratio, test_ratio=test_ratio)

    # Data summary
    deltas = times.diff().dropna().dt.total_seconds()
    bar_seconds = float(deltas.iloc[0]) if len(deltas) else float("nan")
    ds = DataSummary(
        start_timestamp=str(times.iloc[0]),
        end_timestamp=str(times.iloc[-1]),
        n_bars=int(len(times)),
        bar_seconds=bar_seconds,
        class_balance=float(np.mean(y)),
    )

    print("[4/7] Training model...")
    model, aucs = train_model(
        X,
        y,
        (train_idx, val_idx, test_idx),
        seed=seed,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
    )

    print("[5/7] Predicting probabilities...")
    proba = predict_all(model, X)

    if calibrate:
        print("[5b] Calibrating with isotonic on validation set...")
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba[val_idx], y[val_idx])
        proba = ir.transform(proba)

        # Recompute AUCs post-calibration for reporting
        aucs = {
            "auc_train": auc_safe(y[train_idx], proba[train_idx]),
            "auc_val": auc_safe(y[val_idx], proba[val_idx]),
            "auc_test": auc_safe(y[test_idx], proba[test_idx]),
        }

    # Optional threshold tuning on validation set
    tuned_thresh = thresh
    if auto_thresh:
        print("[5c] Tuning threshold on validation set...")
        tuned_thresh = tune_threshold(
            proba=proba,
            prices=prices,
            idx=val_idx,
            fee_bps=fee_bps,
            allow_short=allow_short,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
            thresh_step=thresh_step,
            thresh_exit=thresh_exit if thresh_exit is not None else thresh,
            min_hold=min_hold,
            times=times,
        )
        print(f"[5c] Tuned threshold: {tuned_thresh:.4f}")

    print("[6/7] Backtesting on test set...")
    metrics_bt, eq_df = backtest(
        proba=proba,
        y=y,
        prices=prices,
        test_idx=test_idx,
        thresh=tuned_thresh,
        fee_bps=fee_bps,
        allow_short=allow_short,
        thresh_exit=thresh_exit if thresh_exit is not None else tuned_thresh,
        min_hold=min_hold,
        times=times,
    )

    # Merge metrics
    metrics = {**aucs, **metrics_bt}

    print("[7/7] Saving artifacts...")
    params = {
        "data_path": str(data_path),
        "horizon": horizon,
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "thresh": tuned_thresh,
        "fee_bps": fee_bps,
        "seed": seed,
        "allow_short": allow_short,
        "calibrate": calibrate,
        "trading_params": {
            "thresh_exit": thresh_exit if thresh_exit is not None else tuned_thresh,
            "min_hold": min_hold,
            "auto_thresh": auto_thresh,
            "thresh_min": thresh_min,
            "thresh_max": thresh_max,
            "thresh_step": thresh_step,
        },
        "model_params": {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "subsample": subsample,
            "min_samples_leaf": min_samples_leaf,
        },
        "data_summary": asdict(ds),
    }
    save_artifacts(
        outdir=outdir,
        metrics=metrics,
        params=params,
        eq_df=eq_df,
        model=model,
        feature_names=feature_names,
        plot=plot,
    )

    tiny_model_card(ds, aucs, metrics, feature_names, model)
    print("Done.")
    return 0


# ------------------------- SelfTest Synthetic Data -------------------------


def generate_synthetic_ohlcv(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a simple GBM-like OHLCV series at 1h bars.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252  # daily step proxy
    mu, sigma = 0.10, 0.25
    s0 = 100.0

    shocks = rng.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * math.sqrt(dt), size=n)
    prices = np.empty(n)
    prices[0] = s0
    for t in range(1, n):
        prices[t] = prices[t - 1] * math.exp(shocks[t])

    # Build OHLC with simple intrabar variation
    close = prices
    spread = np.maximum(0.001, rng.lognormal(mean=-4, sigma=0.3, size=n))
    high = close * (1 + spread * rng.uniform(0.2, 1.0, size=n))
    low = close * (1 - spread * rng.uniform(0.2, 1.0, size=n))
    open_ = close * (1 + rng.normal(0, 0.0005, size=n))
    volume = rng.lognormal(mean=10, sigma=0.3, size=n)

    # timestamps hourly
    start = pd.Timestamp("2020-01-01 00:00:00")
    ts = pd.date_range(start, periods=n, freq="H")

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": np.maximum.reduce([open_, high, close]),
            "low": np.minimum.reduce([open_, low, close]),
            "close": close,
            "volume": volume,
        }
    )
    return df


# ------------------------- Entrypoint -------------------------

if __name__ == "__main__":
    sys.exit(main())
