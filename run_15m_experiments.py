#!/usr/bin/env python3
"""
Run a small set of experiments on a 15m OHLCV dataset using baseline.py.

This orchestrates multiple runs (thresholds, horizons, calibration, optional shorts),
collects metrics, and writes a compact summary CSV.

Usage:
  python3 run_15m_experiments.py --data path/to/your_15m.csv \
      --outdir ./runs/15m_sweep --fee_bps 1.0 --seed 42 --plot

Dependencies: stdlib + baseline.py in the same repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Exp:
    name: str
    args: List[str]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> int:
    try:
        print("$", " ".join(shlex.quote(c) for c in cmd))
        res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
        return res.returncode
    except Exception as e:
        print(f"[ERR] Failed: {e}")
        return 1


def discover_python() -> str:
    return sys.executable or "python3"


def make_experiments(
    data: Path,
    base_outdir: Path,
    fee_bps: float,
    seed: int,
    plot: bool,
    horizons: List[int],
    thresholds: List[float],
    include_nocal: bool,
    include_short: bool,
    auto_thresh: bool,
    thresh_min: float,
    thresh_max: float,
    thresh_step: float,
    min_hold: int,
    thresh_exit: Optional[float],
) -> List[Exp]:
    py = discover_python()
    baseline = str(Path(__file__).parent / "baseline.py")

    common = [
        py,
        baseline,
        "--data",
        str(data),
        "--val_ratio",
        "0.15",
        "--test_ratio",
        "0.15",
        "--fee_bps",
        str(fee_bps),
        "--seed",
        str(seed),
        "--n_estimators",
        "200",
        "--max_depth",
        "2",
        "--min_samples_leaf",
        "80",
        "--learning_rate",
        "0.05",
        "--subsample",
        "0.9",
    ]
    if plot:
        common.append("--plot")

    exps: List[Exp] = []

    # Generate long-only calibrated and optionally other variants
    for horizon in horizons:
        if auto_thresh:
            # Calibrated, auto threshold
            name_cal = f"h{horizon}_auto_cal_long_only"
            outdir_cal = base_outdir / name_cal
            args_cal = common + [
                "--horizon", str(horizon),
                "--auto_thresh",
                "--thresh_min", f"{thresh_min:.2f}",
                "--thresh_max", f"{thresh_max:.2f}",
                "--thresh_step", f"{thresh_step:.2f}",
                "--min_hold", str(min_hold),
                "--calibrate",
                "--outdir", str(outdir_cal),
            ]
            if thresh_exit is not None:
                args_cal += ["--thresh_exit", f"{thresh_exit:.2f}"]
            exps.append(Exp(name=name_cal, args=args_cal))

            if include_nocal:
                name_nc = f"h{horizon}_auto_nocal_long_only"
                outdir_nc = base_outdir / name_nc
                args_nc = common + [
                    "--horizon", str(horizon),
                    "--auto_thresh",
                    "--thresh_min", f"{thresh_min:.2f}",
                    "--thresh_max", f"{thresh_max:.2f}",
                    "--thresh_step", f"{thresh_step:.2f}",
                    "--min_hold", str(min_hold),
                    "--outdir", str(outdir_nc),
                ]
                if thresh_exit is not None:
                    args_nc += ["--thresh_exit", f"{thresh_exit:.2f}"]
                exps.append(Exp(name=name_nc, args=args_nc))

            if include_short:
                name_short = f"h{horizon}_auto_cal_long_short"
                outdir_short = base_outdir / name_short
                args_short = common + [
                    "--horizon", str(horizon),
                    "--auto_thresh",
                    "--thresh_min", f"{thresh_min:.2f}",
                    "--thresh_max", f"{thresh_max:.2f}",
                    "--thresh_step", f"{thresh_step:.2f}",
                    "--min_hold", str(min_hold),
                    "--calibrate",
                    "--allow_short",
                    "--outdir", str(outdir_short),
                ]
                if thresh_exit is not None:
                    args_short += ["--thresh_exit", f"{thresh_exit:.2f}"]
                exps.append(Exp(name=name_short, args=args_short))
        else:
            for thresh in thresholds:
                # Calibrated with fixed threshold
                name_cal = f"h{horizon}_t{int(thresh*100):03d}_cal_long_only"
                outdir_cal = base_outdir / name_cal
                args_cal = common + [
                    "--horizon", str(horizon),
                    "--thresh", f"{thresh:.2f}",
                    "--min_hold", str(min_hold),
                    "--calibrate",
                    "--outdir", str(outdir_cal),
                ]
                if thresh_exit is not None:
                    args_cal += ["--thresh_exit", f"{thresh_exit:.2f}"]
                exps.append(Exp(name=name_cal, args=args_cal))

                if include_nocal:
                    name_nc = f"h{horizon}_t{int(thresh*100):03d}_nocal_long_only"
                    outdir_nc = base_outdir / name_nc
                    args_nc = common + [
                        "--horizon", str(horizon),
                        "--thresh", f"{thresh:.2f}",
                        "--min_hold", str(min_hold),
                        "--outdir", str(outdir_nc),
                    ]
                    if thresh_exit is not None:
                        args_nc += ["--thresh_exit", f"{thresh_exit:.2f}"]
                    exps.append(Exp(name=name_nc, args=args_nc))

                if include_short:
                    name_short = f"h{horizon}_t{int(thresh*100):03d}_cal_long_short"
                    outdir_short = base_outdir / name_short
                    args_short = common + [
                        "--horizon", str(horizon),
                        "--thresh", f"{thresh:.2f}",
                        "--min_hold", str(min_hold),
                        "--calibrate",
                        "--allow_short",
                        "--outdir", str(outdir_short),
                    ]
                    if thresh_exit is not None:
                        args_short += ["--thresh_exit", f"{thresh_exit:.2f}"]
                    exps.append(Exp(name=name_short, args=args_short))

    return exps


def collect_summary(base_outdir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for run_dir in sorted(p for p in base_outdir.iterdir() if p.is_dir()):
        m = run_dir / "metrics.json"
        p = run_dir / "params.json"
        if not m.exists() or not p.exists():
            continue
        try:
            metrics = json.loads(m.read_text())
            params = json.loads(p.read_text())
        except Exception:
            continue
        rows.append(
            {
                "run": run_dir.name,
                "horizon": params.get("horizon"),
                "thresh": params.get("thresh"),
                "allow_short": params.get("allow_short"),
                "calibrate": params.get("calibrate"),
                "auc_test": metrics.get("auc_test"),
                "sharpe": metrics.get("sharpe"),
                "cum_return": metrics.get("cum_return"),
                "n_trades": metrics.get("n_trades"),
                "hit_rate": metrics.get("hit_rate")
                or (metrics.get("hit_rate_long"), metrics.get("hit_rate_short")),
            }
        )
    return rows


def save_summary_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "run",
        "horizon",
        "thresh",
        "allow_short",
        "calibrate",
        "auc_test",
        "sharpe",
        "cum_return",
        "n_trades",
        "hit_rate",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run 15m experiments with baseline.py")
    ap.add_argument("--data", required=True, help="Path to 15m OHLCV CSV")
    ap.add_argument(
        "--outdir",
        default="./runs/15m_sweep",
        help="Base output folder to store experiment runs and summary",
    )
    ap.add_argument("--fee_bps", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[4, 6, 8],
        help="Horizons (in bars) to test, default: 4 6 8",
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60],
        help="Probability thresholds to test",
    )
    ap.add_argument(
        "--include_nocal",
        action="store_true",
        help="Also run non-calibrated variants",
    )
    ap.add_argument(
        "--include_short",
        action="store_true",
        help="Also run long+short variants for each horizon/threshold",
    )
    ap.add_argument("--auto_thresh", action="store_true", help="Enable validation-based threshold tuning")
    ap.add_argument("--thresh_min", type=float, default=0.52)
    ap.add_argument("--thresh_max", type=float, default=0.65)
    ap.add_argument("--thresh_step", type=float, default=0.01)
    ap.add_argument("--min_hold", type=int, default=2)
    ap.add_argument("--thresh_exit", type=float, default=None)
    args = ap.parse_args(argv)

    data = Path(args.data)
    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    exps = make_experiments(
        data,
        base_outdir,
        args.fee_bps,
        args.seed,
        args.plot,
        args.horizons,
        args.thresholds,
        args.include_nocal,
        args.include_short,
        args.auto_thresh,
        args.thresh_min,
        args.thresh_max,
        args.thresh_step,
        args.min_hold,
        args.thresh_exit,
    )
    print(f"[INFO] Running {len(exps)} experiments, writing under {base_outdir}")

    for exp in exps:
        rc = run_cmd(exp.args)
        if rc != 0:
            print(f"[WARN] Experiment {exp.name} failed with code {rc}")

    rows = collect_summary(base_outdir)
    if not rows:
        print("[WARN] No results collected. Check run outputs.")
        return 1

    # Sort by Sharpe desc (None last)
    rows.sort(key=lambda r: (r["sharpe"] is None, -(r["sharpe"] or -1e9)))
    for r in rows:
        print(
            f"{r['run']:28s}  H={r['horizon']:<2} thr={r['thresh']:.2f}  "
            f"short={bool(r['allow_short'])!s:<5}  auc={r['auc_test']:.3f}  "
            f"sharpe={r['sharpe']}  ret={r['cum_return']}  trades={r['n_trades']}  hit={r['hit_rate']}"
        )

    save_summary_csv(rows, base_outdir / "summary.csv")
    print(f"[INFO] Summary saved to {base_outdir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
