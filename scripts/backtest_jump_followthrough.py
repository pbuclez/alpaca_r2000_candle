#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.follow_through_backtest import BacktestConfig, parse_horizons, run_dataset_backtest


DEFAULT_HORIZONS = "1m,5m,15m,30m,1h,5h,eod"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest whether stocks follow through or revert after a rolling "
            "5-minute jump."
        )
    )
    parser.add_argument("--data", default="data/raw/r2000_1min", help="Parquet dataset directory.")
    parser.add_argument("--jump-pct", type=float, default=3.0, help="Jump threshold percent.")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=5,
        help="Lookback window in 1-minute bars.",
    )
    parser.add_argument(
        "--horizons",
        default=DEFAULT_HORIZONS,
        help="Comma-separated horizons, for example 1m,5m,1h,5h,eod.",
    )
    parser.add_argument(
        "--cooldown-minutes",
        type=int,
        default=5,
        help="Minimum bars before another signal can fire for the same symbol/day.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Count every qualifying bar instead of applying signal cooldown.",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol filter, for example SMCI,CAVA.",
    )
    parser.add_argument("--start", default=None, help="Optional start date filter, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date filter, YYYY-MM-DD.")
    parser.add_argument("--signals-out", default=None, help="Optional CSV path for signal-level rows.")
    parser.add_argument("--summary-out", default=None, help="Optional CSV or JSON path for summary rows.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N Parquet files. Use 0 to disable progress output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = _resolve(Path(args.data))
    horizons = parse_horizons(args.horizons)
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    cooldown = 0 if args.allow_overlap else args.cooldown_minutes
    config = BacktestConfig(
        data_dir=data_dir,
        jump_pct=args.jump_pct,
        window_minutes=args.window_minutes,
        horizons=horizons,
        cooldown_minutes=cooldown,
        symbols=symbols,
        start=args.start,
        end=args.end,
    )

    print(f"Starting backtest in {data_dir}", flush=True)
    print(f"Jump rule: close / close.shift({args.window_minutes}) - 1 >= {args.jump_pct:.2f}%", flush=True)
    print(f"Horizons: {', '.join(horizon.label for horizon in horizons)}", flush=True)
    print(f"Cooldown minutes: {cooldown}", flush=True)
    result = run_dataset_backtest(
        config,
        progress_callback=_progress_callback(args.progress_every),
    )
    signals = result.signals
    summary = result.summary

    print(f"Data directory: {data_dir}")
    print(f"Parquet files scanned: {result.file_count:,}")
    print(f"Bars loaded: {result.row_count:,}")
    print(f"Unique symbols: {len(result.symbols):,}")
    print(f"Trading days: {len(result.dates):,}")
    print()
    print(_format_summary(summary))

    if args.signals_out:
        path = _resolve(Path(args.signals_out))
        path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(path, index=False)
        print(f"\nSignal rows written to {path}")

    if args.summary_out:
        path = _resolve(Path(args.summary_out))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(_json_records(summary), indent=2), encoding="utf-8")
        else:
            summary.to_csv(path, index=False)
        print(f"Summary written to {path}")

    return 0


def _format_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No signals found."

    formatted = summary.copy()
    for column in ["win_rate", "avg_future_return", "median_future_return"]:
        formatted[column] = formatted[column].map(_format_pct)
    formatted["follow_revert_ratio"] = formatted["follow_revert_ratio"].map(_format_float)
    formatted["signal_frequency_per_day"] = formatted["signal_frequency_per_day"].map(_format_float)
    return formatted.to_string(index=False)


def _format_pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_float(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def _json_records(summary: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in summary.to_dict(orient="records"):
        record: dict[str, object] = {}
        for key, value in row.items():
            if value is None or pd.isna(value):
                record[key] = None
            elif isinstance(value, float) and math.isinf(value):
                record[key] = "inf" if value > 0 else "-inf"
            else:
                record[key] = value
        records.append(record)
    return records


def _progress_callback(progress_every: int):
    if progress_every <= 0:
        return None

    def report(file_index: int, total_files: int, path: Path, row_count: int, signal_rows: int) -> None:
        if file_index == 1 or file_index == total_files or file_index % progress_every == 0:
            print(
                f"Scanned {file_index:,}/{total_files:,} files "
                f"({row_count:,} bars, {signal_rows:,} signal-horizon rows) "
                f"latest={path.name}",
                flush=True,
            )

    return report


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
