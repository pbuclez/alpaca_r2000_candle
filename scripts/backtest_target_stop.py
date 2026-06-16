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

from src.target_stop_backtest import (
    TargetStopConfig,
    parse_max_hold,
    run_dataset_target_stop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest whether a jump signal hits a profit target before a stop loss."
    )
    parser.add_argument("--data", default="data/raw/r2000_1min", help="Parquet dataset directory.")
    parser.add_argument("--jump-pct", type=float, default=3.0, help="Jump threshold percent.")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=5,
        help="Lookback window in 1-minute bars.",
    )
    parser.add_argument("--target-pct", type=float, default=2.0, help="Profit target percent.")
    parser.add_argument("--stop-pct", type=float, default=1.0, help="Stop loss percent.")
    parser.add_argument(
        "--entry",
        choices=["next_open", "signal_close"],
        default="next_open",
        help="Entry price assumption.",
    )
    parser.add_argument(
        "--max-hold",
        default="eod",
        help="Maximum hold, for example eod, 30m, 1h, or 120.",
    )
    parser.add_argument(
        "--same-bar-policy",
        choices=["stop_first", "target_first", "ambiguous"],
        default="stop_first",
        help="How to classify bars where high hits target and low hits stop.",
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
    parser.add_argument("--trades-out", default=None, help="Optional CSV path for trade-level rows.")
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
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    cooldown = 0 if args.allow_overlap else args.cooldown_minutes
    max_hold_minutes = parse_max_hold(args.max_hold)
    config = TargetStopConfig(
        data_dir=data_dir,
        jump_pct=args.jump_pct,
        window_minutes=args.window_minutes,
        target_pct=args.target_pct,
        stop_pct=args.stop_pct,
        entry_mode=args.entry,
        max_hold_minutes=max_hold_minutes,
        same_bar_policy=args.same_bar_policy,
        cooldown_minutes=cooldown,
        symbols=symbols,
        start=args.start,
        end=args.end,
    )

    max_hold_label = "eod" if max_hold_minutes is None else f"{max_hold_minutes}m"
    print(f"Starting target/stop backtest in {data_dir}", flush=True)
    print(f"Signal rule: close / close.shift({args.window_minutes}) - 1 >= {args.jump_pct:.2f}%", flush=True)
    print(
        f"Entry={args.entry} target=+{args.target_pct:.2f}% "
        f"stop=-{args.stop_pct:.2f}% max_hold={max_hold_label}",
        flush=True,
    )
    print(f"Same-bar policy: {args.same_bar_policy}", flush=True)
    print(f"Cooldown minutes: {cooldown}", flush=True)

    result = run_dataset_target_stop(
        config,
        progress_callback=_progress_callback(args.progress_every),
    )
    trades = result.trades
    summary = result.summary

    print(f"Data directory: {data_dir}")
    print(f"Parquet files scanned: {result.file_count:,}")
    print(f"Bars loaded: {result.row_count:,}")
    print(f"Unique symbols: {len(result.symbols):,}")
    print(f"Trading days: {len(result.dates):,}")
    print()
    print(_format_summary(summary))

    if args.trades_out:
        path = _resolve(Path(args.trades_out))
        path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(path, index=False)
        print(f"\nTrade rows written to {path}")

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
        return "No trades found."

    formatted = summary.copy()
    pct_columns = [
        "target_before_stop_rate",
        "profitable_rate_all_trades",
        "avg_realized_return",
        "median_realized_return",
    ]
    for column in pct_columns:
        formatted[column] = formatted[column].map(_format_pct)
    for column in [
        "median_signal_trade_count",
        "median_minutes_to_exit",
        "avg_signal_trade_count",
        "avg_signal_volume",
    ]:
        formatted[column] = formatted[column].map(_format_float)
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

    def report(file_index: int, total_files: int, path: Path, row_count: int, trades: int) -> None:
        if file_index == 1 or file_index == total_files or file_index % progress_every == 0:
            print(
                f"Scanned {file_index:,}/{total_files:,} files "
                f"({row_count:,} bars, {trades:,} trades) latest={path.name}",
                flush=True,
            )

    return report


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
