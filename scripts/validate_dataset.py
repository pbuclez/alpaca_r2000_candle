#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow.parquet as pq

from src.manifest import Manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a downloaded Alpaca Parquet dataset.")
    parser.add_argument("--out", default="data/raw/r2000_1min", help="Parquet dataset directory.")
    parser.add_argument(
        "--manifest",
        default="logs/download_manifest.sqlite",
        help="SQLite manifest path.",
    )
    parser.add_argument(
        "--check-sorted",
        action="store_true",
        help="Sample symbols and verify timestamps are sorted.",
    )
    parser.add_argument("--sample-symbols", type=int, default=5, help="Symbols to sample for sorting checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = _resolve(Path(args.out))
    manifest_path = _resolve(Path(args.manifest))

    files = sorted(out_dir.rglob("*.parquet"))
    if not files:
        print(f"No Parquet files found under {out_dir}")
    else:
        stats = scan_parquet_files(files)
        print(f"Parquet files: {stats['file_count']}")
        print(f"Total rows: {stats['row_count']:,}")
        print(f"Unique symbols: {len(stats['symbols']):,}")
        print(f"Min timestamp: {stats['min_timestamp']}")
        print(f"Max timestamp: {stats['max_timestamp']}")

        if args.check_sorted:
            sample = random.sample(
                sorted(stats["symbols"]),
                min(args.sample_symbols, len(stats["symbols"])),
            )
            check_sorted(files, sample)

    if manifest_path.exists():
        report_manifest(manifest_path)
    else:
        print(f"Manifest not found: {manifest_path}")
    return 0


def scan_parquet_files(files: list[Path]) -> dict:
    symbols: set[str] = set()
    row_count = 0
    min_timestamp = None
    max_timestamp = None

    for file in files:
        metadata = pq.ParquetFile(file).metadata
        row_count += metadata.num_rows
        table = pq.read_table(file, columns=["symbol", "timestamp"])
        df = table.to_pandas()
        if df.empty:
            continue
        symbols.update(df["symbol"].dropna().astype(str).unique())
        current_min = df["timestamp"].min()
        current_max = df["timestamp"].max()
        min_timestamp = current_min if min_timestamp is None else min(min_timestamp, current_min)
        max_timestamp = current_max if max_timestamp is None else max(max_timestamp, current_max)

    return {
        "file_count": len(files),
        "row_count": row_count,
        "symbols": symbols,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
    }


def check_sorted(files: list[Path], symbols: list[str]) -> None:
    if not symbols:
        print("No symbols available for sorted timestamp check.")
        return

    print(f"Checking timestamp sort order for: {', '.join(symbols)}")
    for symbol in symbols:
        timestamps = []
        for file in files:
            df = pd.read_parquet(file, columns=["symbol", "timestamp"])
            selected = df.loc[df["symbol"] == symbol, "timestamp"]
            if not selected.empty:
                timestamps.extend(selected.tolist())
        is_sorted = all(left <= right for left, right in zip(timestamps, timestamps[1:]))
        status = "OK" if is_sorted else "NOT SORTED"
        print(f"  {symbol}: {status} ({len(timestamps):,} bars)")


def report_manifest(manifest_path: Path) -> None:
    manifest = Manifest(manifest_path)
    try:
        print(f"Manifest status counts: {manifest.status_counts()}")
        problems = manifest.problem_rows()
        if problems:
            print("Non-completed manifest rows:")
            for row in problems[:25]:
                print(
                    f"  {row['date']} batch={row['batch_index']:03d} "
                    f"feed={row['feed']} adjustment={row['adjustment']} "
                    f"status={row['status']} error={row['error_message']}"
                )
            if len(problems) > 25:
                print(f"  ... {len(problems) - 25} more")

        missing_files = [
            row
            for row in manifest.completed_rows()
            if not row.get("output_path") or not Path(row["output_path"]).exists()
        ]
        if missing_files:
            print("Completed manifest rows with missing output files:")
            for row in missing_files[:25]:
                print(
                    f"  {row['date']} batch={row['batch_index']:03d} "
                    f"feed={row['feed']} adjustment={row['adjustment']} "
                    f"path={row['output_path']}"
                )
            if len(missing_files) > 25:
                print(f"  ... {len(missing_files) - 25} more")
        elif not problems:
            print("Manifest has no failed/pending rows and no missing completed files.")
    finally:
        manifest.close()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
