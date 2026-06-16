from __future__ import annotations

import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


PARQUET_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "date",
    "source_feed",
    "adjustment",
]


def batch_output_path(out_dir: Path, trading_date: date, batch_index: int) -> Path:
    return (
        out_dir
        / f"year={trading_date:%Y}"
        / f"month={trading_date:%m}"
        / f"date={trading_date.isoformat()}"
        / f"batch_{batch_index:03d}.parquet"
    )


def flatten_bars(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    trading_date: date,
    feed: str,
    adjustment: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, bars in bars_by_symbol.items():
        for bar in bars:
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": bar.get("t"),
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "trade_count": bar.get("n"),
                    "vwap": bar.get("vw"),
                    "date": trading_date.isoformat(),
                    "source_feed": feed,
                    "adjustment": adjustment,
                }
            )
    return rows


def write_batch_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    compression: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")

    df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(["symbol", "timestamp"], kind="mergesort")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False, compression=compression)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return len(df)
