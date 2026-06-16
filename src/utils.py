from __future__ import annotations

import csv
import hashlib
import logging
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar
from zoneinfo import ZoneInfo


T = TypeVar("T")
NEW_YORK = ZoneInfo("America/New_York")


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("alpaca_r2000_candle")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def read_symbols_csv(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "symbol" not in reader.fieldnames:
            raise ValueError("Symbols CSV must include a 'symbol' column")

        symbols: list[str] = []
        seen: set[str] = set()
        for row in reader:
            raw = (row.get("symbol") or "").strip().upper()
            if not raw or raw in seen:
                continue
            symbols.append(raw)
            seen.add(raw)

    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    return symbols


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def symbols_hash(symbols: Iterable[str]) -> str:
    joined = "\n".join(symbols)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def regular_session_utc(trading_date: date) -> tuple[datetime, datetime]:
    market_open = datetime.combine(trading_date, time(9, 30), tzinfo=NEW_YORK)
    market_close = datetime.combine(trading_date, time(16, 0), tzinfo=NEW_YORK)
    return market_open.astimezone(UTC), market_close.astimezone(UTC)


def format_rfc3339(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def get_nyse_trading_days(start: date, end: date) -> list[date]:
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise RuntimeError(
            "pandas-market-calendars is required. Install dependencies with "
            "pip install -r requirements.txt."
        ) from exc

    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return [idx.date() for idx in schedule.index]
