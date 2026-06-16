from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


BAR_COLUMNS = ["symbol", "timestamp", "close", "date"]
SIGNAL_COLUMNS = [
    "symbol",
    "date",
    "signal_timestamp",
    "signal_close",
    "window_minutes",
    "jump_return",
    "horizon",
    "horizon_minutes",
    "future_timestamp",
    "future_close",
    "future_return",
    "outcome",
    "eligible",
]


@dataclass(frozen=True)
class Horizon:
    label: str
    minutes: int | None

    @property
    def sort_value(self) -> int:
        return self.minutes if self.minutes is not None else 24 * 60


@dataclass(frozen=True)
class BacktestConfig:
    data_dir: Path
    jump_pct: float = 3.0
    window_minutes: int = 5
    horizons: tuple[Horizon, ...] = (
        Horizon("1m", 1),
        Horizon("5m", 5),
        Horizon("15m", 15),
        Horizon("30m", 30),
        Horizon("1h", 60),
        Horizon("5h", 300),
        Horizon("eod", None),
    )
    cooldown_minutes: int = 5
    symbols: tuple[str, ...] = ()
    start: str | None = None
    end: str | None = None

    @property
    def jump_threshold(self) -> float:
        return self.jump_pct / 100.0

    def validate(self) -> None:
        if self.jump_pct <= 0:
            raise ValueError("jump_pct must be greater than 0")
        if self.window_minutes < 1:
            raise ValueError("window_minutes must be at least 1")
        if self.cooldown_minutes < 0:
            raise ValueError("cooldown_minutes cannot be negative")
        if not self.horizons:
            raise ValueError("at least one horizon is required")


@dataclass(frozen=True)
class DatasetBacktestResult:
    signals: pd.DataFrame
    summary: pd.DataFrame
    file_count: int
    row_count: int
    symbols: tuple[str, ...]
    dates: tuple[str, ...]


def parse_horizons(value: str) -> tuple[Horizon, ...]:
    horizons: list[Horizon] = []
    seen: set[str] = set()
    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        horizon = _parse_horizon(part)
        if horizon.label in seen:
            continue
        horizons.append(horizon)
        seen.add(horizon.label)
    if not horizons:
        raise ValueError("No horizons were provided")
    return tuple(sorted(horizons, key=lambda item: item.sort_value))


def load_bars(
    data_dir: Path,
    *,
    symbols: Iterable[str] = (),
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {data_dir}")

    frames = [pd.read_parquet(path, columns=BAR_COLUMNS) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return _prepare_bars(df)

    selected_symbols = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
    if selected_symbols:
        df = df[df["symbol"].astype(str).str.upper().isin(selected_symbols)]
    if start is not None:
        df = df[df["date"].astype(str) >= start]
    if end is not None:
        df = df[df["date"].astype(str) <= end]
    return _prepare_bars(df)


ProgressCallback = Callable[[int, int, Path, int, int], None]


def run_dataset_backtest(
    config: BacktestConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> DatasetBacktestResult:
    config.validate()
    files = _filtered_parquet_files(config.data_dir, start=config.start, end=config.end)
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {config.data_dir}")

    signal_rows: list[dict[str, object]] = []
    row_count = 0
    symbols: set[str] = set()
    dates: set[str] = set()
    total_files = len(files)
    for file_index, file in enumerate(files, start=1):
        bars = pd.read_parquet(file, columns=BAR_COLUMNS)
        if config.symbols:
            selected_symbols = set(config.symbols)
            bars = bars[bars["symbol"].astype(str).str.upper().isin(selected_symbols)]
        if config.start is not None:
            bars = bars[bars["date"].astype(str) >= config.start]
        if config.end is not None:
            bars = bars[bars["date"].astype(str) <= config.end]
        bars = _prepare_bars(bars)
        if bars.empty:
            continue

        row_count += len(bars)
        symbols.update(bars["symbol"].unique())
        dates.update(bars["date"].unique())
        signals, _ = run_backtest(bars, config)
        if not signals.empty:
            signal_rows.extend(signals.to_dict(orient="records"))
        if progress_callback is not None:
            progress_callback(file_index, total_files, file, row_count, len(signal_rows))

    signals = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    summary = summarize_signals(signals, config.horizons) if not signals.empty else _empty_summary(config.horizons)
    return DatasetBacktestResult(
        signals=signals,
        summary=summary,
        file_count=len(files),
        row_count=row_count,
        symbols=tuple(sorted(symbols)),
        dates=tuple(sorted(dates)),
    )


def run_backtest(bars: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.validate()
    bars = _prepare_bars(bars)
    if bars.empty:
        return pd.DataFrame(), pd.DataFrame()

    signal_rows: list[dict[str, object]] = []
    grouped = bars.groupby(["symbol", "date"], sort=False, group_keys=False)
    for (_, _), group in grouped:
        group_signals = _signals_for_group(group, config)
        if not group_signals.empty:
            signal_rows.extend(group_signals.to_dict(orient="records"))

    if not signal_rows:
        return pd.DataFrame(), _empty_summary(config.horizons)

    signals = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    summary = summarize_signals(signals, config.horizons)
    return signals, summary


def summarize_signals(signals: pd.DataFrame, horizons: Iterable[Horizon]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horizon_order = {horizon.label: index for index, horizon in enumerate(horizons)}
    for horizon_label, group in signals.groupby("horizon", sort=False):
        eligible = group[group["eligible"]].copy()
        follow = int((eligible["outcome"] == "follow_through").sum())
        revert = int((eligible["outcome"] == "revert").sum())
        flat = int((eligible["outcome"] == "flat").sum())
        eligible_count = int(len(eligible))
        days = int(eligible["date"].nunique()) if eligible_count else 0
        rows.append(
            {
                "horizon": horizon_label,
                "eligible_signals": eligible_count,
                "follow_through": follow,
                "revert": revert,
                "flat": flat,
                "follow_revert_ratio": _follow_revert_ratio(follow, revert),
                "win_rate": _safe_ratio(follow, eligible_count),
                "signal_frequency_per_day": _safe_ratio(eligible_count, days),
                "avg_future_return": _mean_or_none(eligible["future_return"]),
                "median_future_return": _median_or_none(eligible["future_return"]),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return _empty_summary(horizons)
    summary["_order"] = summary["horizon"].map(horizon_order)
    return summary.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _signals_for_group(group: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    group = group.sort_values("timestamp", kind="mergesort").reset_index(drop=True).copy()
    group["jump_return"] = group["close"] / group["close"].shift(config.window_minutes) - 1.0
    candidates = group.index[group["jump_return"] >= config.jump_threshold].tolist()
    if config.cooldown_minutes:
        candidates = _apply_cooldown(candidates, config.cooldown_minutes)
    if not candidates:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    last_index = len(group) - 1
    eod_close = float(group.at[last_index, "close"])
    eod_timestamp = group.at[last_index, "timestamp"]

    for index in candidates:
        signal_close = float(group.at[index, "close"])
        for horizon in config.horizons:
            future_index = None if horizon.minutes is None else index + horizon.minutes
            if horizon.minutes is None:
                eligible = index < last_index
                future_close = eod_close if eligible else None
                future_timestamp = eod_timestamp if eligible else pd.NaT
            else:
                eligible = future_index <= last_index
                future_close = float(group.at[future_index, "close"]) if eligible else None
                future_timestamp = group.at[future_index, "timestamp"] if eligible else pd.NaT

            future_return = None
            outcome = "unavailable"
            if eligible and future_close is not None:
                future_return = future_close / signal_close - 1.0
                outcome = _classify_return(future_return)

            rows.append(
                {
                    "symbol": group.at[index, "symbol"],
                    "date": group.at[index, "date"],
                    "signal_timestamp": group.at[index, "timestamp"],
                    "signal_close": signal_close,
                    "window_minutes": config.window_minutes,
                    "jump_return": float(group.at[index, "jump_return"]),
                    "horizon": horizon.label,
                    "horizon_minutes": horizon.minutes,
                    "future_timestamp": future_timestamp,
                    "future_close": future_close,
                    "future_return": future_return,
                    "outcome": outcome,
                    "eligible": bool(eligible),
                }
            )
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


def _apply_cooldown(indices: list[int], cooldown_minutes: int) -> list[int]:
    selected: list[int] = []
    next_allowed = -1
    for index in indices:
        if index < next_allowed:
            continue
        selected.append(index)
        next_allowed = index + cooldown_minutes
    return selected


def _prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    missing = [column for column in BAR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    prepared = df.loc[:, BAR_COLUMNS].copy()
    prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
    prepared["date"] = prepared["date"].astype(str)
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared = prepared.dropna(subset=["symbol", "timestamp", "date", "close"])
    return prepared.sort_values(["symbol", "date", "timestamp"], kind="mergesort").reset_index(drop=True)


def _parse_horizon(value: str) -> Horizon:
    if value in {"eod", "end", "endofday", "end-of-day"}:
        return Horizon("eod", None)
    if value.endswith("m"):
        minutes = int(value[:-1])
        return Horizon(f"{minutes}m", minutes)
    if value.endswith("h"):
        hours = int(value[:-1])
        return Horizon(f"{hours}h", hours * 60)
    minutes = int(value)
    return Horizon(f"{minutes}m", minutes)


def _filtered_parquet_files(data_dir: Path, *, start: str | None, end: str | None) -> list[Path]:
    files: list[Path] = []
    for file in sorted(data_dir.rglob("*.parquet")):
        file_date = _date_from_path(file)
        if file_date is not None:
            if start is not None and file_date < start:
                continue
            if end is not None and file_date > end:
                continue
        files.append(file)
    return files


def _date_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("date="):
            return part.removeprefix("date=")
    return None


def _classify_return(value: float) -> str:
    if value > 0:
        return "follow_through"
    if value < 0:
        return "revert"
    return "flat"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _follow_revert_ratio(follow: int, revert: int) -> float | None:
    if revert == 0:
        return float("inf") if follow > 0 else None
    return float(follow) / float(revert)


def _mean_or_none(series: pd.Series) -> float | None:
    series = series.dropna()
    return None if series.empty else float(series.mean())


def _median_or_none(series: pd.Series) -> float | None:
    series = series.dropna()
    return None if series.empty else float(series.median())


def _empty_summary(horizons: Iterable[Horizon]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": horizon.label,
                "eligible_signals": 0,
                "follow_through": 0,
                "revert": 0,
                "flat": 0,
                "follow_revert_ratio": None,
                "win_rate": None,
                "signal_frequency_per_day": None,
                "avg_future_return": None,
                "median_future_return": None,
            }
            for horizon in horizons
        ]
    )
