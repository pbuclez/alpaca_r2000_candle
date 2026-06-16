from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


BAR_COLUMNS = [
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
]
TRADE_COLUMNS = [
    "symbol",
    "date",
    "signal_timestamp",
    "signal_close",
    "signal_volume",
    "signal_trade_count",
    "signal_vwap",
    "window_minutes",
    "jump_return",
    "entry_timestamp",
    "entry_price",
    "entry_mode",
    "target_price",
    "stop_price",
    "exit_timestamp",
    "exit_price",
    "path_outcome",
    "classified_outcome",
    "realized_return",
    "bars_to_exit",
    "minutes_to_exit",
]


@dataclass(frozen=True)
class TargetStopConfig:
    data_dir: Path
    jump_pct: float = 3.0
    window_minutes: int = 5
    target_pct: float = 2.0
    stop_pct: float = 1.0
    entry_mode: str = "next_open"
    max_hold_minutes: int | None = None
    same_bar_policy: str = "stop_first"
    cooldown_minutes: int = 5
    symbols: tuple[str, ...] = ()
    start: str | None = None
    end: str | None = None

    @property
    def jump_threshold(self) -> float:
        return self.jump_pct / 100.0

    @property
    def target_return(self) -> float:
        return self.target_pct / 100.0

    @property
    def stop_return(self) -> float:
        return self.stop_pct / 100.0

    def validate(self) -> None:
        if self.jump_pct <= 0:
            raise ValueError("jump_pct must be greater than 0")
        if self.window_minutes < 1:
            raise ValueError("window_minutes must be at least 1")
        if self.target_pct <= 0:
            raise ValueError("target_pct must be greater than 0")
        if self.stop_pct <= 0:
            raise ValueError("stop_pct must be greater than 0")
        if self.entry_mode not in {"next_open", "signal_close"}:
            raise ValueError("entry_mode must be 'next_open' or 'signal_close'")
        if self.same_bar_policy not in {"stop_first", "target_first", "ambiguous"}:
            raise ValueError("same_bar_policy must be 'stop_first', 'target_first', or 'ambiguous'")
        if self.max_hold_minutes is not None and self.max_hold_minutes < 1:
            raise ValueError("max_hold_minutes must be at least 1, or use eod")
        if self.cooldown_minutes < 0:
            raise ValueError("cooldown_minutes cannot be negative")


@dataclass(frozen=True)
class DatasetTargetStopResult:
    trades: pd.DataFrame
    summary: pd.DataFrame
    file_count: int
    row_count: int
    symbols: tuple[str, ...]
    dates: tuple[str, ...]


ProgressCallback = Callable[[int, int, Path, int, int], None]


def parse_max_hold(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"eod", "end", "endofday", "end-of-day"}:
        return None
    if normalized.endswith("m"):
        return int(normalized[:-1])
    if normalized.endswith("h"):
        return int(normalized[:-1]) * 60
    return int(normalized)


def run_dataset_target_stop(
    config: TargetStopConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> DatasetTargetStopResult:
    config.validate()
    files = _filtered_parquet_files(config.data_dir, start=config.start, end=config.end)
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {config.data_dir}")

    trade_rows: list[dict[str, object]] = []
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
        trades, _ = run_target_stop(bars, config)
        if not trades.empty:
            trade_rows.extend(trades.to_dict(orient="records"))
        if progress_callback is not None:
            progress_callback(file_index, total_files, file, row_count, len(trade_rows))

    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    summary = summarize_trades(trades) if not trades.empty else _empty_summary()
    return DatasetTargetStopResult(
        trades=trades,
        summary=summary,
        file_count=len(files),
        row_count=row_count,
        symbols=tuple(sorted(symbols)),
        dates=tuple(sorted(dates)),
    )


def run_target_stop(bars: pd.DataFrame, config: TargetStopConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.validate()
    bars = _prepare_bars(bars)
    if bars.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS), _empty_summary()

    trade_rows: list[dict[str, object]] = []
    grouped = bars.groupby(["symbol", "date"], sort=False, group_keys=False)
    for (_, _), group in grouped:
        group_trades = _trades_for_group(group, config)
        if group_trades:
            trade_rows.extend(group_trades)

    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    summary = summarize_trades(trades) if not trades.empty else _empty_summary()
    return trades, summary


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return _empty_summary()

    median_trade_count = float(trades["signal_trade_count"].median())
    low = trades[trades["signal_trade_count"] <= median_trade_count]
    high = trades[trades["signal_trade_count"] > median_trade_count]
    rows = [_summary_row("all", trades, median_trade_count)]
    rows.append(_summary_row("trade_count_low_or_equal_median", low, median_trade_count))
    rows.append(_summary_row("trade_count_above_median", high, median_trade_count))
    return pd.DataFrame(rows)


def _trades_for_group(group: pd.DataFrame, config: TargetStopConfig) -> list[dict[str, object]]:
    group = group.sort_values("timestamp", kind="mergesort").reset_index(drop=True).copy()
    group["jump_return"] = group["close"] / group["close"].shift(config.window_minutes) - 1.0
    candidates = group.index[group["jump_return"] >= config.jump_threshold].tolist()
    if config.cooldown_minutes:
        candidates = _apply_cooldown(candidates, config.cooldown_minutes)

    rows: list[dict[str, object]] = []
    last_index = len(group) - 1
    for signal_index in candidates:
        entry_index = signal_index + 1 if config.entry_mode == "next_open" else signal_index
        if entry_index > last_index:
            rows.append(_no_entry_row(group, signal_index, config))
            continue

        entry_price = _entry_price(group, entry_index, config.entry_mode)
        target_price = entry_price * (1.0 + config.target_return)
        stop_price = entry_price * (1.0 - config.stop_return)
        max_exit_index = _max_exit_index(entry_index, last_index, config.max_hold_minutes)
        exit_info = _resolve_path(
            group,
            entry_index,
            max_exit_index,
            entry_price,
            target_price,
            stop_price,
            config.same_bar_policy,
        )
        rows.append(
            {
                "symbol": group.at[signal_index, "symbol"],
                "date": group.at[signal_index, "date"],
                "signal_timestamp": group.at[signal_index, "timestamp"],
                "signal_close": float(group.at[signal_index, "close"]),
                "signal_volume": group.at[signal_index, "volume"],
                "signal_trade_count": group.at[signal_index, "trade_count"],
                "signal_vwap": group.at[signal_index, "vwap"],
                "window_minutes": config.window_minutes,
                "jump_return": float(group.at[signal_index, "jump_return"]),
                "entry_timestamp": group.at[entry_index, "timestamp"],
                "entry_price": entry_price,
                "entry_mode": config.entry_mode,
                "target_price": target_price,
                "stop_price": stop_price,
                "exit_timestamp": exit_info["exit_timestamp"],
                "exit_price": exit_info["exit_price"],
                "path_outcome": exit_info["path_outcome"],
                "classified_outcome": exit_info["classified_outcome"],
                "realized_return": exit_info["realized_return"],
                "bars_to_exit": exit_info["bars_to_exit"],
                "minutes_to_exit": exit_info["minutes_to_exit"],
            }
        )
    return rows


def _resolve_path(
    group: pd.DataFrame,
    entry_index: int,
    max_exit_index: int,
    entry_price: float,
    target_price: float,
    stop_price: float,
    same_bar_policy: str,
) -> dict[str, object]:
    for index in range(entry_index, max_exit_index + 1):
        hit_target = float(group.at[index, "high"]) >= target_price
        hit_stop = float(group.at[index, "low"]) <= stop_price
        if hit_target and hit_stop:
            if same_bar_policy == "target_first":
                classified_outcome = "target_first"
                exit_price = target_price
            elif same_bar_policy == "ambiguous":
                classified_outcome = "ambiguous"
                exit_price = None
            else:
                classified_outcome = "stop_first"
                exit_price = stop_price
            return _exit_info(
                group,
                index,
                entry_index,
                entry_price,
                exit_price,
                path_outcome="both_same_bar",
                classified_outcome=classified_outcome,
            )
        if hit_target:
            return _exit_info(
                group,
                index,
                entry_index,
                entry_price,
                target_price,
                path_outcome="target_first",
                classified_outcome="target_first",
            )
        if hit_stop:
            return _exit_info(
                group,
                index,
                entry_index,
                entry_price,
                stop_price,
                path_outcome="stop_first",
                classified_outcome="stop_first",
            )

    exit_price = float(group.at[max_exit_index, "close"])
    return _exit_info(
        group,
        max_exit_index,
        entry_index,
        entry_price,
        exit_price,
        path_outcome="no_resolution_eod",
        classified_outcome="unresolved",
    )


def _exit_info(
    group: pd.DataFrame,
    exit_index: int,
    entry_index: int,
    entry_price: float,
    exit_price: float | None,
    *,
    path_outcome: str,
    classified_outcome: str,
) -> dict[str, object]:
    bars_to_exit = exit_index - entry_index
    realized_return = None if exit_price is None else exit_price / entry_price - 1.0
    return {
        "exit_timestamp": group.at[exit_index, "timestamp"],
        "exit_price": exit_price,
        "path_outcome": path_outcome,
        "classified_outcome": classified_outcome,
        "realized_return": realized_return,
        "bars_to_exit": bars_to_exit,
        "minutes_to_exit": bars_to_exit,
    }


def _summary_row(group_name: str, trades: pd.DataFrame, median_trade_count: float | None) -> dict[str, object]:
    total = int(len(trades))
    target = int((trades["classified_outcome"] == "target_first").sum()) if total else 0
    stop = int((trades["classified_outcome"] == "stop_first").sum()) if total else 0
    ambiguous = int((trades["path_outcome"] == "both_same_bar").sum()) if total else 0
    unresolved = int((trades["path_outcome"] == "no_resolution_eod").sum()) if total else 0
    no_entry = int((trades["path_outcome"] == "no_entry").sum()) if total else 0
    resolved = target + stop
    profitable = int((trades["realized_return"] > 0).sum()) if total else 0
    return {
        "group": group_name,
        "median_signal_trade_count": median_trade_count,
        "trades": total,
        "resolved_trades": resolved,
        "target_first": target,
        "stop_first": stop,
        "both_same_bar": ambiguous,
        "unresolved_eod": unresolved,
        "no_entry": no_entry,
        "target_before_stop_rate": _safe_ratio(target, resolved),
        "profitable_rate_all_trades": _safe_ratio(profitable, total),
        "avg_realized_return": _mean_or_none(trades["realized_return"]) if total else None,
        "median_realized_return": _median_or_none(trades["realized_return"]) if total else None,
        "median_minutes_to_exit": _median_or_none(trades["minutes_to_exit"]) if total else None,
        "avg_signal_trade_count": _mean_or_none(trades["signal_trade_count"]) if total else None,
        "avg_signal_volume": _mean_or_none(trades["signal_volume"]) if total else None,
    }


def _no_entry_row(group: pd.DataFrame, signal_index: int, config: TargetStopConfig) -> dict[str, object]:
    return {
        "symbol": group.at[signal_index, "symbol"],
        "date": group.at[signal_index, "date"],
        "signal_timestamp": group.at[signal_index, "timestamp"],
        "signal_close": float(group.at[signal_index, "close"]),
        "signal_volume": group.at[signal_index, "volume"],
        "signal_trade_count": group.at[signal_index, "trade_count"],
        "signal_vwap": group.at[signal_index, "vwap"],
        "window_minutes": config.window_minutes,
        "jump_return": float(group.at[signal_index, "jump_return"]),
        "entry_timestamp": pd.NaT,
        "entry_price": None,
        "entry_mode": config.entry_mode,
        "target_price": None,
        "stop_price": None,
        "exit_timestamp": pd.NaT,
        "exit_price": None,
        "path_outcome": "no_entry",
        "classified_outcome": "no_entry",
        "realized_return": None,
        "bars_to_exit": None,
        "minutes_to_exit": None,
    }


def _entry_price(group: pd.DataFrame, entry_index: int, entry_mode: str) -> float:
    if entry_mode == "next_open":
        return float(group.at[entry_index, "open"])
    return float(group.at[entry_index, "close"])


def _max_exit_index(entry_index: int, last_index: int, max_hold_minutes: int | None) -> int:
    if max_hold_minutes is None:
        return last_index
    return min(last_index, entry_index + max_hold_minutes)


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
    for column in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["symbol", "timestamp", "date", "open", "high", "low", "close"])
    return prepared.sort_values(["symbol", "date", "timestamp"], kind="mergesort").reset_index(drop=True)


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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean_or_none(series: pd.Series) -> float | None:
    series = series.dropna()
    return None if series.empty else float(series.mean())


def _median_or_none(series: pd.Series) -> float | None:
    series = series.dropna()
    return None if series.empty else float(series.median())


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": "all",
                "median_signal_trade_count": None,
                "trades": 0,
                "resolved_trades": 0,
                "target_first": 0,
                "stop_first": 0,
                "both_same_bar": 0,
                "unresolved_eod": 0,
                "no_entry": 0,
                "target_before_stop_rate": None,
                "profitable_rate_all_trades": None,
                "avg_realized_return": None,
                "median_realized_return": None,
                "median_minutes_to_exit": None,
                "avg_signal_trade_count": None,
                "avg_signal_volume": None,
            }
        ]
    )
