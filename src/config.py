from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils import parse_date


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    api_secret: str


@dataclass(frozen=True)
class DownloadConfig:
    symbols_path: Path
    start: date
    end: date
    timeframe: str = "1Min"
    feed: str = "sip"
    adjustment: str = "raw"
    out_dir: Path = Path("data/raw/r2000_1min")
    batch_size: int = 300
    requests_per_minute: int = 180
    resume: bool = False
    manifest_path: Path = Path("logs/download_manifest.sqlite")
    log_file: Path = Path("logs/downloader.log")
    compression: str = "zstd"
    max_retries: int = 5
    timeout_seconds: float = 30.0

    def validate(self) -> None:
        if self.start > self.end:
            raise ValueError("--start must be earlier than or equal to --end")
        if self.timeframe != "1Min":
            raise ValueError("This downloader is built for timeframe=1Min")
        if self.feed not in {"sip", "iex"}:
            raise ValueError("--feed must be either 'sip' or 'iex'")
        if self.adjustment not in {"raw", "all"}:
            raise ValueError("--adjustment must be either 'raw' or 'all'")
        if self.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        if self.requests_per_minute < 1:
            raise ValueError("--requests-per-minute must be at least 1")
        if self.max_retries < 0:
            raise ValueError("--max-retries cannot be negative")


def load_credentials(env_path: Path | None = None) -> AlpacaCredentials:
    if env_path is not None:
        load_dotenv(env_path)
    load_dotenv()

    import os

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    api_secret = os.getenv("ALPACA_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise ValueError(
            "Missing Alpaca credentials. Create a .env file with "
            "ALPACA_API_KEY and ALPACA_API_SECRET."
        )
    return AlpacaCredentials(api_key=api_key, api_secret=api_secret)


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def build_download_config(args: Any, project_root: Path) -> DownloadConfig:
    file_config = load_yaml_config(args.config)

    def value(cli_name: str, config_name: str, default: Any = None) -> Any:
        cli_value = getattr(args, cli_name)
        if cli_value is not None:
            return cli_value
        return file_config.get(config_name, default)

    start_value = value("start", "start")
    end_value = value("end", "end")
    symbols_value = value("symbols", "symbols")
    if not start_value:
        raise ValueError("Missing required --start date, for example 2025-01-01")
    if not end_value:
        raise ValueError("Missing required --end date, for example 2025-12-31")
    if not symbols_value:
        raise ValueError("Missing required --symbols path")

    cfg = DownloadConfig(
        symbols_path=_resolve(project_root, Path(symbols_value)),
        start=parse_date(start_value),
        end=parse_date(end_value),
        timeframe=value("timeframe", "timeframe", "1Min"),
        feed=value("feed", "feed", "sip"),
        adjustment=value("adjustment", "adjustment", "raw"),
        out_dir=_resolve(project_root, Path(value("out", "out", "data/raw/r2000_1min"))),
        batch_size=int(value("batch_size", "batch_size", 300)),
        requests_per_minute=int(value("requests_per_minute", "requests_per_minute", 180)),
        resume=_parse_bool(value("resume", "resume", False)),
        manifest_path=_resolve(
            project_root,
            Path(value("manifest", "manifest", "logs/download_manifest.sqlite")),
        ),
        log_file=_resolve(project_root, Path(value("log_file", "log_file", "logs/downloader.log"))),
        compression=value("compression", "compression", "zstd"),
        max_retries=int(value("max_retries", "max_retries", 5)),
        timeout_seconds=float(value("timeout_seconds", "timeout_seconds", 30.0)),
    )
    cfg.validate()
    return cfg


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)
