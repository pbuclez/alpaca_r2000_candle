#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alpaca_client import AlpacaClient, FeedPermissionError
from src.config import build_download_config, load_credentials
from src.downloader import Downloader
from src.manifest import Manifest
from src.rate_limiter import RateLimiter
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 1-minute historical stock bars from Alpaca by NYSE trading day."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--symbols", default=None, help="CSV file with a required 'symbol' column.")
    parser.add_argument("--start", default=None, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD.")
    parser.add_argument("--timeframe", default=None, help="Bar timeframe. This project supports 1Min.")
    parser.add_argument("--feed", default=None, choices=["sip", "iex"], help="Alpaca data feed.")
    parser.add_argument("--adjustment", default=None, choices=["raw", "all"], help="Adjustment mode.")
    parser.add_argument("--out", default=None, help="Output directory for partitioned Parquet files.")
    parser.add_argument("--batch-size", type=int, default=None, help="Symbols per request batch.")
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=None,
        help="Client-side request limit. Default is 180.",
    )
    parser.add_argument("--resume", action="store_true", default=None, help="Skip completed manifest rows.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resume.")
    parser.add_argument("--manifest", default=None, help="SQLite manifest path.")
    parser.add_argument("--log-file", default=None, help="Log file path.")
    parser.add_argument("--compression", default=None, help="Parquet compression, e.g. zstd or snappy.")
    parser.add_argument("--max-retries", type=int, default=None, help="Retries per failed page request.")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="HTTP timeout per request.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = build_download_config(args, PROJECT_ROOT)
        logger = setup_logging(config.log_file)
        credentials = load_credentials(PROJECT_ROOT / ".env")
        manifest = Manifest(config.manifest_path)
        try:
            downloader = Downloader(
                config=config,
                client=AlpacaClient(credentials),
                manifest=manifest,
                rate_limiter=RateLimiter(config.requests_per_minute),
                logger=logger,
            )
            downloader.run()
        finally:
            manifest.close()
        return 0
    except FeedPermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("Interrupted. Rerun with --resume to skip completed batches.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
