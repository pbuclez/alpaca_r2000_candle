from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from src.alpaca_client import AlpacaAPIError, AlpacaClient, BarsPage, FeedPermissionError
from src.config import DownloadConfig
from src.manifest import Manifest
from src.rate_limiter import RateLimiter
from src.storage import batch_output_path, flatten_bars, write_batch_parquet
from src.utils import chunked, get_nyse_trading_days, read_symbols_csv, regular_session_utc, symbols_hash


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class FetchCounters:
    page_count: int = 0
    request_count: int = 0


@dataclass
class FetchResult:
    rows: list[dict]
    page_count: int
    request_count: int


class Downloader:
    def __init__(
        self,
        *,
        config: DownloadConfig,
        client: AlpacaClient,
        manifest: Manifest,
        rate_limiter: RateLimiter,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.client = client
        self.manifest = manifest
        self.rate_limiter = rate_limiter
        self.logger = logger

    def run(self) -> None:
        symbols = read_symbols_csv(self.config.symbols_path)
        batches = list(chunked(symbols, self.config.batch_size))
        trading_days = get_nyse_trading_days(self.config.start, self.config.end)

        self.logger.info(
            "Starting download: %s symbols, %s batches/day, %s trading days, feed=%s, adjustment=%s",
            len(symbols),
            len(batches),
            len(trading_days),
            self.config.feed,
            self.config.adjustment,
        )

        for trading_date in trading_days:
            self.logger.info("Starting trading day %s", trading_date.isoformat())
            start_utc, end_utc = regular_session_utc(trading_date)
            for batch_index, batch_symbols in enumerate(batches):
                self._run_batch(
                    trading_date=trading_date.isoformat(),
                    batch_index=batch_index,
                    batch_symbols=batch_symbols,
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
            self.logger.info("Finished trading day %s", trading_date.isoformat())

        self.logger.info("Download finished. Manifest status counts: %s", self.manifest.status_counts())

    def _run_batch(
        self,
        *,
        trading_date: str,
        batch_index: int,
        batch_symbols: list[str],
        start_utc: datetime,
        end_utc: datetime,
    ) -> None:
        trading_day = date.fromisoformat(trading_date)
        batch_hash = symbols_hash(batch_symbols)
        output_path = batch_output_path(self.config.out_dir, trading_day, batch_index)

        if self._should_skip_completed(
            trading_date=trading_date,
            batch_index=batch_index,
            batch_hash=batch_hash,
            output_path=output_path,
        ):
            self.logger.info("Skipping completed batch %s/%03d", trading_date, batch_index)
            return

        self.manifest.mark_pending(
            trading_date=trading_date,
            batch_index=batch_index,
            symbols_hash=batch_hash,
            symbol_count=len(batch_symbols),
            start_time=start_utc.isoformat(),
            end_time=end_utc.isoformat(),
            feed=self.config.feed,
            adjustment=self.config.adjustment,
            output_path=str(output_path),
        )

        self.logger.info(
            "Starting batch %s/%03d: %s symbols",
            trading_date,
            batch_index,
            len(batch_symbols),
        )
        result: FetchResult | None = None
        try:
            result = self._fetch_batch(
                batch_symbols=batch_symbols,
                trading_date=trading_day,
                start_utc=start_utc,
                end_utc=end_utc,
            )
            row_count = write_batch_parquet(
                result.rows,
                output_path,
                compression=self.config.compression,
            )
            self.manifest.mark_completed(
                trading_date=trading_date,
                batch_index=batch_index,
                feed=self.config.feed,
                adjustment=self.config.adjustment,
                row_count=row_count,
                page_count=result.page_count,
                request_count=result.request_count,
            )
            self.logger.info(
                "Finished batch %s/%03d: pages=%s requests=%s rows=%s output=%s",
                trading_date,
                batch_index,
                result.page_count,
                result.request_count,
                row_count,
                output_path,
            )
        except FeedPermissionError as exc:
            counters = getattr(exc, "counters", FetchCounters())
            self.manifest.mark_failed(
                trading_date=trading_date,
                batch_index=batch_index,
                feed=self.config.feed,
                adjustment=self.config.adjustment,
                error_message=str(exc),
                page_count=counters.page_count,
                request_count=counters.request_count,
            )
            self.logger.error(str(exc))
            raise
        except Exception as exc:
            counters = (
                FetchCounters(result.page_count, result.request_count)
                if result is not None
                else getattr(exc, "counters", FetchCounters())
            )
            self.manifest.mark_failed(
                trading_date=trading_date,
                batch_index=batch_index,
                feed=self.config.feed,
                adjustment=self.config.adjustment,
                error_message=str(exc),
                page_count=counters.page_count,
                request_count=counters.request_count,
            )
            self.logger.exception("Failed batch %s/%03d: %s", trading_date, batch_index, exc)

    def _should_skip_completed(
        self,
        *,
        trading_date: str,
        batch_index: int,
        batch_hash: str,
        output_path: Path,
    ) -> bool:
        if not self.config.resume:
            return False

        record = self.manifest.get_record(
            trading_date=trading_date,
            batch_index=batch_index,
            feed=self.config.feed,
            adjustment=self.config.adjustment,
        )
        if record is None or record.status != "completed":
            return False
        if record.symbols_hash != batch_hash:
            self.logger.warning(
                "Re-downloading %s/%03d because symbols_hash changed",
                trading_date,
                batch_index,
            )
            return False
        manifest_path = Path(record.output_path) if record.output_path else output_path
        if not manifest_path.exists():
            self.logger.warning(
                "Re-downloading %s/%03d because manifest output is missing: %s",
                trading_date,
                batch_index,
                manifest_path,
            )
            return False
        return True

    def _fetch_batch(
        self,
        *,
        batch_symbols: list[str],
        trading_date: date,
        start_utc: datetime,
        end_utc: datetime,
    ) -> FetchResult:
        counters = FetchCounters()
        rows: list[dict] = []
        page_token: str | None = None

        while True:
            page = self._request_page_with_retries(
                batch_symbols=batch_symbols,
                start_utc=start_utc,
                end_utc=end_utc,
                page_token=page_token,
                counters=counters,
            )
            counters.page_count += 1
            rows.extend(
                flatten_bars(
                    page.bars,
                    trading_date=trading_date,
                    feed=self.config.feed,
                    adjustment=self.config.adjustment,
                )
            )
            self.logger.info(
                "Fetched page %s for %s: next_page_token=%s",
                counters.page_count,
                trading_date,
                bool(page.next_page_token),
            )
            if not page.next_page_token:
                break
            page_token = page.next_page_token

        return FetchResult(
            rows=rows,
            page_count=counters.page_count,
            request_count=counters.request_count,
        )

    def _request_page_with_retries(
        self,
        *,
        batch_symbols: list[str],
        start_utc: datetime,
        end_utc: datetime,
        page_token: str | None,
        counters: FetchCounters,
    ) -> BarsPage:
        retry_count = 0
        while True:
            try:
                self.rate_limiter.wait()
                counters.request_count += 1
                return self.client.get_bars_page(
                    symbols=batch_symbols,
                    start=start_utc,
                    end=end_utc,
                    timeframe=self.config.timeframe,
                    feed=self.config.feed,
                    adjustment=self.config.adjustment,
                    page_token=page_token,
                    timeout_seconds=self.config.timeout_seconds,
                )
            except FeedPermissionError as exc:
                exc.counters = counters
                raise
            except AlpacaAPIError as exc:
                if exc.status_code not in RETRYABLE_STATUS_CODES or retry_count >= self.config.max_retries:
                    exc.counters = counters
                    raise
                delay = _retry_delay(retry_count, exc.retry_after)
                retry_count += 1
                self.logger.warning(
                    "Retryable Alpaca API error status=%s retry=%s/%s sleeping %.1fs: %s",
                    exc.status_code,
                    retry_count,
                    self.config.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if retry_count >= self.config.max_retries:
                    exc.counters = counters
                    raise
                delay = _retry_delay(retry_count, None)
                retry_count += 1
                self.logger.warning(
                    "Retryable network error retry=%s/%s sleeping %.1fs: %s",
                    retry_count,
                    self.config.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)


def _retry_delay(retry_count: int, retry_after: str | None) -> float:
    retry_after_seconds = _parse_retry_after(retry_after)
    if retry_after_seconds is not None:
        return retry_after_seconds
    base = min(60.0, 2.0**retry_count)
    return base + random.uniform(0.0, min(1.0, base * 0.25))


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at.astimezone(UTC) - datetime.now(UTC)).total_seconds())
