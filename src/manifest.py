from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import utc_now_iso


@dataclass(frozen=True)
class ManifestRecord:
    date: str
    batch_index: int
    symbols_hash: str
    status: str
    output_path: str | None


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def get_record(
        self,
        *,
        trading_date: str,
        batch_index: int,
        feed: str,
        adjustment: str,
    ) -> ManifestRecord | None:
        row = self._conn.execute(
            """
            SELECT date, batch_index, symbols_hash, status, output_path
            FROM download_manifest
            WHERE date = ? AND batch_index = ? AND feed = ? AND adjustment = ?
            """,
            (trading_date, batch_index, feed, adjustment),
        ).fetchone()
        if row is None:
            return None
        return ManifestRecord(
            date=row["date"],
            batch_index=row["batch_index"],
            symbols_hash=row["symbols_hash"],
            status=row["status"],
            output_path=row["output_path"],
        )

    def mark_pending(
        self,
        *,
        trading_date: str,
        batch_index: int,
        symbols_hash: str,
        symbol_count: int,
        start_time: str,
        end_time: str,
        feed: str,
        adjustment: str,
        output_path: str,
    ) -> None:
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO download_manifest (
                date, batch_index, symbols_hash, symbol_count, start_time, end_time,
                feed, adjustment, output_path, row_count, page_count, request_count,
                status, error_message, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 'pending', NULL, ?, ?)
            ON CONFLICT(date, batch_index, feed, adjustment) DO UPDATE SET
                symbols_hash = excluded.symbols_hash,
                symbol_count = excluded.symbol_count,
                start_time = excluded.start_time,
                end_time = excluded.end_time,
                output_path = excluded.output_path,
                row_count = 0,
                page_count = 0,
                request_count = 0,
                status = 'pending',
                error_message = NULL,
                updated_at = excluded.updated_at
            """,
            (
                trading_date,
                batch_index,
                symbols_hash,
                symbol_count,
                start_time,
                end_time,
                feed,
                adjustment,
                output_path,
                now,
                now,
            ),
        )
        self._conn.commit()

    def mark_completed(
        self,
        *,
        trading_date: str,
        batch_index: int,
        feed: str,
        adjustment: str,
        row_count: int,
        page_count: int,
        request_count: int,
    ) -> None:
        self._conn.execute(
            """
            UPDATE download_manifest
            SET row_count = ?,
                page_count = ?,
                request_count = ?,
                status = 'completed',
                error_message = NULL,
                updated_at = ?
            WHERE date = ? AND batch_index = ? AND feed = ? AND adjustment = ?
            """,
            (
                row_count,
                page_count,
                request_count,
                utc_now_iso(),
                trading_date,
                batch_index,
                feed,
                adjustment,
            ),
        )
        self._conn.commit()

    def mark_failed(
        self,
        *,
        trading_date: str,
        batch_index: int,
        feed: str,
        adjustment: str,
        error_message: str,
        page_count: int,
        request_count: int,
    ) -> None:
        self._conn.execute(
            """
            UPDATE download_manifest
            SET page_count = ?,
                request_count = ?,
                status = 'failed',
                error_message = ?,
                updated_at = ?
            WHERE date = ? AND batch_index = ? AND feed = ? AND adjustment = ?
            """,
            (
                page_count,
                request_count,
                error_message[:2000],
                utc_now_iso(),
                trading_date,
                batch_index,
                feed,
                adjustment,
            ),
        )
        self._conn.commit()

    def status_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS count FROM download_manifest GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def problem_rows(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT date, batch_index, feed, adjustment, status, output_path, error_message
            FROM download_manifest
            WHERE status != 'completed'
            ORDER BY date, batch_index, feed, adjustment
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def completed_rows(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT date, batch_index, feed, adjustment, output_path
            FROM download_manifest
            WHERE status = 'completed'
            ORDER BY date, batch_index, feed, adjustment
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS download_manifest (
                date TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                symbols_hash TEXT NOT NULL,
                symbol_count INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                feed TEXT NOT NULL,
                adjustment TEXT NOT NULL,
                output_path TEXT,
                row_count INTEGER NOT NULL DEFAULT 0,
                page_count INTEGER NOT NULL DEFAULT 0,
                request_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL CHECK(status IN ('pending', 'completed', 'failed')),
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(date, batch_index, feed, adjustment)
            )
            """
        )
        self._conn.commit()
