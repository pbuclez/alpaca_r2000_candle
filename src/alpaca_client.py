from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from src.config import AlpacaCredentials
from src.utils import format_rfc3339


ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


@dataclass(frozen=True)
class BarsPage:
    bars: dict[str, list[dict[str, Any]]]
    next_page_token: str | None


class AlpacaAPIError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class FeedPermissionError(AlpacaAPIError):
    pass


class AlpacaClient:
    def __init__(
        self,
        credentials: AlpacaCredentials,
        session: requests.Session | None = None,
        bars_url: str = ALPACA_BARS_URL,
    ) -> None:
        self._session = session or requests.Session()
        self._bars_url = bars_url
        self._headers = {
            "APCA-API-KEY-ID": credentials.api_key,
            "APCA-API-SECRET-KEY": credentials.api_secret,
        }

    def get_bars_page(
        self,
        *,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        feed: str,
        adjustment: str,
        page_token: str | None,
        timeout_seconds: float,
    ) -> BarsPage:
        params: dict[str, str | int] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": format_rfc3339(start),
            "end": format_rfc3339(end),
            "limit": 10000,
            "feed": feed,
            "adjustment": adjustment,
        }
        if page_token:
            params["page_token"] = page_token

        response = self._session.get(
            self._bars_url,
            headers=self._headers,
            params=params,
            timeout=timeout_seconds,
        )
        if response.status_code == 403:
            raise FeedPermissionError(
                403,
                _feed_permission_message(feed, response.text),
                response.headers.get("Retry-After"),
            )
        if response.status_code >= 400:
            raise AlpacaAPIError(
                response.status_code,
                _error_message(response),
                response.headers.get("Retry-After"),
            )

        payload = response.json()
        bars_payload = payload.get("bars") or {}
        if isinstance(bars_payload, list):
            bars = _group_list_bars_by_symbol(bars_payload)
        elif isinstance(bars_payload, dict):
            bars = bars_payload
        else:
            bars = {}

        return BarsPage(
            bars=bars,
            next_page_token=payload.get("next_page_token") or None,
        )


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"Alpaca API error {response.status_code}: {response.text[:500]}"
    message = payload.get("message") or payload.get("error") or str(payload)
    return f"Alpaca API error {response.status_code}: {message}"


def _feed_permission_message(feed: str, response_text: str) -> str:
    return (
        f"Alpaca returned HTTP 403 for feed='{feed}'. Your account may not have "
        f"access to that market data feed. Try --feed iex, or check your Alpaca "
        f"data subscription for SIP access. Response: {response_text[:500]}"
    )


def _group_list_bars_by_symbol(bars_payload: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for bar in bars_payload:
        symbol = bar.get("S") or bar.get("symbol")
        if not symbol:
            continue
        grouped.setdefault(str(symbol), []).append(bar)
    return grouped
