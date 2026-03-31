import requests
from typing import List, Dict, Optional

class KalshiMarketDataTool:
    """
    TECHNIQUE 2: AGENT TOOL
    Kalshi Market Data Tool
    """

    def __init__(self, base_url: str = "https://api.elections.kalshi.com/trade-api/v2"):
        self.base_url = base_url.rstrip("/")

    def get_markets_by_ticker_list(self, tickers: List[str]) -> List[Dict]:
        if not tickers:
            return []

        url = f"{self.base_url}/markets"
        params = {"tickers": ",".join(tickers)}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("markets", [])

    def get_market_price_snapshot(self, market_ticker: str) -> Optional[Dict]:
        markets = self.get_markets_by_ticker_list([market_ticker])
        if not markets:
            return None

        m = markets[0]
        yes_price_cents = m.get("yes_price")

        if yes_price_cents is None:
            yes_price_cents = m.get("last_price")

        if yes_price_cents is None:
            yes_bid = m.get("yes_bid")
            yes_ask = m.get("yes_ask")
            if yes_bid is not None and yes_ask is not None:
                yes_price_cents = (yes_bid + yes_ask) / 2.0

        volume = m.get("volume")
        yes_price_pct = float(yes_price_cents) if yes_price_cents is not None else None

        return {
            "market_ticker": m.get("ticker"),
            "title": m.get("title"),
            "event_ticker": m.get("event_ticker"),
            "series_ticker": m.get("series_ticker"),
            "yes_price_pct": yes_price_pct,
            "volume": volume,
            "raw": m,
        }
