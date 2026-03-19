"""
External Signals Module
========================
Pulls free, public market data from Binance to generate alpha signals
that no Roostoo-only team will have. No API key required.

Signals:
    1. Funding Rate — crowd leverage sentiment
       Extreme positive → market overleveraged long → expect crash
       Extreme negative → shorts crowded → expect squeeze up

    2. Open Interest — total leverage in the system
       Rapidly rising OI + high funding = danger (cascade risk)

Usage:
    signals = ExternalSignals()
    signals.fetch("BTC/USD")           # pull latest data
    scalar = signals.get_risk_scalar() # 0.3 to 1.3 multiplier for position sizing
"""

import requests
import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"

# Map Roostoo pairs to Binance futures symbols
PAIR_TO_BINANCE = {
    "BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT", "BNB/USD": "BNBUSDT",
    "XRP/USD": "XRPUSDT", "SOL/USD": "SOLUSDT", "DOGE/USD": "DOGEUSDT",
    "LTC/USD": "LTCUSDT", "LINK/USD": "LINKUSDT", "ATOM/USD": "ATOMUSDT",
    "TRX/USD": "TRXUSDT", "EOS/USD": "EOSUSDT",
}


class ExternalSignals:
    """
    Fetches Binance perpetual futures data to gauge market sentiment.
    Call fetch() every 2-5 minutes (rate-limited internally).
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "RoostooBot/3.0"

        # Thresholds for extreme funding (per 8-hour period)
        self.funding_extreme_high = cfg.get("funding_extreme_high", 0.0005)   # 0.05%
        self.funding_extreme_low = cfg.get("funding_extreme_low", -0.0003)    # -0.03%

        # History buffers
        self.funding_history = deque(maxlen=100)
        self.oi_history = deque(maxlen=50)

        # Rate limiting
        self._last_fetch = 0
        self._min_interval = cfg.get("min_fetch_interval", 120)  # seconds

    def fetch(self, roostoo_pair: str = "BTC/USD"):
        """Fetch latest funding rate and open interest from Binance."""
        now = time.time()
        if now - self._last_fetch < self._min_interval:
            return  # rate limited

        symbol = PAIR_TO_BINANCE.get(roostoo_pair, "BTCUSDT")
        self._fetch_funding(symbol)
        self._fetch_oi(symbol)
        self._last_fetch = now

    def _fetch_funding(self, symbol: str):
        try:
            r = self.session.get(
                f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex",
                params={"symbol": symbol}, timeout=5
            )
            r.raise_for_status()
            rate = float(r.json().get("lastFundingRate", 0))
            self.funding_history.append(rate)
        except Exception as e:
            logger.debug(f"Funding rate fetch failed: {e}")

    def _fetch_oi(self, symbol: str):
        try:
            r = self.session.get(
                f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest",
                params={"symbol": symbol}, timeout=5
            )
            r.raise_for_status()
            oi = float(r.json().get("openInterest", 0))
            self.oi_history.append(oi)
        except Exception as e:
            logger.debug(f"OI fetch failed: {e}")

    def get_funding_signal(self) -> float:
        """
        Funding rate signal: [-1, +1]
        -1 = extreme positive funding (overleveraged longs → bearish)
        +1 = extreme negative funding (crowded shorts → bullish)
         0 = neutral
        """
        if len(self.funding_history) < 1:
            return 0.0

        rate = self.funding_history[-1]
        if rate > self.funding_extreme_high:
            return float(-np.clip((rate - self.funding_extreme_high) / self.funding_extreme_high, 0, 1))
        elif rate < self.funding_extreme_low:
            return float(np.clip((self.funding_extreme_low - rate) / abs(self.funding_extreme_low), 0, 1))
        return 0.0

    def get_oi_signal(self) -> float:
        """
        OI rate-of-change signal: [-1, +1]
        Rapidly rising OI = leverage buildup = danger → negative
        Falling OI = deleveraging = safer → positive
        """
        if len(self.oi_history) < 10:
            return 0.0
        recent = list(self.oi_history)
        current = np.mean(recent[-5:])
        previous = np.mean(recent[-10:-5])
        if previous == 0:
            return 0.0
        change = (current - previous) / previous
        return float(np.clip(-change * 10, -1, 1))

    def get_risk_scalar(self) -> float:
        """
        Combined risk multiplier for position sizing: [0.3, 1.3]
        < 1.0 = external signals say reduce exposure
        > 1.0 = external signals confirm it's safe
        1.0   = neutral / no data available
        """
        if not self.funding_history:
            return 1.0  # no data → neutral

        funding = self.get_funding_signal()
        oi = self.get_oi_signal()
        composite = 0.6 * funding + 0.4 * oi  # funding is more predictive
        # Map [-1, 1] → [0.3, 1.3]
        return float(np.clip(0.8 + composite * 0.5, 0.3, 1.3))

    def get_state(self) -> dict:
        return {
            "funding_rate": round(self.funding_history[-1], 6) if self.funding_history else None,
            "funding_signal": round(self.get_funding_signal(), 3),
            "oi_signal": round(self.get_oi_signal(), 3),
            "risk_scalar": round(self.get_risk_scalar(), 3),
        }

