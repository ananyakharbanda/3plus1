"""
External Signals — Binance funding rate + open interest
========================================================
Free public data, no API key needed.
Gives us information no Roostoo-only team has.
"""

import requests, numpy as np, time, logging
from collections import deque
from typing import Dict

logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"
PAIR_TO_BINANCE = {
    "BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT", "BNB/USD": "BNBUSDT",
    "XRP/USD": "XRPUSDT", "SOL/USD": "SOLUSDT", "DOGE/USD": "DOGEUSDT",
    "LTC/USD": "LTCUSDT", "LINK/USD": "LINKUSDT", "ATOM/USD": "ATOMUSDT",
    "ONDO/USD": "ONDOUSDT", "AVAX/USD": "AVAXUSDT", "NEAR/USD": "NEARUSDT",
    "DOT/USD": "DOTUSDT",
}

class ExternalSignals:
    def __init__(self, config=None):
        cfg = config or {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "RoostooBot/5.0"
        self.funding_extreme_high = cfg.get("funding_extreme_high", 0.0005)
        self.funding_extreme_low = cfg.get("funding_extreme_low", -0.0003)
        self.funding_history = deque(maxlen=100)
        self.oi_history = deque(maxlen=50)
        self._last_fetch = 0
        self._min_interval = cfg.get("min_fetch_interval", 120)

    def fetch(self, pair="BTC/USD"):
        now = time.time()
        if now - self._last_fetch < self._min_interval:
            return
        symbol = PAIR_TO_BINANCE.get(pair, "BTCUSDT")
        try:
            r = self.session.get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=5)
            r.raise_for_status()
            self.funding_history.append(float(r.json().get("lastFundingRate", 0)))
        except Exception:
            pass
        try:
            r = self.session.get(f"{BINANCE_FAPI}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=5)
            r.raise_for_status()
            self.oi_history.append(float(r.json().get("openInterest", 0)))
        except Exception:
            pass
        self._last_fetch = now

    def get_risk_scalar(self) -> float:
        if not self.funding_history:
            return 1.0
        rate = self.funding_history[-1]
        funding_sig = 0.0
        if rate > self.funding_extreme_high:
            funding_sig = -np.clip((rate - self.funding_extreme_high) / self.funding_extreme_high, 0, 1)
        elif rate < self.funding_extreme_low:
            funding_sig = np.clip((self.funding_extreme_low - rate) / abs(self.funding_extreme_low), 0, 1)
        oi_sig = 0.0
        if len(self.oi_history) >= 10:
            curr = np.mean(list(self.oi_history)[-5:])
            prev = np.mean(list(self.oi_history)[-10:-5])
            if prev > 0:
                oi_sig = float(np.clip(-(curr - prev) / prev * 10, -1, 1))
        composite = 0.6 * funding_sig + 0.4 * oi_sig
        return float(np.clip(0.8 + composite * 0.5, 0.3, 1.3))

    def get_state(self):
        return {
            "funding_rate": round(self.funding_history[-1], 6) if self.funding_history else None,
            "risk_scalar": round(self.get_risk_scalar(), 3),
        }
