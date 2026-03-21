"""
Roostoo API Client
==================
API Docs: https://github.com/roostoo/Roostoo-API-Documents
Base URL: https://mock-api.roostoo.com
Auth: HMAC-SHA256 (RST-API-KEY + MSG-SIGNATURE headers)
Fees: 0.1% taker (market), 0.05% maker (limit)
"""

import requests, time, hmac, hashlib, logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)
BASE_URL = "https://mock-api.roostoo.com"


class RoostooClient:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()

    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> tuple:
        params["timestamp"] = self._ts()
        body = "&".join(f"{k}={params[k]}" for k in sorted(params))
        sig = hmac.new(self.secret_key.encode(), body.encode(), hashlib.sha256).hexdigest()
        return {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": sig}, params, body

    def _get(self, path, params=None, signed=False):
        try:
            if signed:
                h, params, qs = self._sign(params or {})
                resp = self.session.get(f"{BASE_URL}{path}?{qs}", headers=h, timeout=10)
            else:
                resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GET {path}: {e}")
            return None

    def _post(self, path, payload):
        try:
            h, _, body = self._sign(payload)
            h["Content-Type"] = "application/x-www-form-urlencoded"
            resp = self.session.post(f"{BASE_URL}{path}", headers=h, data=body, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"POST {path}: {e}")
            return None

    def server_time(self):
        return (self._get("/v3/serverTime") or {}).get("ServerTime")

    def exchange_info(self):
        return self._get("/v3/exchangeInfo")

    def get_all_pairs(self) -> List[str]:
        info = self.exchange_info()
        if not info or not info.get("TradePairs"):
            return []
        return [p for p, d in info["TradePairs"].items() if d.get("CanTrade")]

    def get_pair_info(self) -> Dict[str, dict]:
        info = self.exchange_info()
        return info.get("TradePairs", {}) if info else {}

    def all_tickers(self) -> Optional[Dict[str, dict]]:
        data = self._get("/v3/ticker", {"timestamp": self._ts()})
        if data and data.get("Success"):
            return data["Data"]
        return None

    def ticker(self, pair: str) -> Optional[dict]:
        data = self._get("/v3/ticker", {"timestamp": self._ts(), "pair": pair})
        if data and data.get("Success"):
            return data["Data"].get(pair)
        return None

    def balance(self) -> Optional[dict]:
        data = self._get("/v3/balance", signed=True)
        if data and data.get("Success"):
            return data.get("Wallet") or data.get("SpotWallet") or {}
        return None

    def place_order(self, pair, side, quantity, order_type="MARKET", price=None):
        payload = {"pair": pair, "side": side.upper(), "type": order_type.upper(),
                   "quantity": str(quantity)}
        if order_type.upper() == "LIMIT":
            if price is None:
                return None
            payload["price"] = str(price)
        result = self._post("/v3/place_order", payload)
        if result and result.get("Success"):
            d = result.get("OrderDetail", {})
            logger.info(f"ORDER {side} {quantity} {pair} @ {d.get('FilledAverPrice', price)} [{d.get('Status')}]")
        else:
            logger.warning(f"Order failed: {result}")
        return result

    def cancel_order(self, order_id=None, pair=None):
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        return self._post("/v3/cancel_order", payload)

    def query_order(self, pair=None, pending_only=None):
        payload = {}
        if pair:
            payload["pair"] = pair
        if pending_only is not None:
            payload["pending_only"] = "TRUE" if pending_only else "FALSE"
        return self._post("/v3/query_order", payload)

    def get_portfolio_value(self, tickers=None):
        wallet = self.balance()
        if not wallet:
            return None
        if tickers is None:
            tickers = self.all_tickers()
        if not tickers:
            return None
        total = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)
        for coin, bal in wallet.items():
            if coin == "USD":
                continue
            held = bal.get("Free", 0) + bal.get("Lock", 0)
            if held > 0 and f"{coin}/USD" in tickers:
                total += held * tickers[f"{coin}/USD"]["LastPrice"]
        return total
