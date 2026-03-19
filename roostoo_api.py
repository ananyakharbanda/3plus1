"""
Roostoo API Client
==================
Handles all communication with the Roostoo Mock Exchange.

API Reference: https://github.com/roostoo/Roostoo-API-Documents
Base URL: https://mock-api.roostoo.com

Authentication:
    - Signed endpoints use HMAC-SHA256
    - API key sent in header as RST-API-KEY
    - Signature sent in header as MSG-SIGNATURE
    - Timestamp must be within 60s of server time

Portfolio: $1,000,000 USD initial balance
Fee structure: 0.1% taker (market), 0.05% maker (limit)
"""

import requests
import time
import hmac
import hashlib
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

BASE_URL = "https://mock-api.roostoo.com"


class RoostooClient:
    """Thread-safe API client for Roostoo Mock Exchange."""

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()

    # ── Authentication helpers ─────────────────────────────────────

    def _timestamp(self) -> str:
        """13-digit millisecond timestamp as required by API."""
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> tuple[dict, dict, str]:
        """
        Create HMAC-SHA256 signature for signed endpoints.
        Returns (headers, params_with_timestamp, encoded_body).
        """
        params["timestamp"] = self._timestamp()
        # Sort params alphabetically, join as key=value&key=value
        body = "&".join(f"{k}={params[k]}" for k in sorted(params))
        sig = hmac.new(
            self.secret_key.encode("utf-8"),
            body.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": sig}
        return headers, params, body

    # ── HTTP methods ───────────────────────────────────────────────

    def _get(self, path: str, params: dict = None, signed: bool = False) -> Optional[dict]:
        """Perform GET request. Returns parsed JSON or None on failure."""
        try:
            if signed:
                headers, params, _ = self._sign(params or {})
                resp = self.session.get(f"{BASE_URL}{path}", headers=headers, params=params, timeout=10)
            else:
                resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GET {path} failed: {e}")
            return None

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        """Perform signed POST request."""
        try:
            headers, _, body = self._sign(payload)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            resp = self.session.post(f"{BASE_URL}{path}", headers=headers, data=body, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"POST {path} failed: {e}")
            return None

    # ── Public endpoints (no auth) ─────────────────────────────────

    def server_time(self) -> Optional[int]:
        """GET /v3/serverTime - Test connectivity, get server timestamp."""
        data = self._get("/v3/serverTime")
        return data.get("ServerTime") if data else None

    def exchange_info(self) -> Optional[dict]:
        """
        GET /v3/exchangeInfo - Trading rules and available pairs.
        Returns: {IsRunning, InitialWallet, TradePairs: {pair: {Coin, PricePrecision, AmountPrecision, MiniOrder}}}
        """
        return self._get("/v3/exchangeInfo")

    def get_all_pairs(self) -> List[str]:
        """Return list of all tradeable pair names like ['BTC/USD', 'ETH/USD', ...]."""
        info = self.exchange_info()
        if not info or not info.get("TradePairs"):
            return []
        return [pair for pair, details in info["TradePairs"].items() if details.get("CanTrade")]

    def get_pair_info(self) -> Dict[str, dict]:
        """Return {pair: {PricePrecision, AmountPrecision, MiniOrder}} for all pairs."""
        info = self.exchange_info()
        if not info:
            return {}
        return info.get("TradePairs", {})

    def all_tickers(self) -> Optional[Dict[str, dict]]:
        """
        GET /v3/ticker - All market tickers.
        Returns: {pair: {MaxBid, MinAsk, LastPrice, Change, CoinTradeValue, UnitTradeValue}}
        """
        data = self._get("/v3/ticker", {"timestamp": self._timestamp()})
        if data and data.get("Success"):
            return data["Data"]
        return None

    def ticker(self, pair: str) -> Optional[dict]:
        """GET /v3/ticker?pair=X - Single pair ticker."""
        data = self._get("/v3/ticker", {"timestamp": self._timestamp(), "pair": pair})
        if data and data.get("Success"):
            return data["Data"].get(pair)
        return None

    # ── Signed endpoints (require auth) ────────────────────────────

    def balance(self) -> Optional[dict]:
        """
        GET /v3/balance - Current wallet balances.
        Returns: {coin: {Free: float, Lock: float}, ...}
        """
        data = self._get("/v3/balance", signed=True)
        if data and data.get("Success"):
            # API returns either "Wallet" or "SpotWallet" depending on version
            return data.get("Wallet") or data.get("SpotWallet") or {}
        return None

    def place_order(self, pair: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: float = None) -> Optional[dict]:
        """
        POST /v3/place_order - Place a new order.

        Args:
            pair: Trading pair e.g. "BTC/USD"
            side: "BUY" or "SELL"
            quantity: Amount of coin to trade
            order_type: "MARKET" (0.1% fee) or "LIMIT" (0.05% fee)
            price: Required for LIMIT orders

        Returns: {Success, OrderDetail: {OrderID, Status, FilledAverPrice, ...}}
        """
        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                logger.error("LIMIT order requires price parameter")
                return None
            payload["price"] = str(price)

        result = self._post("/v3/place_order", payload)
        if result and result.get("Success"):
            d = result.get("OrderDetail", {})
            logger.info(f"ORDER {side} {quantity} {pair} @ {d.get('FilledAverPrice', price)} [{d.get('Status')}]")
        else:
            logger.warning(f"Order failed: {result}")
        return result

    def query_order(self, order_id: int = None, pair: str = None,
                    pending_only: bool = None) -> Optional[dict]:
        """POST /v3/query_order - Query order history."""
        payload = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair:
                payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
        return self._post("/v3/query_order", payload)

    def cancel_order(self, order_id: int = None, pair: str = None) -> Optional[dict]:
        """POST /v3/cancel_order - Cancel pending orders."""
        payload = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        return self._post("/v3/cancel_order", payload)

    # ── Convenience methods ────────────────────────────────────────

    def get_portfolio_value(self, tickers: dict = None) -> Optional[float]:
        """Calculate total portfolio value in USD."""
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
