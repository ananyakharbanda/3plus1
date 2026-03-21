"""
Roostoo Trading Bot — Main Entry Point
========================================
Runs the structural edge strategy on the Roostoo Mock Exchange.

Usage:
    export ROOSTOO_API_KEY="your_key"
    export ROOSTOO_SECRET_KEY="your_secret"
    python bot.py

Deploy on AWS EC2:
    nohup python bot.py > output.log 2>&1 &
    tail -f bot_log.jsonl

Architecture:
    Every 60 seconds, the bot:
    1. Fetches all coin prices from Roostoo
    2. Checks Shannon entropy (is the market orderly or random?)
    3. Scans for lead-lag opportunities across 8 coins
    4. Checks Binance funding rate and open interest
    5. Decides target allocation per coin
    6. Executes rebalancing trades via limit orders (0.05% fee)
    7. Logs everything for transparency and review
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from roostoo_api import RoostooClient
from strategy import Strategy

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # API credentials (set via environment variables)
    "api_key": os.environ.get("ROOSTOO_API_KEY", "YOUR_API_KEY"),
    "secret_key": os.environ.get("ROOSTOO_SECRET_KEY", "YOUR_SECRET_KEY"),

    # Polling interval in seconds (respects API rate limits)
    "poll_interval": 60,

    # Strategy parameters
    "strategy": {
        # Assets to trade
        "primary_assets": [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD",
            "SOL/USD", "DOGE/USD", "LTC/USD", "LINK/USD",
        ],
        # Entropy filter
        "entropy_window": 60,
        "entropy_bins": 10,
        # Lead-lag detector
        "ll_window": 30,
        "ll_max_lag": 5,
        "ll_min_corr": 0.3,
        "ll_move_threshold": 0.005,
        # Trend EMAs
        "ema_fast": 20,
        "ema_slow": 60,
        "ema_long": 200,
        # Position sizing
        "target_annual_vol": 0.15,
        "vol_lookback": 48,
        "max_per_asset": 0.30,
        "max_total_exposure": 0.85,
        "rebalance_threshold": 0.03,
        # Risk management
        "max_drawdown": 0.05,
        "recovery_threshold": 0.03,
        # External signals
        "external": {
            "funding_extreme_high": 0.0005,
            "funding_extreme_low": -0.0003,
            "min_fetch_interval": 120,
        },
    },

    # Use limit orders to halve fee costs (0.05% vs 0.1%)
    "use_limit_orders": True,
    "limit_order_timeout": 25,  # seconds before cancelling unfilled limits

    # Logging
    "log_file": "bot_log.jsonl",
}


# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class TradeLog:
    """Append-only JSONL log for every tick and trade."""
    def __init__(self, path: str):
        self.path = Path(path)

    def write(self, event: dict):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")


# ═══════════════════════════════════════════════════════════════════
# BOT
# ═══════════════════════════════════════════════════════════════════

class Bot:
    def __init__(self, config: dict):
        self.cfg = config
        self.client = RoostooClient(config["api_key"], config["secret_key"])
        self.strategy = Strategy(config["strategy"])
        self.log = TradeLog(config["log_file"])
        self.logger = logging.getLogger("Bot")

    # ── Portfolio state ────────────────────────────────────────────

    def get_current_state(self, wallet: dict, tickers: dict) -> tuple[dict, float]:
        """
        Calculate current allocations and portfolio value.
        Returns (allocations_dict, total_value).
        """
        usd = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)
        total = usd
        holdings = {}

        for coin, bal in wallet.items():
            if coin == "USD":
                continue
            held = bal.get("Free", 0) + bal.get("Lock", 0)
            pair = f"{coin}/USD"
            if held > 0 and pair in tickers:
                value = held * tickers[pair]["LastPrice"]
                total += value
                holdings[pair] = value

        allocs = {pair: val / total for pair, val in holdings.items()} if total > 0 else {}
        return allocs, total

    # ── Execution ──────────────────────────────────────────────────

    def rebalance(self, targets: dict, current: dict, portfolio_value: float,
                  wallet: dict, tickers: dict) -> int:
        """
        Move from current allocations to target allocations.
        Sells first (to free USD), then buys.
        Returns number of trades executed.
        """
        threshold = self.cfg["strategy"].get("rebalance_threshold", 0.03)
        trades = 0

        # Calculate deltas
        all_pairs = set(list(targets.keys()) + list(current.keys()))
        sells, buys = {}, {}
        for pair in all_pairs:
            delta = targets.get(pair, 0) - current.get(pair, 0)
            if abs(delta) < threshold:
                continue
            if delta < 0:
                sells[pair] = delta
            else:
                buys[pair] = delta

        # Execute sells first
        for pair, delta in sells.items():
            trades += self._execute(pair, "SELL", abs(delta), portfolio_value, wallet, tickers)

        # Then buys
        for pair, delta in buys.items():
            trades += self._execute(pair, "BUY", delta, portfolio_value, wallet, tickers)

        return trades

    def _execute(self, pair: str, side: str, alloc_delta: float,
                 pv: float, wallet: dict, tickers: dict) -> int:
        """Execute a single trade. Returns 1 if successful, 0 if skipped."""
        ticker = tickers.get(pair)
        if not ticker:
            return 0

        price = ticker.get("LastPrice", 0)
        if price <= 0:
            return 0

        coin = pair.split("/")[0]
        trade_qty = abs(alloc_delta * pv) / price

        # Check available balance
        if side == "BUY":
            available_usd = wallet.get("USD", {}).get("Free", 0)
            trade_qty = min(trade_qty, (available_usd * 0.998) / price)
        else:
            available_coin = wallet.get(coin, {}).get("Free", 0)
            trade_qty = min(trade_qty, available_coin)

        # Apply precision (6 for BTC, 4 for mid-caps, 2 for others)
        precision = 6 if coin == "BTC" else (4 if coin in ("ETH", "BNB", "SOL") else 2)
        trade_qty = round(trade_qty, precision)

        # Minimum order check
        if trade_qty <= 0 or trade_qty * price < 1.0:
            return 0

        # Place order
        if self.cfg["use_limit_orders"]:
            limit_price = self.strategy.get_limit_price(pair, side, ticker)
            if limit_price:
                result = self.client.place_order(pair, side, trade_qty, "LIMIT", limit_price)
            else:
                result = self.client.place_order(pair, side, trade_qty, "MARKET")
        else:
            result = self.client.place_order(pair, side, trade_qty, "MARKET")

        self.log.write({
            "event": "trade", "pair": pair, "side": side,
            "qty": trade_qty, "price": price, "alloc_delta": round(alloc_delta, 4),
        })
        return 1

    # ── Main loop ──────────────────────────────────────────────────

    def tick(self) -> bool:
        """Execute one cycle. Returns True on success."""
        # 1. Fetch all tickers
        tickers = self.client.all_tickers()
        if not tickers:
            self.logger.error("Failed to fetch tickers")
            return False

        # 2. Feed to strategy
        self.strategy.update(tickers)

        # 3. Fetch external signals (rate-limited internally)
        try:
            self.strategy.external.fetch("BTC/USD")
        except Exception:
            pass  # graceful degradation if Binance unreachable

        # 4. Get portfolio state
        wallet = self.client.balance()
        if not wallet:
            self.logger.error("Failed to fetch balance")
            return False

        current_allocs, portfolio_value = self.get_current_state(wallet, tickers)
        self.strategy.update_drawdown(portfolio_value)

        # 5. Warmup check
        if not self.strategy.is_ready():
            btc = self.strategy._assets.get("BTC/USD")
            btc_count = btc["_count"] if btc else 0
            btc_need = self.strategy.slow_period
            n = sum(1 for a in self.strategy._assets.values() if a["_count"] >= 60)
            mins_left = max(0, btc_need - btc_count)
            self.logger.info(
                f"Warming up... BTC: {btc_count}/{btc_need} ticks (~{mins_left} min left) | "
                f"{n} assets have 60+ ticks | Portfolio: ${portfolio_value:,.0f}"
            )
            return True

        # 6. Get target allocations
        targets = self.strategy.get_target_allocations(portfolio_value)
        state = self.strategy.get_state()

        cash_pct = 1.0 - sum(current_allocs.values())
        self.logger.info(
            f"${portfolio_value:,.0f} | Cash {cash_pct:.0%} | "
            f"Entropy={state.get('entropy', '?')} ({'TREND' if state.get('entropy_trending') else 'RANDOM'}) | "
            f"LL signals={len(state.get('lead_lag_signals', {}))} | "
            f"Targets={targets}"
        )

        # 7. Execute rebalance
        trades = self.rebalance(targets, current_allocs, portfolio_value, wallet, tickers)

        # 8. Cancel unfilled limit orders after timeout
        if self.cfg["use_limit_orders"] and trades > 0:
            time.sleep(self.cfg["limit_order_timeout"])
            for pair in targets:
                self.client.cancel_order(pair=pair)

        # 9. Log state
        self.log.write({
            "event": "tick", "portfolio_value": round(portfolio_value, 2),
            "current": {k: round(v, 4) for k, v in current_allocs.items()},
            "targets": targets, "trades": trades, "state": state,
        })

        return True

    def run(self):
        """Main entry point. Runs forever until interrupted."""
        setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("  ROOSTOO BOT — STRUCTURAL EDGE STRATEGY")
        self.logger.info(f"  Orders: {'LIMIT (0.05%)' if self.cfg['use_limit_orders'] else 'MARKET (0.1%)'}")
        self.logger.info(f"  Assets: {self.cfg['strategy']['primary_assets']}")
        self.logger.info(f"  Poll: {self.cfg['poll_interval']}s")
        self.logger.info("=" * 60)

        # Verify connection
        if not self.client.server_time():
            self.logger.error("Cannot connect to Roostoo API")
            return

        # Log available pairs
        pairs = self.client.get_all_pairs()
        self.logger.info(f"Exchange has {len(pairs)} tradeable pairs")

        # Initial portfolio value
        pv = self.client.get_portfolio_value()
        if pv:
            self.strategy.peak_value = pv
            self.logger.info(f"Starting portfolio: ${pv:,.2f}")

        # Run
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                self.logger.info("Shutting down")
                break
            except Exception as e:
                self.logger.exception(f"Error: {e}")
            time.sleep(self.cfg["poll_interval"])


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Bot(CONFIG).run()
