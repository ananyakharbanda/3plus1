"""
Roostoo Trading Bot — Main Entry Point
========================================
"""

import os, sys, time, json, logging, traceback
from datetime import datetime, timezone
from pathlib import Path
from roostoo_api import RoostooClient
from strategy import Strategy

CONFIG = {
    "api_key": os.environ.get("ROOSTOO_API_KEY", "YOUR_API_KEY"),
    "secret_key": os.environ.get("ROOSTOO_SECRET_KEY", "YOUR_SECRET_KEY"),
    "poll_interval": 60,
    "strategy": {
        "primary_assets": [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD",
            "SOL/USD", "DOGE/USD", "LTC/USD", "LINK/USD",
        ],
        "entropy_window": 60, "entropy_bins": 10,
        "ll_window": 30, "ll_max_lag": 5,
        "ll_min_corr": 0.25, "ll_move_threshold": 0.003,
        "ema_fast": 20, "ema_slow": 60, "ema_long": 200,
        "target_annual_vol": 0.15, "vol_lookback": 48,
        "max_per_asset": 0.30, "max_total_exposure": 0.85,
        "rebalance_threshold": 0.03,
        "max_drawdown": 0.05, "recovery_threshold": 0.03,
        "external": {"funding_extreme_high": 0.0005, "funding_extreme_low": -0.0003, "min_fetch_interval": 120},
    },
    "use_limit_orders": True,
    "limit_order_timeout": 25,
    "log_file": "bot_log.jsonl",
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

class TradeLog:
    def __init__(self, path):
        self.path = Path(path)
    def write(self, event):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass

class Bot:
    def __init__(self, config):
        self.cfg = config
        self.client = RoostooClient(config["api_key"], config["secret_key"])
        self.strategy = Strategy(config["strategy"])
        self.log = TradeLog(config["log_file"])
        self.logger = logging.getLogger("Bot")
        self.initial_value = None
        self.tick_count = 0
        self.pair_info = {}  # {pair: {AmountPrecision, PricePrecision, MiniOrder}}

    def get_current_state(self, wallet, tickers):
        """Returns (allocations, total_value, positions_detail)."""
        usd_free = wallet.get("USD", {}).get("Free", 0)
        usd_lock = wallet.get("USD", {}).get("Lock", 0)
        total = usd_free + usd_lock
        holdings = {}
        positions = {}

        for coin, bal in wallet.items():
            if coin == "USD":
                continue
            held = bal.get("Free", 0) + bal.get("Lock", 0)
            pair = f"{coin}/USD"
            if held > 0 and pair in tickers:
                price = tickers[pair]["LastPrice"]
                value = held * price
                total += value
                holdings[pair] = value
                positions[pair] = {"qty": round(held, 6), "price": round(price, 2),
                                   "value": round(value, 2)}

        allocs = {p: v / total for p, v in holdings.items()} if total > 0 else {}
        for p in positions:
            positions[p]["pct"] = f"{allocs.get(p, 0):.1%}"
        positions["USD"] = {"free": round(usd_free, 2), "locked": round(usd_lock, 2),
                            "pct": f"{(usd_free + usd_lock) / total:.0%}" if total > 0 else "100%"}
        return allocs, total, positions

    def rebalance(self, targets, current, pv, wallet, tickers):
        threshold = self.cfg["strategy"].get("rebalance_threshold", 0.03)
        trades = []
        all_pairs = set(list(targets.keys()) + list(current.keys()))
        sells, buys = {}, {}
        for pair in all_pairs:
            delta = targets.get(pair, 0) - current.get(pair, 0)
            if abs(delta) < threshold:
                continue
            (sells if delta < 0 else buys)[pair] = delta

        for pair, delta in sells.items():
            t = self._execute(pair, "SELL", abs(delta), pv, wallet, tickers)
            if t: trades.append(t)
        for pair, delta in buys.items():
            t = self._execute(pair, "BUY", delta, pv, wallet, tickers)
            if t: trades.append(t)
        return trades

    def _execute(self, pair, side, alloc_delta, pv, wallet, tickers):
        ticker = tickers.get(pair)
        if not ticker:
            return None
        price = ticker.get("LastPrice", 0)
        if price <= 0:
            return None

        coin = pair.split("/")[0]
        trade_qty = abs(alloc_delta * pv) / price

        if side == "BUY":
            avail = wallet.get("USD", {}).get("Free", 0)
            trade_qty = min(trade_qty, (avail * 0.998) / price)
        else:
            avail = wallet.get(coin, {}).get("Free", 0)
            trade_qty = min(trade_qty, avail)

        precision = 2  # safe default
        min_order = 1.0
        if pair in self.pair_info:
            precision = self.pair_info[pair].get("AmountPrecision", 2)
            min_order = self.pair_info[pair].get("MiniOrder", 1.0)
        trade_qty = round(trade_qty, precision)
        if trade_qty <= 0 or (trade_qty * price) < min_order:
            return None

        trade_usd = round(trade_qty * price, 2)

        # Try limit first, fallback to market
        result = None
        order_type = "MARKET"
        if self.cfg["use_limit_orders"]:
            lp = self.strategy.get_limit_price(pair, side, ticker)
            if lp:
                order_type = "LIMIT"
                result = self.client.place_order(pair, side, trade_qty, "LIMIT", lp)

        if not result or not result.get("Success"):
            order_type = "MARKET"
            result = self.client.place_order(pair, side, trade_qty, "MARKET")

        success = result.get("Success", False) if result else False
        detail = result.get("OrderDetail", {}) if result else {}
        status = detail.get("Status", "FAILED")
        filled = detail.get("FilledAverPrice", 0)

        info = {"pair": pair, "side": side, "qty": trade_qty, "price": price,
                "filled": filled, "usd": trade_usd, "type": order_type,
                "status": status, "success": success}

        if success:
            self.logger.info(f"  >>> {side} {trade_qty} {pair} @ ${filled or price:,.2f} (${trade_usd:,.0f}) [{order_type}] {status}")
        else:
            err = result.get("ErrMsg", "unknown") if result else "no response"
            self.logger.warning(f"  >>> FAILED {side} {pair}: {err}")
            info["error"] = err

        self.log.write({"event": "trade", **info})
        return info

    def tick(self):
        self.tick_count += 1

        tickers = self.client.all_tickers()
        if not tickers:
            self.logger.error("Failed to fetch tickers")
            return False

        self.strategy.update(tickers)

        try:
            self.strategy.external.fetch("BTC/USD")
        except Exception:
            pass

        wallet = self.client.balance()
        if not wallet:
            self.logger.error("Failed to fetch balance")
            return False

        allocs, pv, positions = self.get_current_state(wallet, tickers)
        self.strategy.update_drawdown(pv)

        if self.initial_value is None:
            self.initial_value = pv

        pnl = pv - self.initial_value
        pnl_pct = (pnl / self.initial_value * 100) if self.initial_value else 0
        dd = (self.strategy.peak_value - pv) / self.strategy.peak_value if self.strategy.peak_value > 0 else 0

        # Warmup
        if not self.strategy.is_ready():
            btc = self.strategy._assets.get("BTC/USD")
            ct = btc["_count"] if btc else 0
            self.logger.info(f"Warmup BTC:{ct}/{self.strategy.slow_period} | ${pv:,.0f}")
            return True

        targets = self.strategy.get_target_allocations(pv)
        state = self.strategy.get_state()

        # Build position string
        pos_parts = []
        for pair, info in positions.items():
            if pair == "USD":
                continue
            pos_parts.append(f"{pair.split('/')[0]}={info['pct']}(${info['value']:,.0f})")
        pos_str = " ".join(pos_parts) if pos_parts else "none"
        cash_pct = 1.0 - sum(allocs.values())

        # ── Main log line ──
        self.logger.info(
            f"#{self.tick_count} | PV: ${pv:,.0f} | PnL: ${pnl:+,.0f} ({pnl_pct:+.2f}%) | "
            f"DD: {dd:.2%} | Cash: {cash_pct:.0%} | Positions: {pos_str}"
        )
        self.logger.info(
            f"  Entropy: {state.get('entropy', 0):.3f} ({'TREND' if state.get('entropy_trending') else 'RANDOM'}) | "
            f"LL pairs: {state.get('lead_lag_pairs', 0)} | LL signals: {state.get('lead_lag_signals', {})} | "
            f"Ext risk: {state.get('external', {}).get('risk_scalar', 1.0):.2f} | "
            f"CB: {'ON' if state.get('circuit_breaker') else 'off'}"
        )
        if targets:
            self.logger.info(f"  Targets: {targets}")
        else:
            self.logger.info(f"  Targets: NONE (all cash)")

        # ── Conviction scores for debugging ──
        scores = state.get("conviction_scores", {})
        if scores:
            score_str = " | ".join(f"{p.split('/')[0]}:{s:.3f}" for p, s in scores.items() if s != 0)
            if score_str:
                self.logger.info(f"  Scores: {score_str}")

        # Execute
        trades = self.rebalance(targets, allocs, pv, wallet, tickers)

        if self.cfg["use_limit_orders"] and trades:
            time.sleep(self.cfg["limit_order_timeout"])
            for pair in targets:
                try:
                    self.client.cancel_order(pair=pair)
                except Exception:
                    pass

        # Full state log
        self.log.write({
            "event": "tick", "tick": self.tick_count,
            "pv": round(pv, 2), "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 4),
            "dd": round(dd, 4), "cash_pct": round(cash_pct, 4),
            "positions": positions, "targets": targets,
            "trades": len(trades), "entropy": state.get("entropy"),
            "trending": state.get("entropy_trending"),
            "ll_pairs": state.get("lead_lag_pairs", 0),
            "ll_signals": state.get("lead_lag_signals", {}),
            "external": state.get("external", {}),
            "cb": state.get("circuit_breaker", False),
        })
        return True

    def run(self):
        setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("  ROOSTOO BOT — STRUCTURAL EDGE STRATEGY")
        self.logger.info(f"  Orders: {'LIMIT' if self.cfg['use_limit_orders'] else 'MARKET'}")
        self.logger.info(f"  Assets: {self.cfg['strategy']['primary_assets']}")
        self.logger.info("=" * 60)

        if not self.client.server_time():
            self.logger.error("Cannot connect to Roostoo API")
            return

        pairs = self.client.get_all_pairs()
        self.logger.info(f"Exchange has {len(pairs)} tradeable pairs")

        # Load precision info for correct order quantities
        self.pair_info = self.client.get_pair_info()
        if self.pair_info:
            btc_info = self.pair_info.get("BTC/USD", {})
            self.logger.info(f"BTC precision: amount={btc_info.get('AmountPrecision')}, price={btc_info.get('PricePrecision')}, min=${btc_info.get('MiniOrder')}")
        else:
            self.logger.warning("Could not load pair info — using defaults")

        pv = self.client.get_portfolio_value()
        if pv:
            self.strategy.peak_value = pv
            self.initial_value = pv
            self.logger.info(f"Starting portfolio: ${pv:,.2f}")

        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                self.logger.info("Shutting down")
                break
            except Exception as e:
                self.logger.error(f"CRASH tick #{self.tick_count}: {e}")
                self.logger.error(traceback.format_exc())
                self.log.write({"event": "crash", "tick": self.tick_count,
                                "error": str(e), "tb": traceback.format_exc()})
                time.sleep(10)  # wait and retry, don't exit
                continue
            time.sleep(self.cfg["poll_interval"])

if __name__ == "__main__":
    Bot(CONFIG).run()
