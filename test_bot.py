"""
Roostoo Trading Bot v4 — Aggressive Structural Edge
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
        "ll_min_corr": 0.25, "ll_move_threshold": 0.002,
        "ema_fast": 20, "ema_slow": 60, "ema_long": 200,
        "max_per_asset": 0.35, "max_total_exposure": 0.85,
        "min_score": 15, "rebalance_threshold": 0.02,
        "max_coins": 4,
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
        self.pair_info = {}
        self.open_positions = {}  # {pair: {qty, avg_buy_price, total_cost}}
        self.total_realized_pnl = 0.0

    def get_current_state(self, wallet, tickers):
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
                unrealized = 0.0
                avg_buy = 0.0
                if pair in self.open_positions:
                    avg_buy = self.open_positions[pair]["avg_buy_price"]
                    unrealized = (price - avg_buy) * held
                positions[pair] = {"qty": round(held, 6), "price": round(price, 2),
                                   "avg_buy": round(avg_buy, 2), "value": round(value, 2),
                                   "pnl": round(unrealized, 2)}

        allocs = {p: v / total for p, v in holdings.items()} if total > 0 else {}
        for p in positions:
            positions[p]["pct"] = f"{allocs.get(p, 0):.1%}"
        positions["USD"] = {"free": round(usd_free, 2), "locked": round(usd_lock, 2),
                            "pct": f"{(usd_free + usd_lock) / total:.0%}" if total > 0 else "100%"}
        return allocs, total, positions

    def rebalance(self, targets, current, pv, wallet, tickers):
        threshold = self.cfg["strategy"].get("rebalance_threshold", 0.02)
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

        precision = 2
        min_order = 1.0
        if pair in self.pair_info:
            precision = self.pair_info[pair].get("AmountPrecision", 2)
            min_order = self.pair_info[pair].get("MiniOrder", 1.0)
        trade_qty = round(trade_qty, precision)

        if trade_qty <= 0 or (trade_qty * price) < min_order:
            return None

        trade_usd = round(trade_qty * price, 2)

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
        filled_price = detail.get("FilledAverPrice", 0) or price

        # Track P&L
        trade_pnl = None
        if success and status in ("FILLED", "PENDING"):
            if side == "BUY":
                pos = self.open_positions.get(pair, {"qty": 0, "total_cost": 0, "avg_buy_price": 0})
                new_qty = pos["qty"] + trade_qty
                new_cost = pos["total_cost"] + (trade_qty * filled_price)
                self.open_positions[pair] = {
                    "qty": new_qty, "total_cost": new_cost,
                    "avg_buy_price": new_cost / new_qty if new_qty > 0 else 0,
                }
            elif side == "SELL" and pair in self.open_positions:
                pos = self.open_positions[pair]
                if pos["avg_buy_price"] > 0:
                    trade_pnl = round((filled_price - pos["avg_buy_price"]) * trade_qty, 2)
                    self.total_realized_pnl += trade_pnl
                pos["qty"] = max(0, pos["qty"] - trade_qty)
                pos["total_cost"] = pos["qty"] * pos["avg_buy_price"]
                if pos["qty"] <= 0.000001:
                    del self.open_positions[pair]

        if success:
            extra = ""
            if side == "BUY" and pair in self.open_positions:
                extra = f" | Avg: ${self.open_positions[pair]['avg_buy_price']:,.2f}"
            if side == "SELL" and trade_pnl is not None:
                extra = f" | P&L: ${trade_pnl:+,.2f}"
            self.logger.info(f"  >>> {side} {trade_qty} {coin} @ ${filled_price:,.2f} (${trade_usd:,.0f}) [{order_type}] {status}{extra}")
        else:
            err = result.get("ErrMsg", "?") if result else "no response"
            self.logger.warning(f"  >>> FAILED {side} {pair}: {err}")

        self.log.write({"event": "trade", "pair": pair, "side": side, "qty": trade_qty,
                        "price": filled_price, "usd": trade_usd, "type": order_type,
                        "status": status, "success": success, "trade_pnl": trade_pnl})
        return {"pair": pair, "side": side, "success": success}

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

        if not self.strategy.is_ready():
            btc = self.strategy._assets.get("BTC/USD")
            ct = btc["_count"] if btc else 0
            self.logger.info(f"Warmup BTC:{ct}/{self.strategy.slow_period} | ${pv:,.0f}")
            return True

        targets = self.strategy.get_target_allocations(pv)
        state = self.strategy.get_state()

        # Position display
        pos_parts = []
        for pair, info in positions.items():
            if pair == "USD":
                continue
            pnl_s = f" P&L:${info['pnl']:+,.0f}" if info.get('pnl', 0) != 0 else ""
            pos_parts.append(f"{pair.split('/')[0]}={info['pct']}(${info['value']:,.0f}{pnl_s})")
        pos_str = " ".join(pos_parts) if pos_parts else "none"
        cash_pct = 1.0 - sum(allocs.values())

        self.logger.info(
            f"#{self.tick_count} | PV: ${pv:,.0f} | PnL: ${pnl:+,.0f} ({pnl_pct:+.2f}%) | "
            f"Realized: ${self.total_realized_pnl:+,.0f} | DD: {dd:.2%} | "
            f"Cash: {cash_pct:.0%} | Positions: {pos_str}"
        )

        # Per-coin entropy
        ent = state.get("entropies", {})
        if ent:
            ent_str = " ".join(f"{p.split('/')[0]}:{v}" for p, v in ent.items())
            self.logger.info(f"  Entropy per coin: {ent_str}")

        # Scores
        scores = state.get("coin_scores", {})
        if scores:
            score_parts = []
            for p, s in sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True):
                if s["total"] > 0:
                    score_parts.append(
                        f"{p.split('/')[0]}:{s['total']:.0f} (T:{s['trend']:.0f} E:{s['entropy']:.0f} "
                        f"LL:{s['lead_lag']:.0f} M:{s['macro']:.0f})"
                    )
            if score_parts:
                self.logger.info(f"  Scores: {' | '.join(score_parts[:6])}")

        # Lead-lag
        ll = state.get("lead_lag_signals", {})
        if ll:
            self.logger.info(f"  LL signals: {ll}")

        # Targets
        if targets:
            self.logger.info(f"  Targets: {targets}")
        else:
            self.logger.info(f"  Targets: NONE")

        # Open position detail
        for pair, pos in self.open_positions.items():
            cp = tickers.get(pair, {}).get("LastPrice", 0)
            if cp and pos["avg_buy_price"] > 0:
                upnl = (cp - pos["avg_buy_price"]) * pos["qty"]
                chg = ((cp / pos["avg_buy_price"]) - 1) * 100
                self.logger.info(
                    f"  Hold {pair}: {pos['qty']:.6f} @ ${pos['avg_buy_price']:,.2f} → "
                    f"${cp:,.2f} ({chg:+.2f}%) unrealized: ${upnl:+,.2f}"
                )

        # Execute
        trades = self.rebalance(targets, allocs, pv, wallet, tickers)

        if self.cfg["use_limit_orders"] and trades:
            time.sleep(self.cfg["limit_order_timeout"])
            for pair in targets:
                try:
                    self.client.cancel_order(pair=pair)
                except Exception:
                    pass

        self.log.write({
            "event": "tick", "tick": self.tick_count,
            "pv": round(pv, 2), "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 4),
            "realized": round(self.total_realized_pnl, 2), "dd": round(dd, 4),
            "cash_pct": round(cash_pct, 4), "positions": positions,
            "targets": targets, "trades": len(trades),
            "scores": {p: s["total"] for p, s in scores.items()} if scores else {},
            "entropies": ent, "ll_signals": ll,
            "external": state.get("external", {}),
            "cb": state.get("circuit_breaker", False),
        })
        return True

    def run(self):
        setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("  ROOSTOO BOT v4 — AGGRESSIVE STRUCTURAL EDGE")
        self.logger.info(f"  Orders: {'LIMIT' if self.cfg['use_limit_orders'] else 'MARKET'}")
        self.logger.info(f"  Assets: {self.cfg['strategy']['primary_assets']}")
        self.logger.info(f"  Max per coin: {self.cfg['strategy']['max_per_asset']:.0%} | Max total: {self.cfg['strategy']['max_total_exposure']:.0%}")
        self.logger.info("=" * 60)

        if not self.client.server_time():
            self.logger.error("Cannot connect to Roostoo API")
            return

        pairs = self.client.get_all_pairs()
        self.logger.info(f"Exchange has {len(pairs)} tradeable pairs")

        self.pair_info = self.client.get_pair_info()
        if self.pair_info:
            self.logger.info(f"Loaded precision for {len(self.pair_info)} pairs")

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
                self.logger.error(f"CRASH #{self.tick_count}: {e}")
                self.logger.error(traceback.format_exc())
                self.log.write({"event": "crash", "tick": self.tick_count, "error": str(e)})
                time.sleep(10)
                continue
            time.sleep(self.cfg["poll_interval"])

if __name__ == "__main__":
    Bot(CONFIG).run()
