"""
Roostoo Bot v5 — Clean console + detailed file logs
=====================================================
Console: compact 2-line summary per tick
File (bot_detail.log): full scorecard, positions, entropy, everything
File (bot_log.jsonl): machine-readable JSON for analysis
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
            "ONDO/USD", "AVAX/USD", "NEAR/USD", "DOT/USD",
        ],
        "entropy_window": 60, "entropy_bins": 10,
        "ll_window": 30, "ll_max_lag": 5,
        "ll_min_corr": 0.25, "ll_move_threshold": 0.002,
        "ema_fast": 20, "ema_slow": 60, "ema_long": 200,
        "max_per_asset": 0.35, "max_total_exposure": 0.85,
        "min_score": 10, "rebalance_threshold": 0.02,
        "max_coins": 4,
        "max_drawdown": 0.05, "recovery_threshold": 0.03,
        "external": {"funding_extreme_high": 0.0005, "funding_extreme_low": -0.0003, "min_fetch_interval": 120},
    },
    "use_limit_orders": True,
    "limit_order_timeout": 25,
    "log_file": "bot_log.jsonl",
    "detail_log": "bot_detail.log",
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

class DetailLog:
    """Human-readable detailed log file."""
    def __init__(self, path):
        self.path = Path(path)
    def write(self, text):
        try:
            with open(self.path, "a") as f:
                f.write(text + "\n")
        except Exception:
            pass


class Bot:
    def __init__(self, config):
        self.cfg = config
        self.client = RoostooClient(config["api_key"], config["secret_key"])
        self.strategy = Strategy(config["strategy"])
        self.log = TradeLog(config["log_file"])
        self.detail = DetailLog(config["detail_log"])
        self.logger = logging.getLogger("Bot")
        self.initial_value = None
        self.tick_count = 0
        self.pair_info = {}
        self.open_positions = {}
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
                avg_buy = self.open_positions.get(pair, {}).get("avg_buy_price", 0)
                unrealized = (price - avg_buy) * held if avg_buy > 0 else 0
                positions[pair] = {"qty": round(held, 6), "price": round(price, 2),
                                   "avg_buy": round(avg_buy, 2), "value": round(value, 2),
                                   "pnl": round(unrealized, 2)}
        allocs = {p: v / total for p, v in holdings.items()} if total > 0 else {}
        for p in positions:
            positions[p]["pct"] = f"{allocs.get(p, 0):.1%}"
        positions["USD"] = {"free": round(usd_free, 2), "locked": round(usd_lock, 2)}
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

        trade_pnl = None
        if success and status in ("FILLED", "PENDING"):
            if side == "BUY":
                pos = self.open_positions.get(pair, {"qty": 0, "total_cost": 0, "avg_buy_price": 0})
                new_qty = pos["qty"] + trade_qty
                new_cost = pos["total_cost"] + (trade_qty * filled_price)
                self.open_positions[pair] = {"qty": new_qty, "total_cost": new_cost,
                    "avg_buy_price": new_cost / new_qty if new_qty > 0 else 0}
            elif side == "SELL" and pair in self.open_positions:
                pos = self.open_positions[pair]
                if pos["avg_buy_price"] > 0:
                    trade_pnl = round((filled_price - pos["avg_buy_price"]) * trade_qty, 2)
                    self.total_realized_pnl += trade_pnl
                pos["qty"] = max(0, pos["qty"] - trade_qty)
                pos["total_cost"] = pos["qty"] * pos["avg_buy_price"]
                if pos["qty"] <= 0.000001:
                    del self.open_positions[pair]

        # Console: one clean line
        if success:
            pnl_str = f" P&L:${trade_pnl:+,.2f}" if trade_pnl is not None else ""
            self.logger.info(f"  {'📈' if side=='BUY' else '📉'} {side} {trade_qty} {coin} @ ${filled_price:,.2f} (${trade_usd:,.0f}) [{order_type}]{pnl_str}")
        else:
            err = result.get("ErrMsg", "?") if result else "no response"
            self.logger.warning(f"  FAILED {side} {pair}: {err}")

        # Detail log: everything
        self.detail.write(f"  TRADE: {side} {trade_qty} {pair} @ ${filled_price:,.2f} (${trade_usd:,.2f}) [{order_type}] {status}")
        if trade_pnl is not None:
            self.detail.write(f"    Realized P&L: ${trade_pnl:+,.2f} | Total realized: ${self.total_realized_pnl:+,.2f}")
        if side == "BUY" and pair in self.open_positions:
            self.detail.write(f"    Avg buy price: ${self.open_positions[pair]['avg_buy_price']:,.2f}")

        self.log.write({"event": "trade", "pair": pair, "side": side, "qty": trade_qty,
                        "price": filled_price, "usd": trade_usd, "type": order_type,
                        "status": status, "success": success, "trade_pnl": trade_pnl})
        return {"pair": pair, "side": side, "success": success, "pnl": trade_pnl}

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

        # ═══════════════════════════════════════════════════════
        # CONSOLE: compact 2 lines
        # ═══════════════════════════════════════════════════════
        pos_parts = []
        for pair, info in positions.items():
            if pair == "USD":
                continue
            coin = pair.split("/")[0]
            pnl_s = f" {'+' if info.get('pnl',0)>=0 else ''}{info['pnl']:.0f}" if info.get('pnl', 0) != 0 else ""
            pos_parts.append(f"{coin}:{info['pct']}{pnl_s}")
        pos_str = " ".join(pos_parts) if pos_parts else "cash only"
        cash_pct = 1.0 - sum(allocs.values())

        tgt_str = " ".join(f"{p.split('/')[0]}:{v:.0%}" for p, v in targets.items()) if targets else "NONE"

        self.logger.info(
            f"#{self.tick_count} ${pv:,.0f} PnL:${pnl:+,.0f}({pnl_pct:+.2f}%) DD:{dd:.1%} "
            f"Cash:{cash_pct:.0%} | {pos_str} | → {tgt_str}"
        )

        # ═══════════════════════════════════════════════════════
        # DETAIL FILE: everything
        # ═══════════════════════════════════════════════════════
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        d = self.detail
        d.write(f"\n{'='*80}")
        d.write(f"TICK #{self.tick_count} | {now}")
        d.write(f"{'='*80}")
        d.write(f"Portfolio: ${pv:,.2f} | PnL: ${pnl:+,.2f} ({pnl_pct:+.3f}%) | Realized: ${self.total_realized_pnl:+,.2f}")
        d.write(f"Drawdown: {dd:.3%} | Peak: ${self.strategy.peak_value:,.2f} | Circuit breaker: {'ON' if state.get('circuit_breaker') else 'off'}")
        d.write(f"Cash: {cash_pct:.1%} (${pv * cash_pct:,.2f})")

        # Positions
        d.write(f"\nPOSITIONS:")
        for pair, info in positions.items():
            if pair == "USD":
                d.write(f"  USD: ${info['free']:,.2f} free, ${info['locked']:,.2f} locked")
            else:
                pnl_pct_pos = ((info['price'] / info['avg_buy'] - 1) * 100) if info.get('avg_buy', 0) > 0 else 0
                d.write(f"  {pair}: {info['qty']} @ avg ${info['avg_buy']:,.2f} → ${info['price']:,.2f} ({pnl_pct_pos:+.2f}%) = ${info['value']:,.2f} (unrealized: ${info['pnl']:+,.2f})")

        # Entropy per coin
        ent = state.get("entropies", {})
        if ent:
            d.write(f"\nENTROPY (lower = more orderly):")
            for pair, h in sorted(ent.items(), key=lambda x: x[1]):
                trending = "TREND" if h < 2.75 else "NOISY"
                bar = "█" * int((2.9 - h) / 0.065) if h < 2.9 else ""
                d.write(f"  {pair:<12} {h:.3f} [{trending}] {bar}")

        # Scores
        scores = state.get("coin_scores", {})
        if scores:
            d.write(f"\nSCORECARD (need {self.strategy.min_score}+ to trade):")
            for pair, s in sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True):
                d.write(f"  {pair:<12} TOTAL:{s['total']:>5.1f} | dir:{s['direction']:>4} T:{s['trend']:>5.1f} E:{s['entropy']:>5.1f} LL:{s['lead_lag']:>5.1f} M:{s['macro']:>5.1f} | ent:{s['ent_raw']:.3f}")

        # Lead-lag
        ll = state.get("lead_lag_signals", {})
        if ll:
            d.write(f"\nLEAD-LAG SIGNALS:")
            for pair, sig in sorted(ll.items(), key=lambda x: x[1], reverse=True):
                d.write(f"  {pair}: {sig:+.3f}")
        d.write(f"  Relationships tracked: {state.get('lead_lag_pairs', 0)}")

        # External
        ext = state.get("external", {})
        if ext:
            d.write(f"\nEXTERNAL (Binance):")
            d.write(f"  Funding rate: {ext.get('funding_rate', '?')} | Signal: {ext.get('funding_signal', 0):.3f}")
            d.write(f"  OI signal: {ext.get('oi_signal', 0):.3f} | Risk scalar: {ext.get('risk_scalar', 1.0):.3f}")

        # Targets
        d.write(f"\nTARGETS:")
        if targets:
            for pair, alloc in sorted(targets.items(), key=lambda x: x[1], reverse=True):
                current = allocs.get(pair, 0)
                delta = alloc - current
                action = "BUY more" if delta > 0.02 else ("SELL some" if delta < -0.02 else "hold")
                d.write(f"  {pair:<12} target:{alloc:.1%} current:{current:.1%} delta:{delta:+.1%} → {action}")
        else:
            d.write(f"  ALL CASH")

        # Execute
        trades = self.rebalance(targets, allocs, pv, wallet, tickers)

        if self.cfg["use_limit_orders"] and trades:
            time.sleep(self.cfg["limit_order_timeout"])
            for pair in targets:
                try:
                    self.client.cancel_order(pair=pair)
                except Exception:
                    pass

        # JSON log
        self.log.write({
            "event": "tick", "tick": self.tick_count,
            "pv": round(pv, 2), "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 4),
            "realized": round(self.total_realized_pnl, 2), "dd": round(dd, 4),
            "targets": targets, "trades": len(trades),
            "scores": {p: s["total"] for p, s in scores.items()} if scores else {},
        })
        return True

    def run(self):
        setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("  ROOSTOO BOT v5 — AGGRESSIVE STRUCTURAL EDGE")
        self.logger.info(f"  Assets: {self.cfg['strategy']['primary_assets']}")
        self.logger.info(f"  Max: {self.cfg['strategy']['max_per_asset']:.0%}/coin, {self.cfg['strategy']['max_total_exposure']:.0%} total")
        self.logger.info(f"  Logs: console (compact) + {self.cfg['detail_log']} (full detail)")
        self.logger.info("=" * 60)

        if not self.client.server_time():
            self.logger.error("Cannot connect to Roostoo API")
            return

        pairs = self.client.get_all_pairs()
        self.logger.info(f"Exchange: {len(pairs)} pairs")
        self.pair_info = self.client.get_pair_info()

        pv = self.client.get_portfolio_value()
        if pv:
            self.strategy.peak_value = pv
            self.initial_value = pv
            self.logger.info(f"Starting: ${pv:,.2f}")

        self.detail.write(f"\n{'#'*80}")
        self.detail.write(f"BOT STARTED at {datetime.now()}")
        self.detail.write(f"Portfolio: ${pv:,.2f}" if pv else "Portfolio: unknown")
        self.detail.write(f"Assets: {self.cfg['strategy']['primary_assets']}")
        self.detail.write(f"{'#'*80}")

        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                self.logger.info("Shutting down")
                break
            except Exception as e:
                self.logger.error(f"CRASH #{self.tick_count}: {e}")
                self.detail.write(f"\nCRASH #{self.tick_count}: {e}\n{traceback.format_exc()}")
                self.log.write({"event": "crash", "tick": self.tick_count, "error": str(e)})
                time.sleep(10)
                continue
            time.sleep(self.cfg["poll_interval"])

if __name__ == "__main__":
    Bot(CONFIG).run()
