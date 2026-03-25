"""
Roostoo Bot — DATA-DRIVEN FINAL
=================================
Fixes based on analyzing 391 trades and $60K in losses:
1. BTC regime filter — no buys in bear market
2. Position data saved to disk — survives restarts
3. Min trade $20K — no more $55 DOGE buys
4. 8% rebalance threshold — trade very rarely
5. 60-tick minimum hold — stop churning
"""

import os, sys, time, json, logging, traceback
from datetime import datetime, timezone
from pathlib import Path
from roostoo_api import RoostooClient
from strategy import Strategy
from telegram_notify import TelegramNotifier

COMPETITION_START_VALUE = 1_000_000.0

CONFIG = {
    "api_key": os.environ.get("ROOSTOO_API_KEY", "YOUR_API_KEY"),
    "secret_key": os.environ.get("ROOSTOO_SECRET_KEY", "YOUR_SECRET_KEY"),
    "poll_interval": 60,
    "strategy": {
        "primary_assets": [
            "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD",
            "XRP/USD", "DOGE/USD", "LINK/USD",
        ],
        "entropy_window": 40, "entropy_bins": 10,
        "ll_window": 30, "ll_max_lag": 5, "ll_min_corr": 0.25, "ll_move_threshold": 0.002,
        "ema_fast": 20, "ema_slow": 50, "ema_long": 150,
        "max_per_asset": 0.45, "max_total_exposure": 0.90,
        "min_score": 10, "rebalance_threshold": 0.08, "min_trade_usd": 20000,
        "max_coins": 2, "btc_floor": 0.0, "min_hold_ticks": 60,
        "profit_take_pct": 0.03, "profit_take_sell": 0.40,
        "max_drawdown": 0.08, "recovery_threshold": 0.04,
        "enter_threshold": 0.001, "exit_threshold": 0.005,
        "external": {"funding_extreme_high": 0.0005, "funding_extreme_low": -0.0003, "min_fetch_interval": 120},
    },
    "use_limit_orders": True,
    "limit_order_timeout": 25,
    "log_file": "bot_log.jsonl",
    "detail_log": "bot_detail.log",
    "positions_file": "positions.json",  # persists across restarts
}

WANTED_COINS = set(CONFIG["strategy"]["primary_assets"])
MIN_TRADE_USD = CONFIG["strategy"].get("min_trade_usd", 20000)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

class FileLog:
    def __init__(self, p):
        self.p = Path(p)
    def j(self, e):
        e["ts"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.p, "a") as f: f.write(json.dumps(e, default=str) + "\n")
        except Exception: pass

class DetailLog:
    def __init__(self, p):
        self.p = Path(p)
    def w(self, t):
        try:
            with open(self.p, "a") as f: f.write(t + "\n")
        except Exception: pass


class Bot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = RoostooClient(cfg["api_key"], cfg["secret_key"])
        self.strategy = Strategy(cfg["strategy"])
        self.jlog = FileLog(cfg["log_file"])
        self.dlog = DetailLog(cfg["detail_log"])
        self.L = logging.getLogger("Bot")
        self.initial_value = COMPETITION_START_VALUE
        self.tick_count = 0
        self.pair_info = {}
        self.positions = {}
        self.realized_pnl = 0.0
        self.tg = TelegramNotifier()
        self.positions_file = cfg.get("positions_file", "positions.json")

        # Load saved positions from disk
        self._load_positions()

    def _load_positions(self):
        """Load positions from disk so restarts don't lose avg_buy_price."""
        try:
            if Path(self.positions_file).exists():
                with open(self.positions_file) as f:
                    saved = json.load(f)
                self.positions = saved.get("positions", {})
                self.realized_pnl = saved.get("realized_pnl", 0.0)
                self.L.info(f"Loaded {len(self.positions)} positions from disk (realized: ${self.realized_pnl:+,.2f})")
        except Exception as e:
            self.L.warning(f"Could not load positions: {e}")

    def _save_positions(self):
        """Save positions to disk after every trade."""
        try:
            with open(self.positions_file, "w") as f:
                json.dump({"positions": self.positions, "realized_pnl": self.realized_pnl}, f, indent=2)
        except Exception:
            pass

    def _state(self, wallet, tickers):
        usd = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)
        total = usd
        allocs, pos = {}, {}
        for coin, bal in wallet.items():
            if coin == "USD":
                continue
            held = bal.get("Free", 0) + bal.get("Lock", 0)
            pair = f"{coin}/USD"
            if held > 0 and pair in tickers:
                px = tickers[pair]["LastPrice"]
                val = held * px
                total += val
                avg = self.positions.get(pair, {}).get("avg_price", 0)
                upnl = (px - avg) * held if avg > 0 else 0
                pos[pair] = {"qty": held, "price": px, "avg": avg, "value": val, "pnl": upnl}
        if total > 0:
            allocs = {p: d["value"] / total for p, d in pos.items()}
        return allocs, total, pos, usd

    def _exec(self, pair, side, alloc_delta, pv, wallet, tickers):
        tk = tickers.get(pair)
        if not tk:
            return None
        px = tk.get("LastPrice", 0)
        if px <= 0:
            return None

        trade_usd = abs(alloc_delta * pv)
        if trade_usd < MIN_TRADE_USD:
            return None

        coin = pair.split("/")[0]
        qty = trade_usd / px

        if side == "BUY":
            avail = wallet.get("USD", {}).get("Free", 0)
            qty = min(qty, (avail * 0.998) / px)
        else:
            avail = wallet.get(coin, {}).get("Free", 0)
            qty = min(qty, avail)

        prec = self.pair_info.get(pair, {}).get("AmountPrecision", 2)
        min_ord = self.pair_info.get(pair, {}).get("MiniOrder", 1.0)
        qty = round(qty, prec)
        if qty <= 0 or qty * px < min_ord:
            return None

        result, otype = None, "MARKET"
        if self.cfg["use_limit_orders"]:
            lp = self.strategy.get_limit_price(pair, side, tk)
            if lp:
                otype = "LIMIT"
                result = self.client.place_order(pair, side, qty, "LIMIT", lp)
        if not result or not result.get("Success"):
            otype = "MARKET"
            result = self.client.place_order(pair, side, qty, "MARKET")

        ok = result.get("Success", False) if result else False
        det = result.get("OrderDetail", {}) if result else {}
        fpx = det.get("FilledAverPrice", 0) or px
        status = det.get("Status", "FAIL")
        usd_val = round(qty * px, 2)
        trade_pnl = None

        if ok and status in ("FILLED", "PENDING"):
            if side == "BUY":
                p = self.positions.get(pair, {"qty": 0, "total_cost": 0, "avg_price": 0})
                nq = p["qty"] + qty
                nc = p["total_cost"] + qty * fpx
                self.positions[pair] = {"qty": nq, "total_cost": nc,
                                        "avg_price": nc / nq if nq > 0 else 0}
            elif side == "SELL" and pair in self.positions:
                p = self.positions[pair]
                if p["avg_price"] > 0:
                    trade_pnl = round((fpx - p["avg_price"]) * qty, 2)
                    self.realized_pnl += trade_pnl
                p["qty"] = max(0, p["qty"] - qty)
                p["total_cost"] = p["qty"] * p["avg_price"]
                if p["qty"] < 0.000001:
                    del self.positions[pair]
            self._save_positions()  # persist after every trade

        if ok:
            extra = f" P&L:${trade_pnl:+,.2f}" if trade_pnl is not None else ""
            self.L.info(f"  {side} {qty} {coin} @ ${fpx:,.2f} (${usd_val:,.0f}) [{otype}]{extra}")
            self.dlog.w(f"  TRADE: {side} {qty} {pair} @ ${fpx:,.2f} = ${usd_val:,.2f} [{otype}] {status}{extra}")
            avg = self.positions.get(pair, {}).get("avg_price", 0)
            self.tg.trade_alert(side, qty, pair, fpx, usd_val, otype, trade_pnl, avg)
        else:
            err = result.get("ErrMsg", "?") if result else "fail"
            self.L.warning(f"  FAIL {side} {pair}: {err}")

        self.jlog.j({"e": "trade", "pair": pair, "side": side, "qty": qty, "px": fpx, "pnl": trade_pnl})
        return ok

    def _cleanup_unwanted(self, pos, tickers, wallet):
        """Sell unwanted coins ONLY when profitable."""
        sold_any = False
        for pair, info in list(pos.items()):
            if pair in WANTED_COINS:
                continue
            avg = info.get("avg", 0)
            price = info.get("price", 0)
            qty = info.get("qty", 0)
            if avg <= 0 or price <= 0 or qty <= 0:
                # If avg is 0, try to get from saved positions
                avg = self.positions.get(pair, {}).get("avg_price", 0)
                if avg <= 0:
                    continue
            pnl_pct = (price / avg - 1) * 100
            coin = pair.split("/")[0]
            if pnl_pct >= 0:
                self.L.info(f"  CLEANUP: selling {coin} (+{pnl_pct:.2f}%)")
                avail = wallet.get(coin, {}).get("Free", 0)
                if avail > 0:
                    prec = self.pair_info.get(pair, {}).get("AmountPrecision", 2)
                    sell_qty = round(avail, prec)
                    if sell_qty > 0:
                        result = self.client.place_order(pair, "SELL", sell_qty, "MARKET")
                        ok = result.get("Success", False) if result else False
                        if ok:
                            det = result.get("OrderDetail", {})
                            fpx = det.get("FilledAverPrice", 0) or price
                            trade_pnl = round((fpx - avg) * sell_qty, 2)
                            self.realized_pnl += trade_pnl
                            self.L.info(f"  SOLD {sell_qty} {coin} @ ${fpx:,.2f} P&L:${trade_pnl:+,.2f}")
                            self.tg.trade_alert("SELL", sell_qty, pair, fpx, round(sell_qty * fpx, 2), "MARKET", trade_pnl, 0)
                            if pair in self.positions:
                                del self.positions[pair]
                            self._save_positions()
                            sold_any = True
            else:
                self.L.info(f"  HOLDING {coin}: {pnl_pct:.2f}% loss — waiting")
        return sold_any

    def tick(self):
        self.tick_count += 1
        tickers = self.client.all_tickers()
        if not tickers:
            return
        self.strategy.update(tickers)
        try:
            self.strategy.external.fetch("BTC/USD")
        except Exception:
            pass

        wallet = self.client.balance()
        if not wallet:
            return

        allocs, pv, pos, usd = self._state(wallet, tickers)
        self.strategy.update_drawdown(pv)

        pnl = pv - COMPETITION_START_VALUE
        pnl_pct = pnl / COMPETITION_START_VALUE * 100
        dd = (self.strategy.peak_value - pv) / self.strategy.peak_value if self.strategy.peak_value > 0 else 0
        cash_pct = 1 - sum(allocs.values())

        # Warmup
        if not self.strategy.is_ready():
            self._cleanup_unwanted(pos, tickers, wallet)
            ct = self.strategy._assets.get("BTC/USD", {}).get("_count", 0) if "BTC/USD" in self.strategy._assets else 0
            self.L.info(f"Warmup {ct}/{self.strategy.slow_period} | ${pv:,.0f} | From $1M: ${pnl:+,.0f}")
            return

        # Cleanup unwanted
        cleaned = self._cleanup_unwanted(pos, tickers, wallet)
        if cleaned:
            wallet = self.client.balance()
            if wallet:
                allocs, pv, pos, usd = self._state(wallet, tickers)
                cash_pct = 1 - sum(allocs.values())
                pnl = pv - COMPETITION_START_VALUE
                pnl_pct = pnl / COMPETITION_START_VALUE * 100

        targets = self.strategy.get_target_allocations(pv, self.positions, just_freed_cash=cleaned)
        state = self.strategy.get_state()
        regime = state.get("btc_regime", "?")

        # Console — compact with regime
        pos_str = " ".join(f"{p.split('/')[0]}:{allocs.get(p,0):.0%}" for p in pos) if pos else "empty"
        tgt_str = " ".join(f"{p.split('/')[0]}:{v:.0%}" for p, v in targets.items()) if targets else "CASH"
        self.L.info(
            f"#{self.tick_count} ${pv:,.0f} From$1M:${pnl:+,.0f}({pnl_pct:+.2f}%) "
            f"R:${self.realized_pnl:+,.0f} DD:{dd:.1%} BTC:{regime} Cash:{cash_pct:.0%} [{pos_str}] → [{tgt_str}]"
        )

        # Detail log
        d = self.dlog
        now = datetime.now().strftime("%H:%M:%S")
        d.w(f"\n{'─'*70}")
        d.w(f"#{self.tick_count} {now} | ${pv:,.2f} | From $1M: ${pnl:+,.2f} ({pnl_pct:+.3f}%) | R:${self.realized_pnl:+,.2f} | DD:{dd:.3%}")
        d.w(f"BTC REGIME: {regime}")
        if pos:
            d.w("POSITIONS:")
            for pair, info in pos.items():
                chg = ((info["price"] / info["avg"] - 1) * 100) if info["avg"] > 0 else 0
                tag = "✓" if pair in WANTED_COINS else "✗"
                d.w(f"  {pair:<12} {info['qty']:.4f} @ ${info['avg']:>10,.2f} → ${info['price']:>10,.2f} ({chg:+.2f}%) ${info['value']:>10,.2f} unrl:${info['pnl']:+,.2f} {tag}")
        d.w(f"  USD: ${usd:,.2f} ({cash_pct:.1%})")

        scores = state.get("coin_scores", {})
        if scores:
            d.w(f"SCORES (min {self.strategy.min_score}):")
            for pair, s in sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True):
                d.w(f"  {pair:<12} {s['total']:>5.1f}  dir:{s['direction']:>4} T:{s['trend']:>5.1f} E:{s['entropy']:>5.1f} LL:{s['lead_lag']:>5.1f} M:{s['macro']:>5.1f}")

        d.w("TARGETS:")
        if targets:
            for p, a in sorted(targets.items(), key=lambda x: x[1], reverse=True):
                d.w(f"  {p:<12} {a:.1%} (current:{allocs.get(p,0):.1%})")
        else:
            d.w("  ALL CASH" if regime != "bear" else "  BEAR MARKET — holding, no new buys")

        # Execute
        threshold = self.cfg["strategy"]["rebalance_threshold"]
        all_pairs = set(list(targets) + [p for p in allocs if p in WANTED_COINS])
        sells, buys = {}, {}
        for pair in all_pairs:
            delta = targets.get(pair, 0) - allocs.get(pair, 0)
            if abs(delta) >= threshold:
                (sells if delta < 0 else buys)[pair] = delta

        tc = 0
        for pair, delta in sells.items():
            if self._exec(pair, "SELL", abs(delta), pv, wallet, tickers):
                tc += 1

        if sells:
            fresh = self.client.balance()
            if fresh:
                wallet = fresh

        for pair, delta in sorted(buys.items(), key=lambda x: x[1], reverse=True):
            if self._exec(pair, "BUY", delta, pv, wallet, tickers):
                tc += 1
                spent = abs(delta * pv)
                usd_free = wallet.get("USD", {}).get("Free", 0)
                wallet.setdefault("USD", {})["Free"] = max(0, usd_free - spent)

        if self.cfg["use_limit_orders"] and tc > 0:
            time.sleep(self.cfg["limit_order_timeout"])
            for pair in targets:
                try:
                    self.client.cancel_order(pair=pair)
                except Exception:
                    pass

        self.jlog.j({"e": "tick", "n": self.tick_count, "pv": round(pv, 2),
                     "pnl_from_1m": round(pnl, 2), "rpnl": round(self.realized_pnl, 2),
                     "dd": round(dd, 4), "regime": regime,
                     "tgt": {p: round(float(v), 4) for p, v in targets.items()}, "trades": tc})

        self.tg.hourly_update(pv, pnl, pnl_pct, self.realized_pnl, dd, cash_pct, pos, targets)

    def run(self):
        setup_logging()
        self.L.info("=" * 50)
        self.L.info("  ROOSTOO BOT — DATA-DRIVEN FINAL")
        self.L.info(f"  Min trade: ${MIN_TRADE_USD:,}")
        self.L.info(f"  Rebalance: {CONFIG['strategy']['rebalance_threshold']:.0%}")
        self.L.info(f"  Hold: {CONFIG['strategy']['min_hold_ticks']} ticks min")
        self.L.info(f"  Max coins: {CONFIG['strategy']['max_coins']}")
        self.L.info(f"  Positions file: {self.positions_file}")
        self.L.info("=" * 50)
        if not self.client.server_time():
            self.L.error("Cannot connect")
            return
        self.pair_info = self.client.get_pair_info()
        self.L.info(f"Loaded {len(self.pair_info)} pairs")
        pv = self.client.get_portfolio_value()
        if pv:
            self.strategy.peak_value = pv
            self.L.info(f"Current: ${pv:,.2f} | From $1M: ${pv - COMPETITION_START_VALUE:+,.2f}")
        self.dlog.w(f"\n{'#'*70}\nSTARTED {datetime.now()} | ${pv:,.2f}\n{'#'*70}")
        self.tg.bot_started(pv or 0, len(self.pair_info))

        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                self.L.info("Stopped")
                break
            except Exception as e:
                self.L.error(f"CRASH #{self.tick_count}: {e}")
                self.dlog.w(f"\nCRASH #{self.tick_count}:\n{traceback.format_exc()}")
                self.tg.bot_crash(self.tick_count, str(e))
                time.sleep(10)
                continue
            time.sleep(self.cfg["poll_interval"])

if __name__ == "__main__":
    Bot(CONFIG).run()
