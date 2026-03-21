"""
Roostoo Bot — FINAL Competition Version
"""

import os, sys, time, json, logging, traceback
from datetime import datetime, timezone
from pathlib import Path
from roostoo_api import RoostooClient
from strategy import Strategy
from telegram_notify import TelegramNotifier

CONFIG = {
    "api_key": os.environ.get("ROOSTOO_API_KEY", "YOUR_API_KEY"),
    "secret_key": os.environ.get("ROOSTOO_SECRET_KEY", "YOUR_SECRET_KEY"),
    "poll_interval": 60,
    "strategy": {
        "primary_assets": [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "SOL/USD",
            "DOGE/USD", "LTC/USD", "LINK/USD", "ONDO/USD", "AVAX/USD",
            "NEAR/USD", "DOT/USD",
        ],
        "entropy_window": 40, "entropy_bins": 10,
        "ll_window": 30, "ll_max_lag": 5, "ll_min_corr": 0.20, "ll_move_threshold": 0.002,
        "ema_fast": 12, "ema_slow": 30, "ema_long": 100,
        "max_per_asset": 0.40, "max_total_exposure": 0.90,
        "min_score": 8, "rebalance_threshold": 0.015, "max_coins": 4,
        "btc_floor": 0.15, "profit_take_pct": 0.03, "profit_take_sell": 0.40,
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
        self.initial_value = None
        self.tick_count = 0
        self.pair_info = {}
        self.positions = {}
        self.realized_pnl = 0.0
        self.tg = TelegramNotifier()

    def _state(self, wallet, tickers):
        usd = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)
        total = usd
        allocs, pos = {}, {}
        for coin, bal in wallet.items():
            if coin == "USD": continue
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
        if not tk: return None
        px = tk.get("LastPrice", 0)
        if px <= 0: return None
        coin = pair.split("/")[0]
        qty = abs(alloc_delta * pv) / px
        if side == "BUY":
            avail = wallet.get("USD", {}).get("Free", 0)
            qty = min(qty, (avail * 0.998) / px)
        else:
            avail = wallet.get(coin, {}).get("Free", 0)
            qty = min(qty, avail)
        prec = self.pair_info.get(pair, {}).get("AmountPrecision", 2)
        min_ord = self.pair_info.get(pair, {}).get("MiniOrder", 1.0)
        qty = round(qty, prec)
        if qty <= 0 or qty * px < min_ord: return None

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
                self.positions[pair] = {"qty": nq, "total_cost": nc, "avg_price": nc / nq if nq > 0 else 0}
            elif side == "SELL" and pair in self.positions:
                p = self.positions[pair]
                if p["avg_price"] > 0:
                    trade_pnl = round((fpx - p["avg_price"]) * qty, 2)
                    self.realized_pnl += trade_pnl
                p["qty"] = max(0, p["qty"] - qty)
                p["total_cost"] = p["qty"] * p["avg_price"]
                if p["qty"] < 0.000001: del self.positions[pair]

        if ok:
            extra = f" P&L:${trade_pnl:+,.2f}" if trade_pnl is not None else ""
            self.L.info(f"  {side} {qty} {coin} @ ${fpx:,.2f} (${usd_val:,.0f}) [{otype}]{extra}")
            self.dlog.w(f"  TRADE: {side} {qty} {pair} @ ${fpx:,.2f} = ${usd_val:,.2f} [{otype}] {status}{extra}")
            # Telegram alert
            avg = self.positions.get(pair, {}).get("avg_price", 0)
            self.tg.trade_alert(side, qty, pair, fpx, usd_val, otype, trade_pnl, avg)
        else:
            err = result.get("ErrMsg", "?") if result else "fail"
            self.L.warning(f"  FAIL {side} {pair}: {err}")
        self.jlog.j({"e": "trade", "pair": pair, "side": side, "qty": qty, "px": fpx, "pnl": trade_pnl})
        return ok

    def tick(self):
        self.tick_count += 1
        tickers = self.client.all_tickers()
        if not tickers: return
        self.strategy.update(tickers)
        try: self.strategy.external.fetch("BTC/USD")
        except: pass

        wallet = self.client.balance()
        if not wallet: return

        allocs, pv, pos, usd = self._state(wallet, tickers)
        self.strategy.update_drawdown(pv)
        if self.initial_value is None: self.initial_value = pv
        pnl = pv - self.initial_value
        pnl_pct = pnl / self.initial_value * 100 if self.initial_value else 0
        dd = (self.strategy.peak_value - pv) / self.strategy.peak_value if self.strategy.peak_value > 0 else 0
        cash_pct = 1 - sum(allocs.values())

        if not self.strategy.is_ready():
            ct = self.strategy._assets.get("BTC/USD", {}).get("_count", 0)
            self.L.info(f"Warmup {ct}/{self.strategy.slow_period} | ${pv:,.0f}")
            return

        # Pass open positions for profit-taking
        targets = self.strategy.get_target_allocations(pv, self.positions)
        state = self.strategy.get_state()

        # Console
        pos_str = " ".join(f"{p.split('/')[0]}:{allocs.get(p,0):.0%}" for p in pos) if pos else "empty"
        tgt_str = " ".join(f"{p.split('/')[0]}:{v:.0%}" for p, v in targets.items()) if targets else "CASH"
        self.L.info(f"#{self.tick_count} ${pv:,.0f} PnL:${pnl:+,.0f}({pnl_pct:+.1f}%) R:${self.realized_pnl:+,.0f} DD:{dd:.1%} Cash:{cash_pct:.0%} [{pos_str}] → [{tgt_str}]")

        # Detail log
        d = self.dlog
        now = datetime.now().strftime("%H:%M:%S")
        d.w(f"\n{'─'*70}")
        d.w(f"#{self.tick_count} {now} | ${pv:,.2f} | PnL:${pnl:+,.2f} ({pnl_pct:+.3f}%) | R:${self.realized_pnl:+,.2f} | DD:{dd:.3%}")
        if pos:
            d.w("POSITIONS:")
            for pair, info in pos.items():
                chg = ((info["price"] / info["avg"] - 1) * 100) if info["avg"] > 0 else 0
                d.w(f"  {pair:<12} {info['qty']:.4f} @ ${info['avg']:>10,.2f} → ${info['price']:>10,.2f} ({chg:+.2f}%) ${info['value']:>10,.2f} unrl:${info['pnl']:+,.2f}")
        d.w(f"  USD: ${usd:,.2f} ({cash_pct:.1%})")

        ent = state.get("entropies", {})
        if ent:
            d.w("ENTROPY:")
            for pair, h in sorted(ent.items(), key=lambda x: x[1]):
                d.w(f"  {pair:<12} {h:.3f} bonus:{self.strategy.entropy.bonus(pair):.2f}")

        scores = state.get("coin_scores", {})
        if scores:
            d.w(f"SCORES (min {self.strategy.min_score}):")
            for pair, s in sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True):
                d.w(f"  {pair:<12} {s['total']:>5.1f}  dir:{s['direction']:>4} T:{s['trend']:>5.1f} E:{s['entropy']:>5.1f} LL:{s['lead_lag']:>5.1f} M:{s['macro']:>5.1f} ent:{s['ent_raw']:.3f}")

        ll = state.get("lead_lag_signals", {})
        if ll:
            d.w(f"LEAD-LAG: {', '.join(f'{p}:{v:.2f}' for p,v in ll.items())}")

        d.w("TARGETS:")
        if targets:
            for p, a in sorted(targets.items(), key=lambda x: x[1], reverse=True):
                cur = allocs.get(p, 0)
                d.w(f"  {p:<12} {a:.1%} (current:{cur:.1%} delta:{a-cur:+.1%})")
        else:
            d.w("  ALL CASH")

        # Execute — sells first, then buys with remaining cash
        threshold = self.cfg["strategy"]["rebalance_threshold"]
        all_pairs = set(list(targets) + list(allocs))
        sells, buys = {}, {}
        for pair in all_pairs:
            delta = targets.get(pair, 0) - allocs.get(pair, 0)
            if abs(delta) >= threshold:
                (sells if delta < 0 else buys)[pair] = delta
        tc = 0
        for pair, delta in sells.items():
            if self._exec(pair, "SELL", abs(delta), pv, wallet, tickers): tc += 1

        # Refresh wallet after sells so buys see updated cash
        if sells:
            fresh_wallet = self.client.balance()
            if fresh_wallet:
                wallet = fresh_wallet

        # Buy in order of allocation size (biggest first)
        for pair, delta in sorted(buys.items(), key=lambda x: x[1], reverse=True):
            if self._exec(pair, "BUY", delta, pv, wallet, tickers):
                tc += 1
                # Deduct spent cash from wallet so next buy sees correct balance
                spent = abs(delta * pv)
                usd_free = wallet.get("USD", {}).get("Free", 0)
                wallet.setdefault("USD", {})["Free"] = max(0, usd_free - spent)

        if self.cfg["use_limit_orders"] and tc > 0:
            time.sleep(self.cfg["limit_order_timeout"])
            for pair in targets:
                try: self.client.cancel_order(pair=pair)
                except: pass

        self.jlog.j({"e": "tick", "n": self.tick_count, "pv": round(pv, 2), "pnl": round(pnl, 2),
                     "rpnl": round(self.realized_pnl, 2), "dd": round(dd, 4),
                     "tgt": {p: round(float(v), 4) for p, v in targets.items()}, "trades": tc})

        # Telegram hourly update
        self.tg.hourly_update(pv, pnl, pnl_pct, self.realized_pnl, dd, cash_pct, pos, targets)

    def run(self):
        setup_logging()
        self.L.info("=" * 50)
        self.L.info("  ROOSTOO BOT — FINAL COMPETITION VERSION")
        self.L.info(f"  Warmup: ~{self.cfg['strategy']['ema_slow']} min")
        self.L.info(f"  Detail: {self.cfg['detail_log']}")
        self.L.info("=" * 50)
        if not self.client.server_time():
            self.L.error("Cannot connect"); return
        self.pair_info = self.client.get_pair_info()
        self.L.info(f"Loaded {len(self.pair_info)} pairs")
        pv = self.client.get_portfolio_value()
        if pv:
            self.strategy.peak_value = pv
            self.initial_value = pv
            self.L.info(f"Starting: ${pv:,.2f}")
        self.dlog.w(f"\n{'#'*70}\nSTARTED {datetime.now()} | ${pv:,.2f}\n{'#'*70}")
        self.tg.bot_started(pv or 0, len(self.pair_info))

        while True:
            try: self.tick()
            except KeyboardInterrupt: self.L.info("Stopped"); break
            except Exception as e:
                self.L.error(f"CRASH #{self.tick_count}: {e}")
                self.dlog.w(f"\nCRASH #{self.tick_count}:\n{traceback.format_exc()}")
                self.tg.bot_crash(self.tick_count, str(e))
                time.sleep(10); continue
            time.sleep(self.cfg["poll_interval"])

if __name__ == "__main__":
    Bot(CONFIG).run()
