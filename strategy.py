"""
Strategy — DATA-DRIVEN FINAL
==============================
Built from analyzing 391 real trades and $60K in losses.

Root causes of losses:
    1. $20K burned in fees (391 trades, 157 tiny DOGE buys)
    2. $40K from buying into a falling market (BTC -3.8% over 3 days)
    3. Position data lost on restarts → cleanup couldn't work

Fixes:
    1. BTC REGIME FILTER — if BTC is falling, NO NEW BUYS for any coin
       Crypto is correlated. If BTC drops, everything drops.
       Go to cash and wait.
    2. Minimum trade $20K (not $5K) — eliminates tiny DOGE buys
    3. Rebalance threshold 8% — trade very rarely
    4. Min hold 60 ticks (1 hour) — stop churning
    5. Exit threshold 0.5% — very hard to flip out of a position
    6. When BTC recovers → deploy aggressively into top 2 coins
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Optional
import logging

from external_signals import ExternalSignals

logger = logging.getLogger(__name__)


class CoinEntropy:
    def __init__(self, window=40, n_bins=10):
        self.window = window
        self.n_bins = n_bins
        self._returns = defaultdict(lambda: deque(maxlen=window + 10))
        self._cache = {}
        self._history = defaultdict(lambda: deque(maxlen=100))

    def update(self, pair, ret):
        self._returns[pair].append(ret)
        if len(self._returns[pair]) >= self.window:
            rets = np.array(list(self._returns[pair])[-self.window:])
            counts, _ = np.histogram(rets, bins=self.n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = float(-np.sum(probs * np.log2(probs)))
            self._cache[pair] = h
            self._history[pair].append(h)

    def get(self, pair):
        return self._cache.get(pair)

    def bonus(self, pair):
        h = self.get(pair)
        if h is None:
            return 0.15
        hist = list(self._history.get(pair, []))
        if len(hist) < 10:
            return float(np.clip((2.78 - h) / 0.50 * 0.6, 0.0, 0.6))
        percentile = np.sum(np.array(hist) <= h) / len(hist)
        return float(np.clip((1.0 - percentile) * 0.6, 0.0, 0.6))


class LeadLagDetector:
    def __init__(self, window=30, max_lag=5, min_corr=0.25, move_threshold=0.002):
        self.window = window
        self.max_lag = max_lag
        self.min_corr = min_corr
        self.move_threshold = move_threshold
        self.returns = defaultdict(lambda: deque(maxlen=window + max_lag + 10))
        self.prices = defaultdict(lambda: deque(maxlen=max_lag + 5))
        self.relationships = []
        self._tick = 0
        self._signal_age = defaultdict(int)

    def update(self, pair, price, ret):
        self.returns[pair].append(ret)
        self.prices[pair].append(price)
        self._tick += 1
        if self._tick >= 30:
            self._recalculate()
            self._tick = 0

    def _recalculate(self):
        pairs = [p for p in self.returns if len(self.returns[p]) >= self.window + self.max_lag]
        if len(pairs) < 2:
            return
        results = []
        for leader in pairs:
            for follower in pairs:
                if leader == follower:
                    continue
                lag, corr = self._best_lag(leader, follower)
                if lag and abs(corr) >= self.min_corr:
                    results.append({"leader": leader, "follower": follower, "lag": lag, "corr": corr})
        results.sort(key=lambda x: abs(x["corr"]), reverse=True)
        self.relationships = results[:20]

    def _best_lag(self, leader, follower):
        lr = np.array(list(self.returns[leader]))
        fr = np.array(list(self.returns[follower]))
        n = min(len(lr), len(fr))
        best_lag, best_corr = None, 0.0
        for lag in range(1, self.max_lag + 1):
            if n - lag < 10:
                continue
            x = lr[-(n - lag):-lag]
            y = fr[-n + lag:]
            length = min(len(x), len(y))
            if length < 10:
                continue
            x, y = x[:length], y[:length]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            c = np.corrcoef(x, y)[0, 1]
            if not np.isnan(c) and abs(c) > abs(best_corr):
                best_corr, best_lag = c, lag
        return best_lag, best_corr

    def get_bullish_signals(self):
        signals = {}
        for rel in self.relationships:
            leader = rel["leader"]
            lag = rel["lag"]
            prices = list(self.prices.get(leader, []))
            if len(prices) < lag + 1 or prices[-lag - 1] <= 0:
                continue
            move = (prices[-1] / prices[-lag - 1]) - 1.0
            if move < self.move_threshold:
                continue
            signal = move / self.move_threshold * abs(rel["corr"])
            follower = rel["follower"]
            signal = float(np.clip(signal, 0, 1.0))
            if follower not in signals or signal > signals[follower]:
                signals[follower] = signal
        new_ages = {}
        for pair, sig in signals.items():
            new_ages[pair] = self._signal_age.get(pair, 0) + 1
        self._signal_age = new_ages
        return signals

    def get_signal_age(self, pair):
        return self._signal_age.get(pair, 0)


class Strategy:
    def __init__(self, config=None):
        cfg = config or {}

        self.primary_assets = cfg.get("primary_assets", [
            "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD",
            "XRP/USD", "DOGE/USD", "LINK/USD",
        ])

        self.entropy = CoinEntropy(cfg.get("entropy_window", 40), cfg.get("entropy_bins", 10))
        self.lead_lag = LeadLagDetector(
            cfg.get("ll_window", 30), cfg.get("ll_max_lag", 5),
            cfg.get("ll_min_corr", 0.25), cfg.get("ll_move_threshold", 0.002))
        self.external = ExternalSignals(cfg.get("external", {}))

        self.fast_period = cfg.get("ema_fast", 20)
        self.slow_period = cfg.get("ema_slow", 50)
        self.long_period = cfg.get("ema_long", 150)

        self.enter_threshold = cfg.get("enter_threshold", 0.001)
        self.exit_threshold = cfg.get("exit_threshold", 0.005)  # very hard to exit

        self.max_per_asset = cfg.get("max_per_asset", 0.45)
        self.max_total = cfg.get("max_total_exposure", 0.90)
        self.min_score = cfg.get("min_score", 10)
        self.rebalance_threshold = cfg.get("rebalance_threshold", 0.08)  # 8% — trade very rarely
        self.min_trade_usd = cfg.get("min_trade_usd", 20000)  # no trades under $20K
        self.max_coins = cfg.get("max_coins", 2)  # max 2 coins — concentrated bets

        self.btc_floor = cfg.get("btc_floor", 0.0)  # NO BTC FLOOR — if market is down, go full cash

        self.min_hold_ticks = cfg.get("min_hold_ticks", 60)  # 1 hour minimum hold

        self.profit_take_pct = cfg.get("profit_take_pct", 0.03)
        self.profit_take_sell = cfg.get("profit_take_sell", 0.40)

        self.max_drawdown = cfg.get("max_drawdown", 0.08)  # wider DD tolerance — don't panic sell
        self.recovery_threshold = cfg.get("recovery_threshold", 0.04)
        self.peak_value = 0.0
        self.circuit_breaker_active = False
        self._assets = {}
        self._entry_tick = {}
        self._tick = 0

    def update(self, tickers):
        self._tick += 1
        for pair, data in tickers.items():
            price = data.get("LastPrice", 0)
            if price <= 0:
                continue
            if pair not in self._assets:
                self._assets[pair] = {"prices": deque(maxlen=250), "returns": deque(maxlen=100),
                                      "ema_fast": None, "ema_slow": None, "ema_long": None, "_count": 0}
            a = self._assets[pair]
            prev = a["prices"][-1] if a["prices"] else None
            a["prices"].append(price)
            a["_count"] += 1
            if prev and prev > 0:
                ret = (price - prev) / prev
                a["returns"].append(ret)
                self.entropy.update(pair, ret)
                self.lead_lag.update(pair, price, ret)
            for name, period in [("ema_fast", self.fast_period), ("ema_slow", self.slow_period), ("ema_long", self.long_period)]:
                if a[name] is None:
                    if a["_count"] >= period:
                        a[name] = float(np.mean(list(a["prices"])[-period:]))
                else:
                    alpha = 2.0 / (period + 1)
                    a[name] = alpha * price + (1 - alpha) * a[name]

    def update_drawdown(self, pv):
        if pv > self.peak_value:
            self.peak_value = pv
        if self.peak_value <= 0:
            return
        dd = (self.peak_value - pv) / self.peak_value
        if self.circuit_breaker_active:
            if dd < self.recovery_threshold:
                logger.info(f"Circuit breaker OFF (DD={dd:.2%})")
                self.circuit_breaker_active = False
        elif dd >= self.max_drawdown:
            logger.warning(f"CIRCUIT BREAKER ON: DD={dd:.2%}")
            self.circuit_breaker_active = True

    def _ema_diff(self, pair):
        a = self._assets.get(pair)
        if not a or not a["ema_fast"] or not a["ema_slow"] or a["ema_slow"] == 0:
            return 0.0
        return (a["ema_fast"] - a["ema_slow"]) / a["ema_slow"]

    def _btc_regime(self):
        """
        THE MOST IMPORTANT FUNCTION.
        If BTC is falling → entire market is falling → NO NEW BUYS.
        Returns 'bull', 'bear', or 'neutral'.
        """
        btc = self._assets.get("BTC/USD")
        if not btc or not btc["ema_fast"] or not btc["ema_slow"]:
            return "neutral"
        diff = (btc["ema_fast"] - btc["ema_slow"]) / btc["ema_slow"]
        if diff > 0.001:
            return "bull"
        elif diff < -0.001:
            return "bear"
        return "neutral"

    def _direction(self, pair, holding=False):
        diff = self._ema_diff(pair)
        if holding:
            if diff > 0:
                return "up"
            elif diff < -self.exit_threshold:
                return "down"
            return "flat"
        else:
            if diff > self.enter_threshold:
                return "up"
            elif diff < -self.enter_threshold:
                return "down"
            return "flat"

    def _trend_strength(self, pair):
        diff = self._ema_diff(pair)
        if diff <= 0:
            return 0.0
        return float(np.clip(diff / 0.012, 0, 1))

    def _is_macro_bullish(self, pair):
        a = self._assets.get(pair)
        if not a or not a["ema_long"] or not a["prices"]:
            return False
        return a["prices"][-1] > a["ema_long"]

    def _holding(self, pair, positions):
        return positions is not None and pair in positions and positions[pair].get("qty", 0) > 0

    def _current_price(self, pair):
        a = self._assets.get(pair)
        if a and a["prices"]:
            return a["prices"][-1]
        return 0

    def _held_ticks(self, pair):
        return self._tick - self._entry_tick.get(pair, 0)

    def score_coin(self, pair, positions=None):
        a = self._assets.get(pair)
        if not a or a["_count"] < self.slow_period:
            return {"total": 0, "direction": "?", "trend": 0, "entropy": 0,
                    "lead_lag": 0, "macro": 0, "ent_raw": 0, "holding": False}

        holding = self._holding(pair, positions)
        direction = self._direction(pair, holding=holding)
        strength = self._trend_strength(pair)
        ent_bonus = self.entropy.bonus(pair)
        ll = self.lead_lag.get_bullish_signals()
        ll_str = ll.get(pair, 0)
        ll_age = self.lead_lag.get_signal_age(pair)
        macro = self._is_macro_bullish(pair)

        trend_score = 0.0
        ent_score = 0.0
        macro_score = 0.0

        if direction == "up":
            trend_score = strength * 45
            ent_score = ent_bonus * trend_score
            macro_score = 15.0 if macro else 0.0
        elif direction == "flat" and holding:
            trend_score = max(strength * 25, 5)
            macro_score = 10.0 if macro else 0.0

        ll_score = 0.0
        if ll_str > 0:
            ll_base = ll_str * 45.0
            if ll_age <= 10:
                ll_base *= 2.0
            ll_score = min(ll_base, 90.0)

        trend_path = trend_score + ent_score + macro_score
        total = max(trend_path, ll_score)
        if trend_path >= 8 and ll_score >= 8:
            total += 10

        ent_raw = self.entropy.get(pair)
        return {
            "total": round(float(total), 1),
            "direction": direction,
            "trend": round(float(trend_score), 1),
            "entropy": round(float(ent_score), 1),
            "lead_lag": round(float(ll_score), 1),
            "macro": round(float(macro_score), 1),
            "ent_raw": round(float(ent_raw), 3) if ent_raw is not None else 0,
            "holding": holding,
        }

    def get_target_allocations(self, pv, positions=None, just_freed_cash=False):
        if self.circuit_breaker_active:
            return {}

        regime = self._btc_regime()
        ext = self.external.get_risk_scalar()

        scored = {}
        for pair in self.primary_assets:
            s = self.score_coin(pair, positions)
            if s["total"] >= self.min_score:
                scored[pair] = s

        allocations = {}

        if regime == "bear" and not just_freed_cash:
            # BEAR: hold existing, no new buys
            logger.info(f"  BTC REGIME: BEAR — no new buys")
            if positions:
                for pair in positions:
                    if positions[pair].get("qty", 0) > 0 and pair in self.primary_assets:
                        direction = self._direction(pair, holding=True)
                        if direction != "down":
                            allocations[pair] = round(float(self.max_per_asset * 0.3), 4)
            return allocations

        if regime == "bear" and just_freed_cash:
            # BEAR but just freed cash from cleanup
            # Only buy something with genuine relative strength (score >= 20)
            logger.info(f"  BTC REGIME: BEAR but cash freed — looking for relative strength")
            strong = {p: s for p, s in scored.items() if s["direction"] == "up" and s["total"] >= 20}
            if strong:
                best_pair, best_score = max(strong.items(), key=lambda x: x[1]["total"])
                allocations[best_pair] = round(float(self.max_per_asset), 4)
                logger.info(f"  REINVEST: {best_pair} score={best_score['total']} — strongest in weak market")
            else:
                logger.info(f"  No strong coins found — cash stays as cash")
            # Keep existing holdings too
            if positions:
                for pair in positions:
                    if positions[pair].get("qty", 0) > 0 and pair in self.primary_assets and pair not in allocations:
                        direction = self._direction(pair, holding=True)
                        if direction != "down":
                            allocations[pair] = round(float(self.max_per_asset * 0.3), 4)
            return allocations

        # BULL or NEUTRAL: normal scoring
        if scored:
            ranked = sorted(scored.items(), key=lambda x: x[1]["total"], reverse=True)[:self.max_coins]
            ts = sum(s["total"] for _, s in ranked)
            if ts > 0:
                for pair, s in ranked:
                    alloc = (s["total"] / ts) * self.max_total * ext
                    alloc = min(alloc, self.max_per_asset)
                    if alloc >= 0.08:
                        allocations[pair] = round(float(alloc), 4)

        # Min hold: 60 ticks
        if positions:
            for pair in list(positions.keys()):
                if positions[pair].get("qty", 0) <= 0:
                    continue
                if pair not in self.primary_assets:
                    continue
                held = self._held_ticks(pair)
                if held < self.min_hold_ticks and pair not in allocations:
                    direction = self._direction(pair, holding=True)
                    avg = positions[pair].get("avg_price", 0)
                    cp = self._current_price(pair)
                    at_profit = (cp >= avg) if avg > 0 and cp > 0 else False

                    if direction == "down" and at_profit:
                        logger.info(f"  EMERGENCY EXIT {pair}: strong reversal + at profit ({held} ticks)")
                    elif direction == "down" and not at_profit:
                        # Strong downtrend but at a loss — hold anyway, don't crystallize loss
                        allocations[pair] = round(float(self.max_per_asset * 0.3), 4)
                        loss_pct = ((cp / avg - 1) * 100) if avg > 0 else 0
                        logger.info(f"  HOLD THROUGH DIP {pair}: downtrend but at {loss_pct:.1f}% loss — not selling")
                    else:
                        allocations[pair] = round(float(self.max_per_asset * 0.3), 4)
                        logger.info(f"  HOLD LOCK {pair}: {held}/{self.min_hold_ticks} ticks")

        # Profit-taking
        if positions:
            for pair, pos in positions.items():
                avg = pos.get("avg_price", 0)
                if avg > 0 and pair in self._assets and self._assets[pair]["prices"]:
                    cp = self._assets[pair]["prices"][-1]
                    gain = (cp / avg) - 1
                    if gain >= self.profit_take_pct and pair in allocations:
                        old = allocations[pair]
                        allocations[pair] = round(old * (1 - self.profit_take_sell), 4)
                        logger.info(f"  PROFIT TAKE {pair}: +{gain:.1%}")

        # ═══════════════════════════════════════════════════════
        # LOSS PROTECTION: never sell at a loss
        # If we hold a coin and it's not in targets (would be sold),
        # check if it's at a loss. If yes, force-keep it.
        # Only allow sells when position is at profit or breakeven.
        # ═══════════════════════════════════════════════════════
        if positions:
            for pair, pos_data in positions.items():
                if pos_data.get("qty", 0) <= 0:
                    continue
                avg = pos_data.get("avg_price", 0)
                if avg <= 0:
                    continue

                # Check if this position would be reduced or sold
                current_alloc_approx = (pos_data["qty"] * self._current_price(pair)) / pv if pv > 0 else 0
                target_alloc = allocations.get(pair, 0)

                if target_alloc < current_alloc_approx - 0.02:
                    # Strategy wants to reduce/sell this position
                    cp = self._current_price(pair)
                    if cp > 0 and cp < avg:
                        # AT A LOSS — do not sell, keep at current size
                        loss_pct = (cp / avg - 1) * 100
                        allocations[pair] = round(float(current_alloc_approx), 4)
                        logger.info(f"  LOSS PROTECT {pair}: at {loss_pct:.2f}% loss — holding, not selling")

        # Cap
        total = sum(allocations.values())
        if total > self.max_total:
            s = self.max_total / total
            allocations = {p: round(a * s, 4) for p, a in allocations.items()}

        # Track entries
        if positions:
            for pair in allocations:
                if pair not in positions and pair not in self._entry_tick:
                    self._entry_tick[pair] = self._tick
            for pair in list(self._entry_tick.keys()):
                if pair not in allocations and (not positions or pair not in positions):
                    del self._entry_tick[pair]

        return allocations

    def get_limit_price(self, pair, side, ticker):
        bid, ask = ticker.get("MaxBid", 0), ticker.get("MinAsk", 0)
        if bid <= 0 or ask <= 0:
            return None
        spread = ask - bid
        return round(bid + spread * 0.3, 2) if side == "BUY" else round(ask - spread * 0.3, 2)

    def get_state(self):
        scores = {}
        for pair in self.primary_assets:
            s = self.score_coin(pair)
            if s["total"] > 0:
                scores[pair] = s
        entropies = {}
        for pair in self.primary_assets:
            h = self.entropy.get(pair)
            if h is not None:
                entropies[pair] = round(h, 3)
        return {
            "coin_scores": scores, "entropies": entropies,
            "btc_regime": self._btc_regime(),
            "lead_lag_pairs": len(self.lead_lag.relationships),
            "lead_lag_signals": self.lead_lag.get_bullish_signals(),
            "external": self.external.get_state(),
            "circuit_breaker": self.circuit_breaker_active,
            "assets_ready": sum(1 for a in self._assets.values() if a["_count"] >= self.slow_period),
        }

    def is_ready(self):
        btc = self._assets.get("BTC/USD")
        return btc is not None and btc["_count"] >= self.slow_period and btc["ema_slow"] is not None
