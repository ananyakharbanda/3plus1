"""
Strategy — FINAL Competition Version
======================================
Incorporates ALL observations from live testing + competition optimizations.

Live-tested fixes:
    - Never buy downtrends (LINK bug)
    - Entropy amplifies trend, never replaces it
    - Lead-lag uses cumulative returns (was never firing)
    - Per-coin entropy with normalized bonuses
    - No vol targeting on 1-min data (was crushing allocations)

Competition optimizations:
    - Faster EMAs (12/30/100) → 30 min warmup instead of 60
    - BTC floor: always hold 15% BTC unless strong downtrend
    - Higher exposure: 90% max, 40% per coin
    - Lower rebalance threshold: 1.5% (more responsive)
    - Profit-taking: sell 40% of position when up 3%+
    - Lead-lag allocation increased to 25%
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Optional, List, Tuple
import logging

from external_signals import ExternalSignals

logger = logging.getLogger(__name__)


class CoinEntropy:
    """Per-coin entropy with self-normalizing bonus."""

    def __init__(self, window=40, n_bins=10):
        self.window = window
        self.n_bins = n_bins
        self._returns = defaultdict(lambda: deque(maxlen=window + 10))
        self._cache = {}
        self._history = defaultdict(lambda: deque(maxlen=30))  # track entropy history per coin

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

    def get(self, pair) -> Optional[float]:
        return self._cache.get(pair)

    def bonus(self, pair) -> float:
        """
        Self-normalizing entropy bonus per coin.
        Compares current entropy to that coin's own recent range.
        Returns [0.0, 0.6]: 0.6 = very orderly for THIS coin.

        This fixes the problem where LINK (range 1.3-1.5) and BTC (range 2.6-2.8)
        had completely different scales but were compared with the same threshold.
        """
        h = self.get(pair)
        if h is None:
            return 0.15  # small default during warmup

        hist = list(self._history.get(pair, []))
        if len(hist) < 5:
            # Not enough history — use global calibration from live data
            return float(np.clip((2.78 - h) / 0.50 * 0.6, 0.0, 0.6))

        # Normalized: where does current entropy sit in this coin's own range?
        h_min = min(hist)
        h_max = max(hist)
        h_range = h_max - h_min

        if h_range < 0.05:
            # Very stable entropy — coin is consistently orderly or consistently chaotic
            # Use absolute scale as fallback
            return float(np.clip((2.78 - h) / 0.50 * 0.6, 0.0, 0.6))

        # 0 = at this coin's max entropy (noisiest), 1 = at its min (most orderly)
        normalized = (h_max - h) / h_range
        return float(np.clip(normalized * 0.6, 0.0, 0.6))


class LeadLagDetector:
    def __init__(self, window=30, max_lag=5, min_corr=0.20, move_threshold=0.002):
        self.window = window
        self.max_lag = max_lag
        self.min_corr = min_corr
        self.move_threshold = move_threshold
        self.returns = defaultdict(lambda: deque(maxlen=window + max_lag + 10))
        self.prices = defaultdict(lambda: deque(maxlen=max_lag + 5))
        self.relationships = []
        self._tick = 0

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
        return signals


class Strategy:
    def __init__(self, config=None):
        cfg = config or {}

        self.primary_assets = cfg.get("primary_assets", [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD",
            "SOL/USD", "DOGE/USD", "LTC/USD", "LINK/USD",
            "ONDO/USD", "AVAX/USD", "NEAR/USD", "DOT/USD",
        ])

        self.entropy = CoinEntropy(cfg.get("entropy_window", 40), cfg.get("entropy_bins", 10))
        self.lead_lag = LeadLagDetector(
            cfg.get("ll_window", 30), cfg.get("ll_max_lag", 5),
            cfg.get("ll_min_corr", 0.20), cfg.get("ll_move_threshold", 0.002))
        self.external = ExternalSignals(cfg.get("external", {}))

        # Faster EMAs for 10-day competition (30 min warmup instead of 60)
        self.fast_period = cfg.get("ema_fast", 12)
        self.slow_period = cfg.get("ema_slow", 30)
        self.long_period = cfg.get("ema_long", 100)

        # Aggressive but controlled
        self.max_per_asset = cfg.get("max_per_asset", 0.40)
        self.max_total = cfg.get("max_total_exposure", 0.90)
        self.min_score = cfg.get("min_score", 8)
        self.rebalance_threshold = cfg.get("rebalance_threshold", 0.015)
        self.max_coins = cfg.get("max_coins", 4)

        # BTC floor: always hold some BTC unless downtrending
        self.btc_floor = cfg.get("btc_floor", 0.15)

        # Profit-taking: sell 40% when position is up 3%+
        self.profit_take_pct = cfg.get("profit_take_pct", 0.03)
        self.profit_take_sell = cfg.get("profit_take_sell", 0.40)

        self.max_drawdown = cfg.get("max_drawdown", 0.05)
        self.recovery_threshold = cfg.get("recovery_threshold", 0.03)
        self.peak_value = 0.0
        self.circuit_breaker_active = False
        self._assets = {}

    def update(self, tickers):
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

    def _direction(self, pair):
        a = self._assets.get(pair)
        if not a or not a["ema_fast"] or not a["ema_slow"] or a["ema_slow"] == 0:
            return "flat"
        diff = (a["ema_fast"] - a["ema_slow"]) / a["ema_slow"]
        if diff > 0.0005:
            return "up"
        elif diff < -0.0005:
            return "down"
        return "flat"

    def _trend_strength(self, pair):
        a = self._assets.get(pair)
        if not a or not a["ema_fast"] or not a["ema_slow"] or a["ema_slow"] == 0:
            return 0.0
        diff = (a["ema_fast"] - a["ema_slow"]) / a["ema_slow"]
        if diff <= 0:
            return 0.0
        return float(np.clip(diff / 0.012, 0, 1))

    def _is_macro_bullish(self, pair):
        a = self._assets.get(pair)
        if not a or not a["ema_long"] or not a["prices"]:
            return False
        return a["prices"][-1] > a["ema_long"]

    def score_coin(self, pair):
        a = self._assets.get(pair)
        if not a or a["_count"] < self.slow_period:
            return {"total": 0, "direction": "?", "trend": 0, "entropy": 0,
                    "lead_lag": 0, "macro": 0, "ent_raw": 0}

        direction = self._direction(pair)
        strength = self._trend_strength(pair)
        ent_bonus = self.entropy.bonus(pair)
        ll_signals = self.lead_lag.get_bullish_signals()
        ll_strength = ll_signals.get(pair, 0)
        macro = self._is_macro_bullish(pair)

        # Path A: Trend (REQUIRES uptrend)
        trend_score = 0.0
        ent_score = 0.0
        macro_score = 0.0
        if direction == "up":
            trend_score = strength * 45
            ent_score = ent_bonus * trend_score  # up to 60% bonus
            macro_score = 15.0 if macro else 0.0

        # Path B: Lead-lag (independent, up to 45 points)
        ll_score = ll_strength * 45.0 if ll_strength > 0 else 0.0

        trend_path = trend_score + ent_score + macro_score
        ll_path = ll_score
        total = max(trend_path, ll_path)
        if trend_path >= 8 and ll_path >= 8:
            total += 10  # confluence bonus

        ent_raw = self.entropy.get(pair)
        return {
            "total": round(float(total), 1),
            "direction": direction,
            "trend": round(float(trend_score), 1),
            "entropy": round(float(ent_score), 1),
            "lead_lag": round(float(ll_score), 1),
            "macro": round(float(macro_score), 1),
            "ent_raw": round(float(ent_raw), 3) if ent_raw is not None else 0,
        }

    def get_target_allocations(self, pv, open_positions=None):
        """
        Returns target allocations. Also handles:
        - BTC floor (always 15% unless downtrending)
        - Profit-taking (reduce positions up 3%+)
        """
        if self.circuit_breaker_active:
            return {}

        ext_scalar = self.external.get_risk_scalar()

        # Score all coins
        scored = {}
        all_to_check = set(self.primary_assets)
        for pair in self.lead_lag.get_bullish_signals():
            if pair in self._assets and self._assets[pair]["_count"] >= self.slow_period:
                all_to_check.add(pair)

        for pair in all_to_check:
            s = self.score_coin(pair)
            if s["total"] >= self.min_score:
                scored[pair] = s

        # Rank and allocate top N
        allocations = {}
        if scored:
            ranked = sorted(scored.items(), key=lambda x: x[1]["total"], reverse=True)[:self.max_coins]
            total_score = sum(s["total"] for _, s in ranked)
            if total_score > 0:
                for pair, s in ranked:
                    alloc = (s["total"] / total_score) * self.max_total * ext_scalar
                    alloc = min(alloc, self.max_per_asset)
                    if alloc >= self.rebalance_threshold:
                        allocations[pair] = round(float(alloc), 4)

        # BTC floor: always hold 15% BTC unless it's in a downtrend
        btc_dir = self._direction("BTC/USD")
        if btc_dir != "down":
            current_btc = allocations.get("BTC/USD", 0)
            if current_btc < self.btc_floor:
                allocations["BTC/USD"] = self.btc_floor

        # Profit-taking: if any open position is up 3%+, reduce target by 40%
        if open_positions:
            for pair, pos in open_positions.items():
                if pos.get("avg_price", 0) > 0 and pair in self._assets:
                    current_price = self._assets[pair]["prices"][-1] if self._assets[pair]["prices"] else 0
                    if current_price > 0:
                        gain = (current_price / pos["avg_price"]) - 1
                        if gain >= self.profit_take_pct and pair in allocations:
                            old = allocations[pair]
                            allocations[pair] = round(old * (1 - self.profit_take_sell), 4)
                            logger.info(f"  PROFIT TAKE {pair}: up {gain:.1%}, reducing {old:.1%}→{allocations[pair]:.1%}")

        # Cap total
        total = sum(allocations.values())
        if total > self.max_total:
            scale = self.max_total / total
            allocations = {p: round(a * scale, 4) for p, a in allocations.items()}

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
        for pair in self.lead_lag.get_bullish_signals():
            if pair not in scores and pair in self._assets:
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
            "lead_lag_pairs": len(self.lead_lag.relationships),
            "lead_lag_signals": self.lead_lag.get_bullish_signals(),
            "external": self.external.get_state(),
            "circuit_breaker": self.circuit_breaker_active,
            "assets_ready": sum(1 for a in self._assets.values() if a["_count"] >= self.slow_period),
        }

    def is_ready(self):
        btc = self._assets.get("BTC/USD")
        return btc is not None and btc["_count"] >= self.slow_period and btc["ema_slow"] is not None
