"""
Trading Strategy v4 — Aggressive Structural Edge
==================================================
Per-coin entropy scoring. Aggressive deployment. Risk-managed.

Key changes from v3:
    - Each coin gets its OWN entropy score (not just BTC globally)
    - Coins are RANKED by opportunity quality every tick
    - Allocate to the TOP coins, not all coins equally
    - No vol targeting on 1-min data (was crushing allocations to zero)
    - Entropy is a SIZER not a GATE (always deploy some capital)
    - Lead-lag fixed: uses cumulative returns over lag window
    - Higher base allocations: 40% per coin, 85% total max

Entropy calibration (real Roostoo data, March 2026):
    Observed range: 2.23 — 2.90
    Best coins to trade: those with entropy < 2.6 (strong trend)
    Moderate: 2.6 — 2.75
    Avoid trend-following: > 2.75 (but still do lead-lag)
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Optional, List, Tuple
import logging

from external_signals import ExternalSignals

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# PER-COIN ENTROPY
# ══════════════════════════════════════════════════════════════════

class CoinEntropy:
    """Shannon entropy computed per-coin, not globally."""

    def __init__(self, window: int = 60, n_bins: int = 10):
        self.window = window
        self.n_bins = n_bins
        self._returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window + 10))
        self._cache: Dict[str, float] = {}

    def update(self, pair: str, ret: float):
        self._returns[pair].append(ret)
        if len(self._returns[pair]) >= self.window:
            self._cache[pair] = self._compute(pair)

    def _compute(self, pair: str) -> float:
        rets = np.array(list(self._returns[pair])[-self.window:])
        counts, _ = np.histogram(rets, bins=self.n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def get(self, pair: str) -> Optional[float]:
        return self._cache.get(pair)

    def score(self, pair: str) -> float:
        """
        Opportunity score from entropy: [0, 1].
        1.0 = very orderly (entropy ~2.2) → strong trend, trade big
        0.5 = moderate (entropy ~2.55) → some trend, trade medium
        0.0 = chaotic (entropy ~2.9+) → no trend signal, trade small or skip

        Calibrated on real Roostoo data: range [2.2, 2.9]
        """
        h = self.get(pair)
        if h is None:
            return 0.3  # moderate default during warmup
        return float(np.clip((2.85 - h) / 0.65, 0.0, 1.0))

    def is_trending(self, pair: str) -> bool:
        """Is this specific coin orderly enough for trend-following?"""
        h = self.get(pair)
        if h is None:
            return False
        return h < 2.75


# ══════════════════════════════════════════════════════════════════
# LEAD-LAG DETECTOR (fixed execution)
# ══════════════════════════════════════════════════════════════════

class LeadLagDetector:
    """
    BTC often moves first, altcoins follow 1-5 minutes later.
    Fixed: uses cumulative return over lag window, not single-tick.
    """

    def __init__(self, window: int = 30, max_lag: int = 5,
                 min_corr: float = 0.25, move_threshold: float = 0.002):
        self.window = window
        self.max_lag = max_lag
        self.min_corr = min_corr
        self.move_threshold = move_threshold  # 0.2% cumulative over lag window

        self.returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window + max_lag + 10))
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_lag + 5))
        self.relationships: List[dict] = []
        self._tick = 0

    def update(self, pair: str, price: float, ret: float):
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

    def get_signals(self) -> Dict[str, float]:
        """
        Fixed: uses CUMULATIVE return over the lag window, not single tick.
        This is why signals were always 0 before — single 1-min returns rarely hit 0.3%.
        """
        signals = {}
        for rel in self.relationships:
            leader = rel["leader"]
            lag = rel["lag"]
            prices = list(self.prices.get(leader, []))

            if len(prices) < lag + 1:
                continue

            # Cumulative return of leader over the lag window
            leader_move = (prices[-1] / prices[-lag - 1]) - 1.0 if prices[-lag - 1] > 0 else 0

            if abs(leader_move) < self.move_threshold:
                continue

            signal = np.sign(leader_move) * abs(rel["corr"])
            signal *= min(1.5, abs(leader_move) / self.move_threshold)  # stronger move = stronger signal
            follower = rel["follower"]

            if follower not in signals or abs(signal) > abs(signals[follower]):
                signals[follower] = float(np.clip(signal, -1.0, 1.0))

        return signals


# ══════════════════════════════════════════════════════════════════
# MAIN STRATEGY
# ══════════════════════════════════════════════════════════════════

class Strategy:
    """
    Per-coin scoring → rank → allocate to best opportunities.

    Each coin gets a score from 0-100 based on:
        - Trend strength (EMA alignment)           [0-35 points]
        - Per-coin entropy (how orderly is THIS coin) [0-25 points]
        - Lead-lag signal (is a leader pulling it)    [0-25 points]
        - Macro alignment (price > long EMA)          [0-15 points]

    Top coins get allocated proportionally to score.
    External signals scale everything up or down.
    Circuit breaker overrides all.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        self.primary_assets = cfg.get("primary_assets", [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD",
            "SOL/USD", "DOGE/USD", "LTC/USD", "LINK/USD",
        ])

        # Modules
        self.entropy = CoinEntropy(
            window=cfg.get("entropy_window", 60),
            n_bins=cfg.get("entropy_bins", 10),
        )
        self.lead_lag = LeadLagDetector(
            window=cfg.get("ll_window", 30),
            max_lag=cfg.get("ll_max_lag", 5),
            min_corr=cfg.get("ll_min_corr", 0.25),
            move_threshold=cfg.get("ll_move_threshold", 0.002),
        )
        self.external = ExternalSignals(cfg.get("external", {}))

        # EMA periods
        self.fast_period = cfg.get("ema_fast", 20)
        self.slow_period = cfg.get("ema_slow", 60)
        self.long_period = cfg.get("ema_long", 200)

        # Allocation parameters — AGGRESSIVE
        self.max_per_asset = cfg.get("max_per_asset", 0.35)       # up to 35% per coin
        self.max_total = cfg.get("max_total_exposure", 0.85)      # 85% max invested
        self.min_score_to_trade = cfg.get("min_score", 15)        # need 15/100 points
        self.rebalance_threshold = cfg.get("rebalance_threshold", 0.02)  # 2% min change
        self.max_coins = cfg.get("max_coins", 4)                  # hold max 4 coins at once

        # Drawdown
        self.max_drawdown = cfg.get("max_drawdown", 0.05)
        self.recovery_threshold = cfg.get("recovery_threshold", 0.03)
        self.peak_value = 0.0
        self.circuit_breaker_active = False

        # Per-asset state
        self._assets: Dict[str, dict] = {}

    # ── Price updates ──────────────────────────────────────────────

    def update(self, tickers: Dict[str, dict]):
        for pair, data in tickers.items():
            price = data.get("LastPrice", 0)
            if price <= 0:
                continue
            self._update_asset(pair, price)

    def _update_asset(self, pair: str, price: float):
        if pair not in self._assets:
            self._assets[pair] = {
                "prices": deque(maxlen=250), "returns": deque(maxlen=100),
                "ema_fast": None, "ema_slow": None, "ema_long": None, "_count": 0,
            }

        a = self._assets[pair]
        prev = a["prices"][-1] if a["prices"] else None
        a["prices"].append(price)
        a["_count"] += 1

        if prev and prev > 0:
            ret = (price - prev) / prev
            a["returns"].append(ret)
            self.entropy.update(pair, ret)
            self.lead_lag.update(pair, price, ret)

        for name, period in [("ema_fast", self.fast_period),
                             ("ema_slow", self.slow_period),
                             ("ema_long", self.long_period)]:
            if a[name] is None:
                if a["_count"] >= period:
                    a[name] = np.mean(list(a["prices"])[-period:])
            else:
                alpha = 2.0 / (period + 1)
                a[name] = alpha * price + (1 - alpha) * a[name]

    # ── Drawdown ───────────────────────────────────────────────────

    def update_drawdown(self, portfolio_value: float):
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        if self.peak_value <= 0:
            return
        dd = (self.peak_value - portfolio_value) / self.peak_value
        if self.circuit_breaker_active:
            if dd < self.recovery_threshold:
                logger.info(f"Circuit breaker OFF (DD={dd:.2%})")
                self.circuit_breaker_active = False
        elif dd >= self.max_drawdown:
            logger.warning(f"CIRCUIT BREAKER ON: DD={dd:.2%}")
            self.circuit_breaker_active = True

    # ── Per-coin scoring ───────────────────────────────────────────

    def score_coin(self, pair: str) -> dict:
        """
        Score a coin 0-100. Higher = better opportunity.

        Breakdown:
            Trend (0-35): Is fast EMA above slow? How far apart?
            Entropy (0-25): Is THIS coin orderly? (per-coin, not global)
            Lead-lag (0-25): Is a leader coin pulling this one up?
            Macro (0-15): Is price above long-term EMA?
        """
        a = self._assets.get(pair)
        if not a or a["_count"] < self.slow_period:
            return {"total": 0, "trend": 0, "entropy": 0, "lead_lag": 0, "macro": 0}

        scores = {"trend": 0, "entropy": 0, "lead_lag": 0, "macro": 0}

        # ── Trend: 0-35 points ──
        if a["ema_fast"] and a["ema_slow"] and a["ema_slow"] > 0:
            if a["ema_fast"] > a["ema_slow"]:
                strength = abs(a["ema_fast"] - a["ema_slow"]) / a["ema_slow"]
                scores["trend"] = min(35, strength / 0.02 * 35)
            # Penalty for bearish trend — don't buy into downtrends
            elif a["ema_fast"] < a["ema_slow"]:
                scores["trend"] = -10

        # ── Entropy: 0-25 points (per-coin!) ──
        ent_score = self.entropy.score(pair)  # 0-1
        scores["entropy"] = ent_score * 25

        # ── Lead-lag: 0-25 points ──
        ll_signals = self.lead_lag.get_signals()
        if pair in ll_signals and ll_signals[pair] > 0:
            scores["lead_lag"] = ll_signals[pair] * 25

        # ── Macro: 0-15 points ──
        if a["ema_long"] and a["prices"]:
            if a["prices"][-1] > a["ema_long"]:
                scores["macro"] = 15

        scores["total"] = max(0, sum(scores.values()))
        return scores

    # ── Allocation ─────────────────────────────────────────────────

    def get_target_allocations(self, portfolio_value: float) -> Dict[str, float]:
        """
        Score all coins → rank → allocate to top N.
        No vol targeting (was killing allocations on 1-min data).
        External signals scale everything.
        """
        if self.circuit_breaker_active:
            return {}

        external_scalar = self.external.get_risk_scalar()

        # Score every primary asset
        scored = {}
        for pair in self.primary_assets:
            s = self.score_coin(pair)
            if s["total"] >= self.min_score_to_trade:
                scored[pair] = s

        if not scored:
            return {}

        # Rank by total score, take top N
        ranked = sorted(scored.items(), key=lambda x: x[1]["total"], reverse=True)
        top = ranked[:self.max_coins]

        # Allocate proportionally to score
        total_score = sum(s["total"] for _, s in top)
        if total_score <= 0:
            return {}

        allocations = {}
        for pair, s in top:
            # Base allocation proportional to score share
            alloc = (s["total"] / total_score) * self.max_total

            # Scale by external risk
            alloc *= external_scalar

            # Cap per asset
            alloc = min(alloc, self.max_per_asset)

            if alloc >= self.rebalance_threshold:
                allocations[pair] = round(alloc, 4)

        # Cap total
        total = sum(allocations.values())
        if total > self.max_total:
            scale = self.max_total / total
            allocations = {p: round(a * scale, 4) for p, a in allocations.items()}

        return allocations

    # ── Limit order pricing ────────────────────────────────────────

    def get_limit_price(self, pair: str, side: str, ticker: dict) -> Optional[float]:
        bid = ticker.get("MaxBid", 0)
        ask = ticker.get("MinAsk", 0)
        if bid <= 0 or ask <= 0:
            return None
        spread = ask - bid
        if side == "BUY":
            return round(bid + spread * 0.3, 2)
        else:
            return round(ask - spread * 0.3, 2)

    # ── Diagnostics ────────────────────────────────────────────────

    def get_state(self) -> dict:
        scores = {}
        for pair in self.primary_assets:
            s = self.score_coin(pair)
            if s["total"] > 0:
                scores[pair] = s

        ll_signals = self.lead_lag.get_signals()

        # Per-coin entropy values
        entropies = {}
        for pair in self.primary_assets:
            h = self.entropy.get(pair)
            if h is not None:
                entropies[pair] = round(h, 3)

        return {
            "coin_scores": scores,
            "entropies": entropies,
            "lead_lag_pairs": len(self.lead_lag.relationships),
            "lead_lag_signals": ll_signals,
            "external": self.external.get_state(),
            "circuit_breaker": self.circuit_breaker_active,
            "assets_ready": sum(1 for a in self._assets.values() if a["_count"] >= self.slow_period),
        }

    def is_ready(self) -> bool:
        btc = self._assets.get("BTC/USD")
        return btc is not None and btc["_count"] >= self.slow_period and btc["ema_slow"] is not None
