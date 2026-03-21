"""
Trading Strategy — Structural Edge
====================================
Multi-asset, entropy-gated trend following with cross-asset lead-lag exploitation.

Design philosophy:
    - Crypto is volatile. Most teams will bleed from whipsaw trades and fees.
    - We use Shannon entropy to detect when the market is random (high entropy)
      and REFUSE to trend-follow during those periods. While other teams churn
      and lose money, we sit in cash losing nothing.
    - When entropy drops (market becomes orderly), we enter with full conviction.
    - Lead-lag trades bypass the entropy gate because they exploit structural
      delays between coins, not trend persistence.

Scoring target: 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar
    - 70% of score punishes downside → not trading in chaos is rewarded
    - Must also make top 20 returns → lead-lag keeps us active even during entropy gate

Architecture:
    Layer 1: Shannon entropy — should we trade at all?
    Layer 2: Cross-asset lead-lag — is there a structural delay opportunity?
    Layer 3: Trend signal — which direction is each coin heading?
    Layer 4: Binance external data — is the crowd about to get liquidated?
    Layer 5: Volatility targeting — how big should the position be?
    Layer 6: Drawdown circuit breaker — are we losing too much?

Entropy calibration (from real Roostoo BTC/USD data, March 2026):
    Real observed range: 2.23 — 2.90
    Median: 2.64
    Threshold: 2.75 (trades 70% of the time, blocks worst 30%)
    Confidence mapping: [2.3, 2.8] → [1.0, 0.0]
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Optional, List, Tuple
import logging

from external_signals import ExternalSignals

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# SHANNON ENTROPY REGIME FILTER
# ══════════════════════════════════════════════════════════════════════

class EntropyFilter:
    """
    Measures randomness of recent price returns using Shannon entropy.

    How it works:
        - Take last N returns, bin them into a histogram
        - Compute Shannon entropy: H = -Σ p(x) × log₂(p(x))
        - Low H = returns concentrated in one direction = TRENDING
        - High H = returns scattered everywhere = RANDOM NOISE

    Calibrated on real Roostoo BTC/USD 1-minute data (March 2026):
        Observed entropy range: 2.23 — 2.90
        Threshold 2.75 → trading 70% of the time
    """

    def __init__(self, window: int = 60, n_bins: int = 10):
        self.window = window
        self.n_bins = n_bins
        self.returns = deque(maxlen=window + 10)
        self._entropy_history = deque(maxlen=30)

    def update(self, ret: float):
        """Feed a new return observation."""
        self.returns.append(ret)
        if len(self.returns) >= self.window:
            h = self._compute()
            self._entropy_history.append(h)

    def _compute(self) -> float:
        """Shannon entropy of the return distribution."""
        rets = np.array(list(self.returns)[-self.window:])
        counts, _ = np.histogram(rets, bins=self.n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    @property
    def value(self) -> Optional[float]:
        """Current entropy value, or None if not ready."""
        return self._entropy_history[-1] if self._entropy_history else None

    @property
    def is_trending(self) -> bool:
        """Is the market orderly enough for trend-following?
        Calibrated: real Roostoo entropy ranges 2.23-2.90.
        Threshold 2.75 allows trading 70% of the time."""
        h = self.value
        if h is None:
            return False
        return h < 2.75

    @property
    def trend_confidence(self) -> float:
        """How trending is the market? 0 = random, 1 = very orderly.
        Calibrated: maps real observed range [2.3, 2.8] → [1.0, 0.0]."""
        h = self.value
        if h is None:
            return 0.0
        return float(np.clip((2.8 - h) / 0.5, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════
# CROSS-ASSET LEAD-LAG DETECTOR
# ══════════════════════════════════════════════════════════════════════

class LeadLagDetector:
    """
    Detects which coins lead and which follow with a time delay.

    Core insight: BTC often moves first, altcoins follow minutes later.
    When BTC surges 1% and ETH is still flat, we buy ETH expecting catch-up.

    This is NOT trend following — it's exploiting information propagation delay.
    These trades work in ANY market regime, so they bypass the entropy gate.
    """

    def __init__(self, window: int = 30, max_lag: int = 5, min_corr: float = 0.25,
                 move_threshold: float = 0.003):
        self.window = window
        self.max_lag = max_lag
        self.min_corr = min_corr
        self.move_threshold = move_threshold

        self.returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window + max_lag + 10))
        self.relationships: List[dict] = []
        self._tick_count = 0

    def update(self, pair: str, price: float, prev_price: float):
        """Feed a new price. Provide previous price for return calculation."""
        if prev_price > 0:
            ret = (price - prev_price) / prev_price
            self.returns[pair].append(ret)

        self._tick_count += 1
        if self._tick_count >= 30:
            self._recalculate()
            self._tick_count = 0

    def _recalculate(self):
        """Find all significant leader-follower relationships."""
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
                    results.append({"leader": leader, "follower": follower,
                                    "lag": lag, "corr": corr})

        results.sort(key=lambda x: abs(x["corr"]), reverse=True)
        self.relationships = results[:15]

        if results:
            top = results[0]
            logger.info(f"  Lead-lag: {top['leader']}→{top['follower']} lag={top['lag']} corr={top['corr']:.3f}")

    def _best_lag(self, leader: str, follower: str) -> Tuple[Optional[int], float]:
        """Find the lag that maximizes leader→follower correlation."""
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
            c = np.corrcoef(x[:length], y[:length])[0, 1]
            if not np.isnan(c) and abs(c) > abs(best_corr):
                best_corr, best_lag = c, lag

        return best_lag, best_corr

    def get_signals(self) -> Dict[str, float]:
        """
        Generate lead-lag trading signals.
        Returns {follower_pair: signal} where signal in [-1, +1].
        Positive = leader moved up, follower should follow.
        """
        signals = {}
        for rel in self.relationships:
            leader_rets = list(self.returns.get(rel["leader"], []))
            lag = rel["lag"]
            if len(leader_rets) < lag + 1:
                continue

            leader_move = np.mean(leader_rets[-lag:])
            if abs(leader_move) < self.move_threshold:
                continue

            signal = np.sign(leader_move) * rel["corr"]
            signal *= min(1.0, abs(leader_move) / self.move_threshold)
            follower = rel["follower"]

            if follower not in signals or abs(signal) > abs(signals[follower]):
                signals[follower] = float(np.clip(signal, -1, 1))

        return signals


# ══════════════════════════════════════════════════════════════════════
# MAIN STRATEGY
# ══════════════════════════════════════════════════════════════════════

class Strategy:
    """
    The complete trading strategy combining all layers.

    Decision flow for each asset every tick:
        1. Circuit breaker check → if triggered, sell everything
        2. Is there a lead-lag signal? → trade regardless of entropy
        3. Is entropy low (market orderly)? → allow trend-following trades
        4. Trend check: EMA fast > EMA slow + price > EMA long?
        5. Size position with vol targeting × external signal scalar
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        self.primary_assets = cfg.get("primary_assets", [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD",
            "SOL/USD", "DOGE/USD", "LTC/USD", "LINK/USD",
        ])

        # Modules
        self.entropy = EntropyFilter(
            window=cfg.get("entropy_window", 60),
            n_bins=cfg.get("entropy_bins", 10),
        )
        self.lead_lag = LeadLagDetector(
            window=cfg.get("ll_window", 30),
            max_lag=cfg.get("ll_max_lag", 5),
            min_corr=cfg.get("ll_min_corr", 0.25),
            move_threshold=cfg.get("ll_move_threshold", 0.003),
        )
        self.external = ExternalSignals(cfg.get("external", {}))

        # EMA periods
        self.fast_period = cfg.get("ema_fast", 20)
        self.slow_period = cfg.get("ema_slow", 60)
        self.long_period = cfg.get("ema_long", 200)

        # Position sizing
        self.target_vol = cfg.get("target_annual_vol", 0.15)
        self.vol_lookback = cfg.get("vol_lookback", 48)
        self.max_per_asset = cfg.get("max_per_asset", 0.30)
        self.max_total = cfg.get("max_total_exposure", 0.85)
        self.rebalance_threshold = cfg.get("rebalance_threshold", 0.03)

        # Drawdown protection
        self.max_drawdown = cfg.get("max_drawdown", 0.05)
        self.recovery_threshold = cfg.get("recovery_threshold", 0.03)
        self.peak_value = 0.0
        self.circuit_breaker_active = False

        # Per-asset state
        self._assets: Dict[str, dict] = {}

    # ── Price updates ──────────────────────────────────────────────

    def update(self, tickers: Dict[str, dict]):
        """Feed all current tickers. Call every polling cycle."""
        for pair, data in tickers.items():
            price = data.get("LastPrice", 0)
            if price <= 0:
                continue
            self._update_asset(pair, price)

        # Feed BTC returns to entropy filter
        btc = self._assets.get("BTC/USD")
        if btc and btc["returns"]:
            self.entropy.update(btc["returns"][-1])

    def _update_asset(self, pair: str, price: float):
        """Update EMAs and returns for a single asset."""
        if pair not in self._assets:
            self._assets[pair] = {
                "prices": deque(maxlen=250),
                "returns": deque(maxlen=100),
                "ema_fast": None, "ema_slow": None, "ema_long": None,
                "_count": 0,
            }

        a = self._assets[pair]
        prev = a["prices"][-1] if a["prices"] else None
        a["prices"].append(price)
        a["_count"] += 1

        if prev and prev > 0:
            a["returns"].append((price - prev) / prev)
            self.lead_lag.update(pair, price, prev)

        for name, period in [("ema_fast", self.fast_period),
                             ("ema_slow", self.slow_period),
                             ("ema_long", self.long_period)]:
            if a[name] is None:
                if a["_count"] >= period:
                    a[name] = np.mean(list(a["prices"])[-period:])
            else:
                alpha = 2.0 / (period + 1)
                a[name] = alpha * price + (1 - alpha) * a[name]

    # ── Drawdown protection ────────────────────────────────────────

    def update_drawdown(self, portfolio_value: float):
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        if self.peak_value <= 0:
            return

        dd = (self.peak_value - portfolio_value) / self.peak_value
        if self.circuit_breaker_active:
            if dd < self.recovery_threshold:
                logger.info(f"Circuit breaker OFF (DD recovered to {dd:.2%})")
                self.circuit_breaker_active = False
        elif dd >= self.max_drawdown:
            logger.warning(f"CIRCUIT BREAKER ON: drawdown = {dd:.2%}")
            self.circuit_breaker_active = True

    # ── Signal scoring ─────────────────────────────────────────────

    def _is_bullish(self, pair: str) -> bool:
        a = self._assets.get(pair)
        if not a or not a["ema_fast"] or not a["ema_slow"]:
            return False
        return a["ema_fast"] > a["ema_slow"]

    def _is_macro_bullish(self, pair: str) -> bool:
        a = self._assets.get(pair)
        if not a or not a["ema_long"] or not a["prices"]:
            return False
        return a["prices"][-1] > a["ema_long"]

    def _trend_strength(self, pair: str) -> float:
        a = self._assets.get(pair)
        if not a or not a["ema_fast"] or not a["ema_slow"] or a["ema_slow"] == 0:
            return 0.0
        diff = abs(a["ema_fast"] - a["ema_slow"]) / a["ema_slow"]
        return float(np.clip(diff / 0.02, 0, 1))

    def _realized_vol(self, pair: str) -> Optional[float]:
        a = self._assets.get(pair)
        if not a or len(a["returns"]) < 20:
            return None
        rets = list(a["returns"])[-self.vol_lookback:]
        return float(np.std(rets) * np.sqrt(525600))

    # ── Main allocation logic ──────────────────────────────────────

    def get_target_allocations(self, portfolio_value: float) -> Dict[str, float]:
        """
        Decide how much of the portfolio to allocate to each asset.
        Returns {pair: fraction} where fraction in [0, max_per_asset].

        Two paths:
            Path A: Lead-lag signal → trade regardless of entropy (up to 20%)
            Path B: Trend following → only when entropy says market is orderly (up to 25%)
        """
        if self.circuit_breaker_active:
            return {}

        allocations = {}
        external_scalar = self.external.get_risk_scalar()
        entropy_conf = self.entropy.trend_confidence
        ll_signals = self.lead_lag.get_signals()

        for pair in self.primary_assets:
            a = self._assets.get(pair)
            if not a or a["_count"] < self.slow_period:
                continue

            alloc = 0.0

            # ── Path A: Lead-lag (bypasses entropy gate) ──
            if pair in ll_signals and ll_signals[pair] > 0.2:
                ll_strength = ll_signals[pair]
                alloc = 0.20 * ll_strength  # up to 20%

            # ── Path B: Trend following (requires low entropy) ──
            if self.entropy.is_trending and self._is_bullish(pair):
                trend_alloc = 0.25 * self._trend_strength(pair)  # up to 25%
                if self._is_macro_bullish(pair):
                    trend_alloc *= 1.5  # macro bonus

                # Scale by entropy confidence with floor at 0.4
                trend_alloc *= max(entropy_conf, 0.4)
                alloc = max(alloc, trend_alloc)

            if alloc < 0.03:  # below rebalance threshold, skip
                continue

            # ── Vol targeting with floor ──
            vol = self._realized_vol(pair)
            if vol and vol > 0:
                vol_scalar = self.target_vol / vol
                alloc *= np.clip(vol_scalar, 0.3, 2.0)  # floor 0.3 so vol doesn't kill allocation

            # ── External signals ──
            alloc *= external_scalar

            # ── Cap per asset ──
            alloc = min(alloc, self.max_per_asset)
            if alloc >= 0.03:
                allocations[pair] = round(alloc, 4)

        # ── Cap total exposure ──
        total = sum(allocations.values())
        if total > self.max_total:
            scale = self.max_total / total
            allocations = {p: round(a * scale, 4) for p, a in allocations.items()}

        return allocations

    # ── Limit order pricing ────────────────────────────────────────

    def get_limit_price(self, pair: str, side: str, ticker: dict) -> Optional[float]:
        """Place limit order inside the spread for maker fee (0.05%)."""
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
        # Build conviction scores for logging
        scores = {}
        for pair in self.primary_assets:
            a = self._assets.get(pair)
            if not a or a["_count"] < self.slow_period:
                continue
            score = 0.0
            if self._is_bullish(pair):
                score = self._trend_strength(pair)
            scores[pair] = round(score, 3)

        return {
            "entropy": round(self.entropy.value, 3) if self.entropy.value else None,
            "entropy_trending": self.entropy.is_trending,
            "entropy_confidence": round(self.entropy.trend_confidence, 3),
            "lead_lag_pairs": len(self.lead_lag.relationships),
            "lead_lag_signals": self.lead_lag.get_signals(),
            "external": self.external.get_state(),
            "circuit_breaker": self.circuit_breaker_active,
            "assets_ready": sum(1 for a in self._assets.values() if a["_count"] >= self.slow_period),
            "conviction_scores": scores,
        }

    def is_ready(self) -> bool:
        btc = self._assets.get("BTC/USD")
        return btc is not None and btc["_count"] >= self.slow_period and btc["ema_slow"] is not None
