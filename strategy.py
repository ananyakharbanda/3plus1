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
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Optional, List, Tuple
import logging

from external_signals import ExternalSignals

logger = logging.getLogger(__name__)


# SHANNON ENTROPY REGIME FILTER

class EntropyFilter:
    """
    Measures randomness of recent price returns using Shannon entropy.

    How it works:
        - Take last N returns, bin them into a histogram
        - Compute Shannon entropy: H = -Σ p(x) × log₂(p(x))
        - Low H = returns concentrated in one direction = TRENDING
        - High H = returns scattered everywhere = RANDOM NOISE

    Why it matters:
        In a random market, every trend-following trade is a coin flip minus fees.
        At 0.1% per trade, 10 whipsaw round-trips cost 2% of portfolio.
        Better to sit in cash and let other teams bleed.
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
        """Is the market orderly enough for trend-following?"""
        h = self.value
        if h is None:
            return False  # cautious during warmup
        return h < 2.5  # tunable threshold — calibrate on real data

    @property
    def trend_confidence(self) -> float:
        """How trending is the market? 0 = random, 1 = very orderly."""
        h = self.value
        if h is None:
            return 0.0
        # Map entropy [1.5, 3.2] → confidence [1.0, 0.0]
        return float(np.clip((3.2 - h) / 1.7, 0.0, 1.0))


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

    def __init__(self, window: int = 30, max_lag: int = 5, min_corr: float = 0.3,
                 move_threshold: float = 0.005):
        self.window = window
        self.max_lag = max_lag
        self.min_corr = min_corr
        self.move_threshold = move_threshold  # 0.5% = significant leader move

        self.returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window + max_lag + 10))
        self.relationships: List[dict] = []
        self._tick_count = 0

    def update(self, pair: str, price: float, prev_price: float):
        """Feed a new price. Provide previous price for return calculation."""
        if prev_price > 0:
            ret = (price - prev_price) / prev_price
            self.returns[pair].append(ret)

        # Recalculate relationships every 30 ticks
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

            # Average leader move over the lag period
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

        # Which coins to trade (high liquidity, good for lead-lag)
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
            min_corr=cfg.get("ll_min_corr", 0.3),
            move_threshold=cfg.get("ll_move_threshold", 0.005),
        )
        self.external = ExternalSignals(cfg.get("external", {}))

        # EMA periods for trend detection
        self.fast_period = cfg.get("ema_fast", 20)
        self.slow_period = cfg.get("ema_slow", 60)
        self.long_period = cfg.get("ema_long", 200)

        # Position sizing
        self.target_vol = cfg.get("target_annual_vol", 0.15)
        self.vol_lookback = cfg.get("vol_lookback", 48)
        self.max_per_asset = cfg.get("max_per_asset", 0.30)       # 30% max in one coin
        self.max_total = cfg.get("max_total_exposure", 0.85)      # 85% max total invested
        self.rebalance_threshold = cfg.get("rebalance_threshold", 0.03)  # 3% min change

        # Drawdown protection
        self.max_drawdown = cfg.get("max_drawdown", 0.05)
        self.recovery_threshold = cfg.get("recovery_threshold", 0.03)
        self.peak_value = 0.0
        self.circuit_breaker_active = False

        # Per-asset state: {pair: {prices, ema_fast, ema_slow, ema_long, returns}}
        self._assets: Dict[str, dict] = {}

    # ── Price updates ──────────────────────────────────────────────

    def update(self, tickers: Dict[str, dict]):
        """
        Feed all current tickers. Call every polling cycle.
        tickers: {pair: {LastPrice, MaxBid, MinAsk, ...}}
        """
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

        # Update EMAs incrementally
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
        """Track peak value, activate circuit breaker if drawdown exceeds limit."""
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
        """Is the trend bullish? Fast EMA > Slow EMA."""
        a = self._assets.get(pair)
        if not a or not a["ema_fast"] or not a["ema_slow"]:
            return False
        return a["ema_fast"] > a["ema_slow"]

    def _is_macro_bullish(self, pair: str) -> bool:
        """Is price above the long-term EMA?"""
        a = self._assets.get(pair)
        if not a or not a["ema_long"] or not a["prices"]:
            return False
        return a["prices"][-1] > a["ema_long"]

    def _trend_strength(self, pair: str) -> float:
        """How strong is the trend? [0, 1]"""
        a = self._assets.get(pair)
        if not a or not a["ema_fast"] or not a["ema_slow"] or a["ema_slow"] == 0:
            return 0.0
        diff = abs(a["ema_fast"] - a["ema_slow"]) / a["ema_slow"]
        return float(np.clip(diff / 0.02, 0, 1))

    def _realized_vol(self, pair: str) -> Optional[float]:
        """Annualized realized volatility for an asset."""
        a = self._assets.get(pair)
        if not a or len(a["returns"]) < 20:
            return None
        rets = list(a["returns"])[-self.vol_lookback:]
        return float(np.std(rets) * np.sqrt(525600))  # annualize from 1-min

    # ── Main allocation logic ──────────────────────────────────────

    def get_target_allocations(self, portfolio_value: float) -> Dict[str, float]:
        """
        Decide how much of the portfolio to allocate to each asset.
        Returns {pair: fraction} where fraction in [0, max_per_asset].

        Logic:
            1. Circuit breaker → return empty (sell all)
            2. Lead-lag signals → allocate regardless of entropy
            3. Trend signals → only if entropy says market is orderly
            4. Scale by volatility targeting and external risk scalar
        """
        if self.circuit_breaker_active:
            return {}  # sell everything

        allocations = {}
        external_scalar = self.external.get_risk_scalar()
        entropy_conf = self.entropy.trend_confidence
        ll_signals = self.lead_lag.get_signals()

        for pair in self.primary_assets:
            a = self._assets.get(pair)
            if not a or a["_count"] < self.slow_period:
                continue  # not enough data yet

            alloc = 0.0

            # ── Path A: Lead-lag opportunity (bypasses entropy gate) ──
            if pair in ll_signals and ll_signals[pair] > 0.3:
                ll_strength = ll_signals[pair]
                alloc = 0.15 * ll_strength  # up to 15% allocation from lead-lag

            # ── Path B: Trend following (requires low entropy) ──
            if self.entropy.is_trending and self._is_bullish(pair):
                trend_alloc = 0.20 * self._trend_strength(pair)  # up to 20%
                if self._is_macro_bullish(pair):
                    trend_alloc *= 1.3  # macro alignment bonus

                trend_alloc *= entropy_conf  # scale by how orderly the market is
                alloc = max(alloc, trend_alloc)  # take the stronger signal

            if alloc < 0.02:  # skip tiny allocations
                continue

            # ── Scale by volatility targeting ──
            vol = self._realized_vol(pair)
            if vol and vol > 0:
                vol_scalar = self.target_vol / vol
                alloc *= min(vol_scalar, 2.0)  # cap at 2x

            # ── Scale by external signals ──
            alloc *= external_scalar

            # ── Cap per asset ──
            alloc = min(alloc, self.max_per_asset)
            allocations[pair] = round(alloc, 4)

        # ── Cap total exposure ──
        total = sum(allocations.values())
        if total > self.max_total:
            scale = self.max_total / total
            allocations = {p: round(a * scale, 4) for p, a in allocations.items()}

        return allocations

    # ── Limit order pricing ────────────────────────────────────────

    def get_limit_price(self, pair: str, side: str, ticker: dict) -> Optional[float]:
        """
        Calculate limit order price to qualify as maker (0.05% fee).
        Places order slightly inside the spread for high fill probability.
        """
        bid = ticker.get("MaxBid", 0)
        ask = ticker.get("MinAsk", 0)
        if bid <= 0 or ask <= 0:
            return None

        spread = ask - bid
        if side == "BUY":
            return round(bid + spread * 0.3, 2)  # 30% into spread from bid
        else:
            return round(ask - spread * 0.3, 2)  # 30% into spread from ask

    # ── Diagnostics ────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "entropy": round(self.entropy.value, 3) if self.entropy.value else None,
            "entropy_trending": self.entropy.is_trending,
            "entropy_confidence": round(self.entropy.trend_confidence, 3),
            "lead_lag_pairs": len(self.lead_lag.relationships),
            "lead_lag_signals": self.lead_lag.get_signals(),
            "external": self.external.get_state(),
            "circuit_breaker": self.circuit_breaker_active,
            "assets_ready": sum(1 for a in self._assets.values() if a["_count"] >= self.slow_period),
        }

    def is_ready(self) -> bool:
        """Strategy needs at least BTC with enough data."""
        btc = self._assets.get("BTC/USD")
        return btc is not None and btc["_count"] >= self.slow_period and btc["ema_slow"] is not None

