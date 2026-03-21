"""
Telegram Notifications
=======================
Sends trade alerts and hourly portfolio updates to your Telegram.

Setup (one-time, takes 2 minutes):
    1. Open Telegram, search for @BotFather
    2. Send /newbot → pick a name → pick a username (must end in 'bot')
    3. BotFather gives you a token like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
    4. Search for @userinfobot in Telegram, send /start → it gives your chat ID
    5. Set both as environment variables:
         export TELEGRAM_BOT_TOKEN="your_token_here"
         export TELEGRAM_CHAT_ID="your_chat_id_here"
    6. IMPORTANT: Open a chat with your bot and send /start (bot can't message you first)

No extra pip installs needed — uses requests which is already installed.
"""

import os
import time
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends formatted messages to Telegram.
    Fails silently if not configured (won't crash the bot).
    """

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._last_hourly = 0
        self._hourly_interval = 3600  # 1 hour

        if self.enabled:
            logger.info("Telegram notifications: ON")
        else:
            logger.info("Telegram notifications: OFF (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")

    def _send(self, text: str):
        """Send a message via Telegram Bot API. Fails silently."""
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=5)
            if not resp.ok:
                logger.debug(f"Telegram send failed: {resp.text}")
        except Exception as e:
            logger.debug(f"Telegram error: {e}")

    # ── Trade alerts ───────────────────────────────────────────────

    def trade_alert(self, side: str, qty: float, pair: str, price: float,
                    usd_value: float, order_type: str, trade_pnl: float = None,
                    avg_buy: float = None):
        """Send alert when a trade executes."""
        coin = pair.split("/")[0]
        emoji = "🟢" if side == "BUY" else "🔴"

        msg = f"{emoji} <b>{side} {qty} {coin}</b>\n"
        msg += f"Price: ${price:,.2f}\n"
        msg += f"Value: ${usd_value:,.2f}\n"
        msg += f"Type: {order_type}\n"

        if side == "BUY" and avg_buy:
            msg += f"Avg buy: ${avg_buy:,.2f}\n"

        if side == "SELL" and trade_pnl is not None:
            pnl_emoji = "✅" if trade_pnl >= 0 else "❌"
            msg += f"{pnl_emoji} Trade P&L: ${trade_pnl:+,.2f}\n"

        self._send(msg)

    # ── Hourly portfolio update ────────────────────────────────────

    def hourly_update(self, portfolio_value: float, pnl: float, pnl_pct: float,
                      realized_pnl: float, drawdown: float, cash_pct: float,
                      positions: dict, targets: dict, force: bool = False):
        """
        Send portfolio summary. Only sends once per hour unless force=True.
        """
        now = time.time()
        if not force and (now - self._last_hourly) < self._hourly_interval:
            return
        self._last_hourly = now

        now_str = datetime.now().strftime("%H:%M")
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        msg = f"⏰ <b>Hourly Update</b> ({now_str})\n\n"
        msg += f"💰 Portfolio: <b>${portfolio_value:,.0f}</b>\n"
        msg += f"{pnl_emoji} PnL: ${pnl:+,.0f} ({pnl_pct:+.2f}%)\n"
        msg += f"💵 Realized: ${realized_pnl:+,.0f}\n"
        msg += f"📊 Drawdown: {drawdown:.2%}\n"
        msg += f"💵 Cash: {cash_pct:.0%}\n"

        # Positions
        if positions:
            msg += f"\n<b>Positions:</b>\n"
            for pair, info in positions.items():
                if pair == "USD":
                    continue
                coin = pair.split("/")[0]
                pnl_val = info.get("pnl", 0)
                pnl_s = f" ({'+' if pnl_val >= 0 else ''}{pnl_val:.0f})" if pnl_val != 0 else ""
                pct = info.get("pct", "?")
                msg += f"  • {coin}: {pct} ${info.get('value', 0):,.0f}{pnl_s}\n"

        # Targets
        if targets:
            msg += f"\n<b>Targets:</b>\n"
            for pair, alloc in targets.items():
                coin = pair.split("/")[0]
                msg += f"  • {coin}: {alloc:.0%}\n"

        self._send(msg)

    # ── Special alerts ─────────────────────────────────────────────

    def circuit_breaker_alert(self, drawdown: float, portfolio_value: float):
        """Alert when circuit breaker fires."""
        self._send(
            f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b>\n\n"
            f"Drawdown: {drawdown:.2%}\n"
            f"Portfolio: ${portfolio_value:,.0f}\n"
            f"Action: SELLING EVERYTHING\n"
            f"Will re-enter when DD recovers below 3%"
        )

    def profit_take_alert(self, pair: str, gain_pct: float, old_alloc: float, new_alloc: float):
        """Alert when profit-taking reduces a position."""
        coin = pair.split("/")[0]
        self._send(
            f"💰 <b>PROFIT TAKE: {coin}</b>\n\n"
            f"Position up {gain_pct:.1%}\n"
            f"Reducing: {old_alloc:.0%} → {new_alloc:.0%}"
        )

    def bot_started(self, portfolio_value: float, num_pairs: int):
        """Alert when bot starts."""
        self._send(
            f"🤖 <b>Bot Started</b>\n\n"
            f"💰 Portfolio: ${portfolio_value:,.0f}\n"
            f"📊 Pairs: {num_pairs}\n"
            f"⏱ Warmup: ~30 min\n"
            f"🔔 You'll get alerts on every trade + hourly updates"
        )

    def bot_crash(self, tick: int, error: str):
        """Alert on crash (bot auto-recovers)."""
        self._send(
            f"⚠️ <b>Bot Crash (auto-recovering)</b>\n\n"
            f"Tick: #{tick}\n"
            f"Error: {error[:200]}"
        )
