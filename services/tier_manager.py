import logging
import time

logger = logging.getLogger(__name__)

TIER_UNKNOWN = "UNKNOWN"
TIER_FREE = "FREE"
TIER_BASIC = "BASIC"
TIER_PRO = "PRO"


class TierManager:
    """
    Tier manager with in-memory backoff.
    Safe for free-tier and restart-heavy environments.
    """

    def __init__(self, twitter_client):
        self.twitter = twitter_client
        self.tier = TIER_UNKNOWN
        self.last_check_ts = 0.0
        self.backoff_seconds = 60 * 15  # 15 minutes
        self.usage_percent = 0.0

        logger.info("[TIER] Initializing tier manager...")

    # -------------------------
    # Compatibility method
    # -------------------------
    async def initialize(self):
        """
        Called by main.py on startup.
        Must exist.
        """
        await self.detect_tier()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _in_backoff(self) -> bool:
        return (time.time() - self.last_check_ts) < self.backoff_seconds

    async def detect_tier(self) -> None:
        """
        Detect account tier with graceful backoff.
        """
        if self._in_backoff():
            logger.info("[TIER] Skipping tier detection (backoff active)")
            return

        self.last_check_ts = time.time()

        try:
            usage = await self.twitter.get_tweet_usage()
            used = usage.get("used", 0)
            limit = usage.get("limit", 0)

            if limit <= 0:
                raise ValueError("Invalid usage data")

            self.usage_percent = (used / limit) * 100

            if limit <= 100:
                self.tier = TIER_FREE
            elif limit <= 1000:
                self.tier = TIER_BASIC
            else:
                self.tier = TIER_PRO

            logger.info("==================================================")
            logger.info(f"[TIER] Detected tier: {self.tier}")
            logger.info(f"[TIER] Usage: {used}/{limit} ({self.usage_percent:.1f}%)")
            logger.info("==================================================")

        except Exception as e:
            self.tier = TIER_UNKNOWN
            logger.warning(f"[TIER] Tier detection failed, entering backoff: {e}")

    # -------------------------
    # Public API
    # -------------------------
    def can_post(self) -> tuple[bool, str]:
        if self.tier == TIER_UNKNOWN:
            return False, "tier_unknown"

        if self.tier == TIER_FREE and self.usage_percent >= 95:
            return False, "free_tier_near_limit"

        return True, "ok"

    def get_usage_percent(self) -> float:
        return self.usage_percent

    async def maybe_refresh_tier(self):
        """
        Safe scheduled refresh.
        """
        await self.detect_tier()
