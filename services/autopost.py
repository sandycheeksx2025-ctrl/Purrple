"""
Agent-based autoposting service.

Stable version with persistent duplicate guard and free tier limit enforcement.
"""

import json
import logging
import time
from typing import Any, Set

from services.database import Database
from services.llm import LLMClient
from services.twitter import TwitterClient
from tools.registry import TOOLS, get_tools_description
from config.personality import SYSTEM_PROMPT
from config.prompts.agent_autopost import AUTOPOST_AGENT_PROMPT
from config.schemas import PLAN_SCHEMA, POST_TEXT_SCHEMA, TOOL_REACTION_SCHEMA

logger = logging.getLogger(__name__)

# -----------------------
# In-memory run guard
# -----------------------
_IS_RUNNING = False
_LAST_RUN_TS = 0.0
_MIN_INTERVAL_SECONDS = 60 * 5  # 5 minutes safety

# Persistent duplicate guard (in-memory + DB-backed)
_RECENT_POSTS_SET: Set[str] = set()


def get_agent_system_prompt() -> str:
    tools_desc = get_tools_description()
    return AUTOPOST_AGENT_PROMPT.format(tools_desc=tools_desc)


class AutoPostService:
    """Agent-based autoposting service with persistent duplicate guard and free tier limit enforcement."""

    def __init__(self, db: Database, tier_manager=None):
        self.db = db
        self.llm = LLMClient()
        self.twitter = TwitterClient()
        self.tier_manager = tier_manager

    async def _load_recent_posts(self):
        """Load recent posts from database into duplicate guard set."""
        global _RECENT_POSTS_SET
        posts = await self.db.get_recent_posts_formatted(limit=100)
        _RECENT_POSTS_SET = set(posts)

    def _can_run_now(self) -> bool:
        global _IS_RUNNING, _LAST_RUN_TS

        now = time.time()

        if _IS_RUNNING:
            logger.info("[AUTOPOST] Skipped: already running")
            return False

        if (now - _LAST_RUN_TS) < _MIN_INTERVAL_SECONDS:
            logger.info("[AUTOPOST] Skipped: cooldown active")
            return False

        _IS_RUNNING = True
        _LAST_RUN_TS = now
        return True

    def _release_guard(self):
        global _IS_RUNNING
        _IS_RUNNING = False

    def _validate_plan(self, plan: list[dict]) -> None:
        if len(plan) > 3:
            raise ValueError("Plan too long")

        has_image = False
        for i, step in enumerate(plan):
            tool = step.get("tool")
            if tool not in TOOLS:
                raise ValueError(f"Unknown tool: {tool}")

            if tool == "generate_image":
                if has_image or i != len(plan) - 1:
                    raise ValueError("generate_image must be last")
                has_image = True

    async def run(self) -> dict[str, Any]:
        start = time.time()

        if not self._can_run_now():
            return {"success": False, "error": "guard_blocked"}

        try:
            # Load recent posts for duplicate check
            await self._load_recent_posts()

            # Tier guard
            if self.tier_manager:
                can_post, reason = self.tier_manager.can_post()
                if not can_post:
                    logger.info(f"[AUTOPOST] Blocked by tier: {reason}")
                    return {"success": False, "error": reason}

                # Free tier daily post limit enforcement
                daily_post_limit, _ = self.tier_manager.get_daily_limits()
                recent_posts_count = await self.db.get_today_post_count()
                if recent_posts_count >= daily_post_limit:
                    reason = f"daily_post_limit_reached ({recent_posts_count}/{daily_post_limit})"
                    logger.info(f"[AUTOPOST] Blocked by daily limit: {reason}")
                    return {"success": False, "error": reason}

            logger.info("[AUTOPOST] Starting run")

            previous_posts = await self.db.get_recent_posts_formatted(limit=50)

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + get_agent_system_prompt(),
                },
                {
                    "role": "user",
                    "content": f"""Create a Twitter post. Here are your previous posts (don't repeat):

{previous_posts}

Now create your plan.""",
                },
            ]

            plan_result = await self.llm.chat(messages, PLAN_SCHEMA)
            plan = plan_result["plan"]

            self._validate_plan(plan)
            messages.append({"role": "assistant", "content": json.dumps(plan_result)})

            image_bytes = None

            for step in plan:
                tool = step["tool"]
                params = step.get("params", {})

                if tool == "web_search":
                    result = await TOOLS[tool](params.get("query", ""))
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool result (web_search): {result.get('content', '')}",
                        }
                    )

                elif tool == "generate_image":
                    image_bytes = await TOOLS[tool](params.get("prompt", ""))
                    messages.append(
                        {
                            "role": "user",
                            "content": "Tool result (generate_image): success",
                        }
                    )

                reaction = await self.llm.chat(messages, TOOL_REACTION_SCHEMA)
                messages.append(
                    {"role": "assistant", "content": reaction.get("thinking", "")}
                )

            messages.append(
                {
                    "role": "user",
                    "content": "Now write your final tweet text (max 280 characters).",
                }
            )

            post_result = await self.llm.chat(messages, POST_TEXT_SCHEMA)
            post_text = post_result["post_text"].strip()

            # Check duplicate post
            if post_text in _RECENT_POSTS_SET:
                logger.info("[AUTOPOST] Skipped: duplicate post detected")
                return {"success": False, "error": "duplicate_post"}

            if len(post_text) > 280:
                post_text = post_text[:277] + "..."

            media_ids = None
            if image_bytes:
                media_id = await self.twitter.upload_media(image_bytes)
                media_ids = [media_id]

            tweet = await self.twitter.post(post_text, media_ids=media_ids)
            await self.db.save_post(post_text, tweet["id"], image_bytes is not None)

            # Add to in-memory duplicate set
            _RECENT_POSTS_SET.add(post_text)

            duration = round(time.time() - start, 2)
            logger.info(f"[AUTOPOST] Done in {duration}s")

            return {
                "success": True,
                "tweet_id": tweet["id"],
                "text": post_text,
            }

        except Exception as e:
            logger.exception("[AUTOPOST] Failed")
            return {"success": False, "error": str(e)}

        finally:
            self._release_guard()
