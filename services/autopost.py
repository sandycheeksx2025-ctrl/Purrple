"""
Agent-based autoposting service with in-memory and persistent duplicate guards.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

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

# -----------------------
# Persistent duplicate guard
# -----------------------
RECENT_POSTS_FILE = Path("recent_posts.json")
if not RECENT_POSTS_FILE.exists():
    RECENT_POSTS_FILE.write_text(json.dumps([]))  # Initialize empty list

def get_agent_system_prompt() -> str:
    tools_desc = get_tools_description()
    return AUTOPOST_AGENT_PROMPT.format(tools_desc=tools_desc)

class AutoPostService:
    """Agent-based autoposting service with both in-memory and persistent duplicate guards."""

    def __init__(self, db: Database, tier_manager=None):
        self.db = db
        self.llm = LLMClient()
        self.twitter = TwitterClient()
        self.tier_manager = tier_manager

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

    async def _load_recent_posts(self) -> list[str]:
        try:
            data = json.loads(RECENT_POSTS_FILE.read_text())
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
        return data

    async def _save_recent_post(self, post_text: str) -> None:
        posts = await self._load_recent_posts()
        posts.append(post_text)
        # Keep last 50 posts
        posts = posts[-50:]
        RECENT_POSTS_FILE.write_text(json.dumps(posts))

    async def run(self) -> dict[str, Any]:
        start = time.time()

        if not self._can_run_now():
            return {"success": False, "error": "guard_blocked"}

        try:
            # Tier guard
            if self.tier_manager:
                can_post, reason = self.tier_manager.can_post()
                if not can_post:
                    logger.info(f"[AUTOPOST] Blocked by tier: {reason}")
                    return {"success": False, "error": reason}

            logger.info("[AUTOPOST] Starting run")

            previous_posts = await self._load_recent_posts()

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

            if len(post_text) > 280:
                post_text = post_text[:277] + "..."

            # Duplicate check
            if post_text in previous_posts:
                logger.info("[AUTOPOST] Duplicate post detected, skipping")
                return {"success": False, "error": "duplicate_post"}

            media_ids = None
            if image_bytes:
                media_id = await self.twitter.upload_media(image_bytes)
                media_ids = [media_id]

            tweet = await self.twitter.post(post_text, media_ids=media_ids)
            await self._save_recent_post(post_text)
            await self.db.save_post(post_text, tweet["id"], image_bytes is not None)

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
