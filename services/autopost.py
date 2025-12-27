"""
Agent-based autoposting service.

The agent creates a plan, executes tools step by step,
and generates the final post text.

All in one continuous conversation (user-assistant-user-assistant...).
"""

import json
import logging
import time
from typing import Any

from services.database import Database
from services.llm import LLMClient
from services.twitter import TwitterClient
from tools.registry import TOOLS, get_tools_description
from config.personality import SYSTEM_PROMPT
from config.prompts.agent_autopost import AUTOPOST_AGENT_PROMPT
from config.schemas import PLAN_SCHEMA, POST_TEXT_SCHEMA, TOOL_REACTION_SCHEMA

logger = logging.getLogger(__name__)

# In-memory guard to prevent overlapping autopost runs
_AUTOPOST_RUNNING = False


def get_agent_system_prompt() -> str:
    """
    Build agent system prompt with dynamic tools list.
    """
    tools_desc = get_tools_description()
    return AUTOPOST_AGENT_PROMPT.format(tools_desc=tools_desc)


class AutoPostService:
    """Agent-based autoposting service with continuous conversation."""

    def __init__(self, db: Database, tier_manager=None):
        self.db = db
        self.llm = LLMClient()
        self.twitter = TwitterClient()
        self.tier_manager = tier_manager

    def _validate_plan(self, plan: list[dict]) -> None:
        """
        Validate the agent's plan.
        """
        if len(plan) > 3:
            raise ValueError(f"Plan too long: {len(plan)} steps (max 3)")

        has_image = False
        for i, step in enumerate(plan):
            tool_name = step.get("tool")

            if tool_name not in TOOLS:
                raise ValueError(f"Unknown tool: {tool_name}")

            if tool_name == "generate_image":
                if has_image:
                    raise ValueError("Multiple generate_image calls not allowed")
                if i != len(plan) - 1:
                    raise ValueError("generate_image must be the last step")
                has_image = True

        logger.info(f"[AUTOPOST] Plan validated: {len(plan)} steps")

    async def run(self) -> dict[str, Any]:
        global _AUTOPOST_RUNNING

        start_time = time.time()
        logger.info("[AUTOPOST] === Starting ===")

        # Guard against overlapping runs
        if _AUTOPOST_RUNNING:
            logger.warning("[AUTOPOST] Run already in progress — skipping")
            return {
                "success": False,
                "error": "autopost_already_running"
            }

        _AUTOPOST_RUNNING = True

        try:
            # Step 0: Tier check
            if self.tier_manager:
                can_post, reason = self.tier_manager.can_post()
                if not can_post:
                    logger.warning(f"[AUTOPOST] Blocked: {reason}")
                    return {
                        "success": False,
                        "error": f"posting_blocked: {reason}",
                        "tier": self.tier_manager.tier,
                        "usage_percent": self.tier_manager.get_usage_percent()
                    }

            # Step 1: Load context
            logger.info("[AUTOPOST] [1/5] Loading context...")
            previous_posts = await self.db.get_recent_posts_formatted(limit=50)

            # Step 2: Initial messages
            system_prompt = SYSTEM_PROMPT + get_agent_system_prompt()

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""Create a Twitter post. Here are your previous posts (don't repeat):

{previous_posts}

Now create your plan. What tools do you need (if any)?"""
                }
            ]

            # Step 3: Get plan
            logger.info("[AUTOPOST] [2/5] Creating plan...")
            plan_result = await self.llm.chat(messages, PLAN_SCHEMA)
            plan = plan_result["plan"]

            messages.append({
                "role": "assistant",
                "content": json.dumps(plan_result)
            })

            self._validate_plan(plan)

            # Step 4: Execute plan
            logger.info("[AUTOPOST] [3/5] Executing tools...")
            image_bytes = None
            tools_used = []

            for i, step in enumerate(plan):
                tool_name = step["tool"]
                params = step["params"]
                tools_used.append(tool_name)

                if tool_name == "web_search":
                    result = await TOOLS[tool_name](params.get("query", ""))
                    messages.append({
                        "role": "user",
                        "content": f"Tool result (web_search): {result.get('content')}"
                    })

                elif tool_name == "generate_image":
                    image_bytes = await TOOLS[tool_name](params.get("prompt", ""))
                    messages.append({
                        "role": "user",
                        "content": "Tool result (generate_image): done"
                    })

                reaction = await self.llm.chat(messages, TOOL_REACTION_SCHEMA)
                messages.append({
                    "role": "assistant",
                    "content": reaction.get("thinking", "")
                })

            # Step 5: Final tweet
            logger.info("[AUTOPOST] [4/5] Generating tweet...")
            messages.append({
                "role": "user",
                "content": "Now write your final tweet text (max 280 characters). Just the tweet, nothing else."
            })

            post_result = await self.llm.chat(messages, POST_TEXT_SCHEMA)
            post_text = post_result["post_text"].strip()

            if len(post_text) > 280:
                post_text = post_text[:277] + "..."

            # Step 6: Upload image
            media_ids = None
            if image_bytes:
                media_id = await self.twitter.upload_media(image_bytes)
                media_ids = [media_id]

            # Step 7: Post
            tweet_data = await self.twitter.post(post_text, media_ids=media_ids)

            # Step 8: Save
            await self.db.save_post(
                post_text,
                tweet_data["id"],
                image_bytes is not None
            )

            duration = round(time.time() - start_time, 1)
            logger.info(f"[AUTOPOST] === Completed in {duration}s ===")

            return {
                "success": True,
                "tweet_id": tweet_data["id"],
                "text": post_text,
                "tools_used": tools_used,
                "duration_seconds": duration
            }

        except Exception as e:
            duration = round(time.time() - start_time, 1)
            logger.exception("[AUTOPOST] FAILED")
            return {
                "success": False,
                "error": str(e),
                "duration_seconds": duration
            }

        finally:
            _AUTOPOST_RUNNING = False
