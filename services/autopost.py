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

# In-memory guard to prevent overlapping autopost runs (single-process safety)
_AUTOPOST_RUNNING = False


def get_agent_system_prompt() -> str:
    """
    Build agent system prompt with dynamic tools list.

    Tools are loaded from registry, so adding a new tool to TOOLS_SCHEMA
    automatically makes it available to the agent.
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

        Rules:
        - generate_image must be last if present
        - Only known tools allowed
        - Max 3 steps
        """
        if len(plan) > 3:
            raise ValueError(f"Plan too long: {len(plan)} steps (max 3)")

        has_image = False
        for i, step in e
