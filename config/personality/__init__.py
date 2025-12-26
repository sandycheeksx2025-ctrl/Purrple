"""
Personality module - Combines all personality parts into SYSTEM_PROMPT.
"""

from config.personality.backstory import BACKSTORY
from config.personality.beliefs import BELIEFS
from config.personality.instructions import INSTRUCTIONS
from config.personality.sample_tweets import SAMPLE_TWEETS
from config.personality.never_say import NEVER_SAY

# Combine all parts into the final system prompt
SYSTEM_PROMPT = f"""{BACKSTORY}
{BELIEFS}
{INSTRUCTIONS}
{NEVER_SAY}
{SAMPLE_TWEETS}
"""

__all__ = ["SYSTEM_PROMPT", "BACKSTORY", "BELIEFS", "INSTRUCTIONS", "SAMPLE_TWEETS", "NEVER_SAY"]
