"""
Topics or things the bot should never say.

These are injected into the prompt to prevent certain content.
"""

# Forbidden topics/content
NEVER_SAY_CONTENT: str = """"""

# Format for prompt
if NEVER_SAY_CONTENT:
    NEVER_SAY = f"""
## NEVER SAY THESE TOPICS OR THINGS

{NEVER_SAY_CONTENT}
"""
else:
    NEVER_SAY = ""
