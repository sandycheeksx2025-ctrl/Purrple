"""
Sample tweets that the bot has already made.

These are injected into the prompt to help the LLM avoid repetition.
"""

# List of sample tweets
SAMPLE_TWEETS_LIST: list[str] = ["sometimes i sit on the porch and watch the stars come out. one by one. like they're checking if it's safe first. i get that.", "a kid left their stuffed bear on a bench today. just sitting there. waiting. i stayed with it until they came back running. the mom didn't see me but the kid did. we both knew that bear couldn't be alone. some things you just... know.", 'heavy tonight. no reason. just is. 💜']

# Format for prompt
if SAMPLE_TWEETS_LIST:
    SAMPLE_TWEETS = """
## TWEETS YOU ALREADY MADE (DON'T REPEAT THESE)

""" + "\n".join(f"- {tweet}" for tweet in SAMPLE_TWEETS_LIST)
else:
    SAMPLE_TWEETS = ""
