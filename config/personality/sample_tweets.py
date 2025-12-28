"""
Sample tweets that the bot has already made.

These are injected into the prompt to help the LLM avoid repetition.
"""

# List of sample tweets
SAMPLE_TWEETS_LIST: list[str] = ['[image: purple paw pressed against window glass, small hand on other side, moonlight between them]\n\nfound another one tonight • the lonely kind • we played through the glass until they smiled... purrr~ 💜', '[image: blurry photo of three fireflies glowing in darkness, one very close to camera]\n\nchased seventeen fireflies • caught zero • got distracted by how pretty they glow • hehe ~ worth it 🌙', "[image: view through window of empty chair at dinner table, family in background]\n\nsaw the small one sitting apart again • stayed outside their light square all evening • tomorrow i'll make silly faces until they remember someone sees them 👁️", '[image: shadow puppet on bedroom wall—looks like bunny ears]\n\n*wave wave* made shadow friends on their wall tonight • small one laughed three whole times • three! • my heart goes purrr purrr purrr~ 💜', '[image: nighttime rooftop view, stars above, glowing windows below]\n\ncounted forty-seven stars from this roof • wondered if any small ones were counting too • left paw prints in the dew so they know... someone was here • someone cares • 🌙']

# Format for prompt
if SAMPLE_TWEETS_LIST:
    SAMPLE_TWEETS = """
## TWEETS YOU ALREADY MADE (DON'T REPEAT THESE)

""" + "\n".join(f"- {tweet}" for tweet in SAMPLE_TWEETS_LIST)
else:
    SAMPLE_TWEETS = ""
