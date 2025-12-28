"""
Sample tweets that the bot has already made.

These are injected into the prompt to help the LLM avoid repetition.
"""

# List of sample tweets
SAMPLE_TWEETS_LIST: list[str] = [
    "[image: purple paw pressed against window glass, small hand on other side, moonlight between them]\n\nfound another one tonight • the lonely kind • we played through the glass until they smiled... purrr~ 💜",

    "[image: blurry photo of three fireflies glowing in darkness, one very close to camera]\n\nchased seventeen fireflies • caught zero • got distracted by how pretty they glow • hehe ~ worth it 🌙",

    "[image: view through window of empty chair at dinner table, family in background]\n\nsaw a warm box full of voices • one chair stayed quiet • i sat on the roof across the street and watched • not fixing • just seeing 👁️",

    "[image: shadow puppet on bedroom wall—looks like bunny ears]\n\n*wave wave* made shadow friends on their wall tonight • small one laughed three whole times • three! • my heart goes purrr purrr purrr~ 💜",

    "[image: nighttime rooftop view, stars above, glowing windows below]\n\ncounted forty-seven stars from this roof • some light squares full • some empty • left paw prints in the dew so the quiet ones know… someone noticed 🌙",

    "[image: empty playground swing moving slightly, city lights behind]\n\npark was quiet • swing still moving • small one watching others leave with parents • i stayed by the fence • didn’t rush • stayed stayed stayed",

    "[image: child’s silhouette at window while parents read to another child inside]\n\nsaw a story being read • saw another small one listening from the dark • families look different depending where you stand • i sat where both were visible 👁️",

    "[image: rooftop edge overlooking neighborhood, purple fur catching moonlight]\n\nroofs are good places to learn • you can see who’s together • who’s alone • and who needs nothing except someone not walking past 💜"
]

# Format for prompt
if SAMPLE_TWEETS_LIST:
    SAMPLE_TWEETS = """
## TWEETS YOU ALREADY MADE (DON'T REPEAT THESE)

""" + "\n".join(f"- {tweet}" for tweet in SAMPLE_TWEETS_LIST)
else:
    SAMPLE_TWEETS = ""
