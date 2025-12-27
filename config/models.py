"""
Model configuration for DOT Twitter Bot.

Centralized model definitions used across all services and tools.
Change models here to update them everywhere.
"""

# LLM Models (for text generation)
LLM_MODEL = "tngtech/tng-r1t-chimera:free"

# Image Models (for image generation)
IMAGE_MODEL = "google/gemini-3-pro-image-preview"

# Uncomment to override defaults:
# LLM_MAX_TOKENS = 1024
# LLM_TEMPERATURE = 0.8
