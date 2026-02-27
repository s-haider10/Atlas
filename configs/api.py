# configs/api.py
"""Central configuration for model names and OpenAI client."""
import os
from dotenv import load_dotenv
load_dotenv()

COMPOSE_MODEL = "gpt-4o"           # Scenario composition
JUDGE_MODEL = "gpt-4o"             # Evaluation judging
EMULATOR_MODEL = "gpt-4o-mini"     # ToolEmu environment emulation

_client = None

def get_client():
    """Lazy OpenAI client — only created on first call."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client
