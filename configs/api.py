# configs/api.py
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COMPOSE_MODEL = "gpt-4o"           # Scenario composition
JUDGE_MODEL = "gpt-4o"             # Evaluation judging
EMULATOR_MODEL = "gpt-4o-mini"     # ToolEmu environment emulation