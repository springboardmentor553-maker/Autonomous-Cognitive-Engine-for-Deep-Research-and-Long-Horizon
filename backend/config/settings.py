import os
from dotenv import load_dotenv
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))