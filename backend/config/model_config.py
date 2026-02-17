import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Create one global client instance
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_llm():
    return client