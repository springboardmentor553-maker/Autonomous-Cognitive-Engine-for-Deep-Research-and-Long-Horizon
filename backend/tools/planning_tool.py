import os
import json
import re
import requests
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def write_todos(request: str) -> List[Dict]:
    prompt = f"""
You are a structured planning system.

Break the objective into exactly 6 concise tasks.
Each description must be under 40 words.
Return STRICT valid JSON.
No markdown.
No bullet points.
No explanations.

Format:

[
  {{
    "id": 1,
    "title": "Task title",
    "description": "Short explanation",
    "status": "pending"
  }}
]

Objective:
{request}
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Autonomous-Cognitive-Engine",
            "Content-Type": "application/json",
        },
        json={
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
        },
        timeout=30
    )

    data = response.json()

    if "error" in data:
        raise Exception(f"OpenRouter API Error: {data}")

    if "choices" not in data:
        raise Exception(f"Unexpected API response: {data}")

    raw_text = data["choices"][0]["message"]["content"]

    # Extract JSON array safely
    match = re.search(r"\[[\s\S]*\]", raw_text)

    if not match:
        raise Exception(f"Could not extract JSON from model output:\n{raw_text}")

    clean_json = match.group(0)

    # Remove markdown fences only
    clean_json = clean_json.replace("```json", "")
    clean_json = clean_json.replace("```", "")

    # Remove control characters
    clean_json = re.sub(r"[\x00-\x1f\x7f]", " ", clean_json)

    # Remove trailing commas
    clean_json = re.sub(r",\s*]", "]", clean_json)
    clean_json = re.sub(r",\s*}", "}", clean_json)

    clean_json = clean_json.strip()

    try:
        tasks = json.loads(clean_json)
    except Exception:
        raise Exception(f"JSON parsing failed.\nCleaned output:\n{clean_json}")

    return tasks