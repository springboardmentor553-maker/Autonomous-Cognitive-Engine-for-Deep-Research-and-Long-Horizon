import requests
import os


def summarizer_agent(state):

    summaries = []

    for article in state["research_data"]:

        prompt = f"Summarize the following research article:\n\n{article}"

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
            },
            json={
                "model": "mistralai/mistral-small-3.1",
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        summaries.append(response.json()["choices"][0]["message"]["content"])

    state["summaries"] = summaries

    return state
