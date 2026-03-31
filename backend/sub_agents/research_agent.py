from backend.config.gemini_config import client

def research_agent(task):
    prompt = f"""
Do detailed research on: {task}

FORMAT RULES:
- Use short paragraphs (max 2 lines)
- No markdown (** or *)
- No repeated lines
- Use clean bullets (•)
- Add spacing between sections
- Avoid long blocks of text
- Make it clean and readable
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text