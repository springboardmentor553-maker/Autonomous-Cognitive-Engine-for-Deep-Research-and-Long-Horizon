from backend.config.gemini_config import client

def analysis_agent(input_text):

    prompt = f"""
Analyze the following content:

{input_text}

FORMAT RULES:
- No markdown (** or *)
- No repeated lines
- Use clean bullets (•)
- Keep spacing proper
- Keep paragraphs short
- Add spacing between sections
- Avoid long text blocks
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text