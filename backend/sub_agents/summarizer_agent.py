from backend.config.gemini_config import client

def summarizer_agent(input_text):
    prompt = f"""
You are a professional AI report generator.

Your job is to convert the given analysis into a clean, structured, and highly readable report.

STRICT RULES:
- Keep it concise (max 200-250 words)
- Use ONLY these sections:
  1. 🔍 Overview (2 lines)
  2. 🚀 Key Points (max 5 bullets)
  3. ⚠️ Challenges (max 4 bullets)
  4. 🔮 Future Scope (max 3 bullets)
  5. ✅ Final Takeaway (1 line)

- No markdown (** or *)
- No repeated lines
- Use clean bullets (•)
- Make output visually clean
- Avoid long paragraphs
- Avoid unnecessary explanations
- Make it easy to read

Now process this input:

{input_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text