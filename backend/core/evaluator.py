from backend.config.gemini_config import client

def evaluate_output(final_report):

    print("\n🔍 EVALUATION STARTED...\n")

    prompt = f"""
Evaluate this report:

{final_report}

Give:
- Score (1-10)
- Strengths
- Improvements
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\n📊 EVALUATION RESULT:\n")
    print(response.text)

    return response.text