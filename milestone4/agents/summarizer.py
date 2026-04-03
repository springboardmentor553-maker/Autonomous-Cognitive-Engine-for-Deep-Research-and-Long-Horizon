from utils.llm import llm

def summarizer(text):

    # 🔴 Handle empty input
    if not text or not text.strip():
        return "No content to summarize."

    # 🔴 Truncate long input (avoid token overflow)
    text = text.strip()[:800]

    try:
        response = llm.invoke(
            f"""
            Summarize the following in 3–4 clear bullet points.
            Keep it concise and simple.

            Text:
            {text}
            """
        )

        return response.content.strip()

    except Exception as e:
        return f"Summary failed: {str(e)}"