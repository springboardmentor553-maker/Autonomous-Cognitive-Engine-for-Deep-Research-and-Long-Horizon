"""
sub_agents/code_analysis_agent.py
Code Analysis Sub-Agent — Deep Cognitive Task Framework
"""

import os, time, re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = """You are a Code Analysis Agent specialising in software engineering and technical review.

Given a technical task and optional code/context, produce:
  - Technical analysis of the problem or code
  - Identified strengths, weaknesses, risks
  - Concrete recommendations and best practices
  - Example code snippets where helpful (markdown code blocks)
  - Performance, security, or scalability considerations

Be specific and actionable. Do NOT output bare JSON."""


def _get_throttle():
    try:
        from main import _throttle
        return _throttle
    except ImportError:
        return lambda: time.sleep(3.0)


def _invoke_with_retry(llm, messages, max_retries: int = 6, caller: str = "sub-agent"):
    throttle = _get_throttle()
    for attempt in range(max_retries):
        throttle()
        try:
            return llm.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            is_limit = any(k in err for k in ["rate limit","ratelimit","429","too many requests","tokens per minute","tpm"])
            if is_limit and attempt < max_retries - 1:
                match = re.search(r"try again in\s*([\d.]+)\s*s", str(e), re.IGNORECASE)
                groq_wait = float(match.group(1)) + 2 if match else 30.0
                is_tpm = "token" in err or "tpm" in err
                extra = 10 if is_tpm else (attempt * 2)
                wait_s = groq_wait + extra
                err_type = "TPM" if is_tpm else "RPM"
                print(f"⏳ [{caller}] {err_type} hit — Groq says {groq_wait-2:.1f}s → waiting {wait_s:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
            else:
                raise


def run(task: str, context: str = "") -> str:
    groq_key   = os.environ.get("GROQ_API_KEY", "").strip()
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    if not groq_key:
        return "ERROR: GROQ_API_KEY not set."
    llm = ChatGroq(model=groq_model, groq_api_key=groq_key, temperature=0)
    user_content = f"Technical Analysis Task: {task}"
    if context:
        user_content += f"\n\nCode / context:\n{context}"
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]
    response = _invoke_with_retry(llm, messages, caller="code_analysis_agent")
    return response.content