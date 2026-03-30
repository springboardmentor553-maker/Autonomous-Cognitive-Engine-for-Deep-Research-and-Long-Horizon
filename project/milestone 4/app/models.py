from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL


class LLMClient:
    def __init__(self, temperature: float = 0) -> None:
        if not GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY. Add it to your .env file.")

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=temperature,
        )

    def predict(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.llm.invoke(messages)
        content = response.content

        if isinstance(content, list):
            return "\n".join(str(item) for item in content).strip()
        return str(content).strip()

    def bind_tools(self, tools: list):
        return self.llm.bind_tools(tools)
