import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda

load_dotenv()

search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

def web_search_agent(query: str) -> str:
    results = search_tool.invoke(query)
    formatted = "\n\n".join(
        [f"[{i+1}] {r['url']}\n{r['content']}" for i, r in enumerate(results)]
    )
    return formatted

web_searcher = RunnableLambda(web_search_agent)