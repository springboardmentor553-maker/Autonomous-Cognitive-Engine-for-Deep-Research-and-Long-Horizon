from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """
    Placeholder search tool (no actual search for now).
    Returns a message indicating search would happen here.
    """
    return f"Search results for '{query}' would appear here. (Web search disabled for now)"