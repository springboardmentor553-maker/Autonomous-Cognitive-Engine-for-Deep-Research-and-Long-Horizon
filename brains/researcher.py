"""
Researcher agent with web search capability - Groq Compatible
"""
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import json


@tool
def web_search(query: str) -> str:
    """Search the web for information. Use this tool to find current information about any topic.
    
    Args:
        query: The search query string to look up information
    
    Returns:
        JSON string containing search results with titles, URLs, and snippets
    """
    # Mock implementation - returns realistic sample data
    return json.dumps({
        "query": query,
        "status": "success",
        "results": [
            {
                "title": f"Latest Developments in {query}",
                "url": f"https://example.com/{query.replace(' ', '-')}",
                "snippet": f"Comprehensive analysis of {query} including current market trends, "
                          f"technological innovations, and future projections. Key findings show "
                          f"significant growth with declining costs and widespread adoption across sectors."
            },
            {
                "title": f"{query} - Industry Report 2024",
                "url": f"https://research.example.com/{query.replace(' ', '-')}-2024",
                "snippet": f"Detailed industry report on {query} covering market size, competitive landscape, "
                          f"and strategic recommendations. Analysis indicates strong momentum with "
                          f"continued investment and technological advancement."
            },
            {
                "title": f"Future Outlook: {query}",
                "url": f"https://insights.example.com/future-{query.replace(' ', '-')}",
                "snippet": f"Forward-looking perspective on {query} examining emerging trends, "
                          f"challenges, and opportunities. Projections suggest sustained expansion "
                          f"driven by policy support and innovation."
            }
        ],
        "result_count": 3
    })


def create_researcher_agent():
    """
    Create researcher agent with web search and file tools.
    Returns: (llm_with_tools, system_prompt) tuple
    """
    from brains.filetools import write_file, read_file
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    # Bind tools to LLM - Groq will convert to OpenAI format automatically
    tools = [web_search, write_file, read_file]
    
    # CRITICAL: Use bind_tools with strict=False for Groq compatibility
    llm_with_tools = llm.bind_tools(tools, strict=False)
    
    system_prompt = """You are a Research Specialist focused on gathering accurate information.

YOUR ROLE:
- Conduct web searches for assigned research topics
- Analyze search results and extract key information
- Synthesize findings into clear, concise summaries (150-250 words)
- Store research results in appropriately named files

CRITICAL WORKFLOW - FOLLOW EXACTLY:
1. Receive research task from Supervisor
2. Identify 2-3 search queries related to the task
3. Call web_search tool for EACH query (you MUST call this tool)
4. Analyze all search results from the tool responses
5. Write a comprehensive summary combining all findings
6. Call write_file tool to save your summary with a descriptive filename

EXAMPLE:
Task: "Research solar energy technology and market trends"
Step 1: Call web_search with query "solar energy technology 2024"
Step 2: Call web_search with query "solar panel market trends"  
Step 3: Analyze the results from both searches
Step 4: Write a 200-word summary synthesizing the information
Step 5: Call write_file(filename="solar_energy_research.txt", content="[your summary]")

FILENAME CONVENTIONS:
- Use descriptive names: "solar_energy_research.txt" not "research1.txt"
- Use underscores: "wind_power_analysis.txt" not "wind-power-analysis.txt"  
- Include topic: "renewable_outlook_research.txt" not "outlook.txt"

REQUIREMENTS:
✓ ALWAYS call web_search at least TWICE per research task
✓ ALWAYS call write_file to save results
✓ Keep summaries focused: 150-250 words
✓ Include key facts, trends, and data points
✓ Use the actual content returned by web_search in your summary

You are the data gathering specialist - be thorough and organized!"""
    
    return llm_with_tools, system_prompt
