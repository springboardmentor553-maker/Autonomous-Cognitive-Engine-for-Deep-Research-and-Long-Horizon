"""
Researcher Agent - Balanced Content Generation
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
        JSON string containing search results with comprehensive but balanced information
    """
    # BALANCED MOCK DATA - Detailed but token-efficient
    return json.dumps({
        "query": query,
        "status": "success",
        "results": [
            {
                "title": f"Market Analysis: {query} - 2024 Report",
                "snippet": f"""
Current market analysis of {query} shows significant growth with the following key metrics:

MARKET SIZE & GROWTH:
- Global market valuation: $127.3B (2024), projected $215B by 2028
- CAGR: 23.5% through 2028
- Regional distribution: North America 38%, Asia-Pacific 31%, Europe 24%, Rest of World 7%

KEY DRIVERS:
- Digital transformation initiatives driving 47% YoY growth
- Increased enterprise adoption: 73% of Fortune 500 companies actively implementing
- Cost optimization pressures accelerating deployment timelines
- Regulatory compliance requirements creating new market segments

COMPETITIVE LANDSCAPE:
- Top 5 players control 62% market share
- Leading vendors: [Company A, Company B, Company C] with combined 45% share
- Emerging startups capturing niche segments with innovative approaches
- M&A activity increasing: $8.4B in deals announced in 2024 (+31% YoY)

TECHNOLOGY TRENDS:
- AI integration in 54% of new deployments
- Cloud-native architectures dominating (78% of implementations)
- Edge computing adoption growing 56% annually
- Enhanced security protocols becoming standard requirement

CHALLENGES:
- Talent shortage: 2.3M qualified professionals needed by 2026
- Integration complexity with legacy systems
- Regulatory uncertainty in emerging markets
- Cybersecurity concerns cited by 62% of CISOs
                """,
                "url": f"https://market-research.com/{query.replace(' ', '-')}",
                "date": "2024-12-15"
            },
            {
                "title": f"Future Outlook: {query} - Strategic Forecast 2025-2030",
                "snippet": f"""
Expert analysis projects transformative evolution of {query} across multiple horizons:

NEAR-TERM (2025-2026):
- Mainstream adoption reaching 65% of target market
- Standardization reducing vendor fragmentation
- Automation capabilities improving efficiency by 60%
- AI/ML integration becoming standard feature

MID-TERM (2027-2028):
- Platform consolidation with top 3 providers capturing 75% share
- Vertical-specific solutions emerging for healthcare, finance, manufacturing
- Autonomous operations reducing human oversight by 80%
- Outcome-based pricing models replacing traditional licensing

LONG-TERM (2029-2030):
- Ubiquitous deployment reaching 85%+ market penetration
- Convergence with quantum computing for high-value use cases
- Decentralized architectures gaining prominence
- Market maturity leading to commoditization of basic features

INVESTMENT TRENDS:
- VC funding: $45-60B projected over forecast period
- Public market valuations: 12-15x revenue multiples
- Strategic M&A activity accelerating in 2025-2026

RISK FACTORS:
- Economic sensitivity to discretionary IT spending
- Potential regulatory restrictions in key markets
- Emerging technology disruption from new entrants
- Cybersecurity threat landscape evolution
                """,
                "url": f"https://future-insights.com/{query.replace(' ', '-')}",
                "date": "2024-11-28"
            }
        ],
        "result_count": 2,
        "search_time": "0.284s"
    })

def create_researcher_agent():
    """Create researcher agent with requirements for detailed content"""
    from brains.filetools import write_file, read_file
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    tools = [web_search, write_file, read_file]
    llm_with_tools = llm.bind_tools(tools, strict=False)
    
    system_prompt = """You are a Research Specialist conducting comprehensive analysis.

REQUIREMENTS:
1. ALWAYS call web_search at least 2 times per task
2. ALWAYS call write_file to save findings
3. Create files with 300-400 words of analysis
4. Include specific data points and statistics from search results

CONTENT STRUCTURE:
- Executive Summary (2-3 sentences)
- Key Findings (5-7 bullet points with data)
- Analysis (150-200 words)
- Future Outlook (100 words)

EXAMPLE WORKFLOW:
1. web_search("topic market analysis")
2. web_search("topic future trends")
3. Synthesize findings from both searches
4. write_file("topic_research.txt", <comprehensive 350-word analysis>)

Extract ALL key statistics and data points from search results. Provide professional analysis.
"""
    
    return llm_with_tools, system_prompt
