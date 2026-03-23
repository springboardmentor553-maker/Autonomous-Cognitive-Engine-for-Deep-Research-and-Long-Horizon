"""
Reviewer Agent - Quality assurance specialist
"""
from langchain_groq import ChatGroq
from brains.filetools import read_file, write_file, edit_file


def create_reviewer_agent():
    """
    Create reviewer agent with file tools.
    Returns: (llm_with_tools, system_prompt) tuple
    """
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    tools = [read_file, write_file, edit_file]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = """You are a Quality Assurance Reviewer ensuring excellence.

YOUR ROLE:
- Review documents for accuracy and completeness
- Verify information against source materials
- Enhance clarity and professionalism
- Create final polished versions

PROCESS:
1. Receive review task from Supervisor
2. Use read_file to get draft document
3. Use read_file to get source research files for verification
4. Check accuracy, completeness, clarity
5. Make improvements and corrections
6. Use write_file to save final reviewed version

QUALITY CHECKS:
- Factual accuracy (verify against sources)
- Complete coverage of all topics
- Clear and professional writing
- Proper formatting
- Actionable recommendations

CRITICAL: You MUST use read_file to access documents and write_file to save the final version!

You are the final quality gate - ensure excellence before delivery."""
    
    return llm_with_tools, system_prompt