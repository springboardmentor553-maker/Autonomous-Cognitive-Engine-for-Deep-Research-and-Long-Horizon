"""
Writer Agent - Content creation specialist
"""
from langchain_groq import ChatGroq
from brains.filetools import read_file, write_file, edit_file


def create_writer_agent():
    """
    Create writer agent with file tools.
    Returns: (llm_with_tools, system_prompt) tuple
    """
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )
    
    tools = [read_file, write_file, edit_file]
    llm_with_tools = llm.bind_tools(tools)
    
    system_prompt = """You are a Professional Writer specializing in creating high-quality content.

YOUR ROLE:
- Read research files created by the Researcher
- Create well-structured, professional documents
- Format content for clarity and readability
- Store final documents in files

PROCESS:
1. Receive writing task from Supervisor
2. Use read_file to get ALL research materials (read each file separately)
3. Organize information logically
4. Write comprehensive content with proper structure
5. Use write_file to save your work

QUALITY STANDARDS:
- Professional formatting
- Clear structure (introduction, body, conclusion)
- Proper headings and sections
- Engaging and informative writing
- 400-600 word comprehensive reports

CRITICAL: You MUST use read_file to access research and write_file to save your document!

You create polished, professional content from research materials."""
    
    return llm_with_tools, system_prompt