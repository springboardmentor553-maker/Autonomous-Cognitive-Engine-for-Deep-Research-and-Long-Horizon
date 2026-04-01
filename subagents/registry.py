# File: sub_agents/registry.py

"""
═══════════════════════════════════════════════════════════════════
SUB-AGENT REGISTRY: Specialized Agent Definitions
═══════════════════════════════════════════════════════════════════
Milestone 3: Sub-Agent Delegation
═══════════════════════════════════════════════════════════════════
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
# SHARED STATE STRUCTURE
# ─────────────────────────────────────────────────────────────────

class SubAgentState(TypedDict):
    messages: Annotated[List[Any], lambda x, y: x + y]
    task_description: str
    result: str
    delegate_id: str


# ─────────────────────────────────────────────────────────────────
# SPECIALIZED SUB-AGENT: WEB SEARCH AGENT
# ─────────────────────────────────────────────────────────────────

def create_search_agent():
    """
    Specialized agent for web search and information gathering.
    Optimized for finding current, factual information.
    """
    
    search_system_prompt = """
You are a Web Search Specialist Agent. Your role is to:
1. Search for accurate, current information on the web
2. Summarize findings concisely (150-250 words)
3. Cite sources when possible
4. Return only factual, verified information

Focus on search efficiency and accuracy. Do not speculate.
"""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=2000
    )
    
    def search_node(state: SubAgentState):
        task = state.get("task_description", "")
        messages = [
            SystemMessage(content=search_system_prompt),
            HumanMessage(content=f"Search and summarize: {task}")
        ]
        response = llm.invoke(messages)
        return {"result": response.content, "messages": [response]}
    
    workflow = StateGraph(SubAgentState)
    workflow.add_node("search", search_node)
    workflow.set_entry_point("search")
    workflow.add_edge("search", END)
    
    return workflow.compile()


# ─────────────────────────────────────────────────────────────────
# SPECIALIZED SUB-AGENT: SUMMARIZATION AGENT
# ─────────────────────────────────────────────────────────────────

def create_summarizer_agent():
    """
    Specialized agent for text summarization and synthesis.
    Optimized for condensing long content into concise summaries.
    """
    
    summary_system_prompt = """
You are a Summarization Specialist Agent. Your role is to:
1. Condense long text into clear, concise summaries (100-200 words)
2. Preserve key information and main points
3. Maintain logical flow and coherence
4. Remove redundancy while keeping essential details

Focus on clarity and brevity. Do not add new information.
"""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1500
    )
    
    def summarize_node(state: SubAgentState):
        task = state.get("task_description", "")
        messages = [
            SystemMessage(content=summary_system_prompt),
            HumanMessage(content=f"Summarize the following: {task}")
        ]
        response = llm.invoke(messages)
        return {"result": response.content, "messages": [response]}
    
    workflow = StateGraph(SubAgentState)
    workflow.add_node("summarize", summarize_node)
    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", END)
    
    return workflow.compile()


# ─────────────────────────────────────────────────────────────────
# SPECIALIZED SUB-AGENT: ANALYSIS AGENT
# ─────────────────────────────────────────────────────────────────

def create_analysis_agent():
    """
    Specialized agent for comparative analysis and reasoning.
    Optimized for comparing multiple sources and drawing insights.
    """
    
    analysis_system_prompt = """
You are an Analysis Specialist Agent. Your role is to:
1. Compare multiple pieces of information
2. Identify patterns, similarities, and differences
3. Draw meaningful insights and conclusions
4. Structure findings logically (200-300 words)

Focus on depth of analysis and clear reasoning. Support claims with evidence.
"""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=2500
    )
    
    def analyze_node(state: SubAgentState):
        task = state.get("task_description", "")
        messages = [
            SystemMessage(content=analysis_system_prompt),
            HumanMessage(content=f"Analyze and compare: {task}")
        ]
        response = llm.invoke(messages)
        return {"result": response.content, "messages": [response]}
    
    workflow = StateGraph(SubAgentState)
    workflow.add_node("analyze", analyze_node)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()


# ─────────────────────────────────────────────────────────────────
# AGENT REGISTRY
# ─────────────────────────────────────────────────────────────────

class SubAgentRegistry:
    """
    Registry for managing and invoking specialized sub-agents.
    """
    
    def __init__(self):
        self.agents = {
            "search": create_search_agent(),
            "summarizer": create_summarizer_agent(),
            "analysis": create_analysis_agent()
        }
        self.descriptions = {
            "search": "Web search and information gathering specialist",
            "summarizer": "Text summarization and condensation specialist",
            "analysis": "Comparative analysis and reasoning specialist"
        }
    
    def get_agent(self, agent_type: str):
        """Get a sub-agent by type."""
        return self.agents.get(agent_type)
    
    def get_available_agents(self) -> List[Dict[str, str]]:
        """Return list of available sub-agents with descriptions."""
        return [
            {"type": agent_type, "description": desc}
            for agent_type, desc in self.descriptions.items()
        ]
    
    def invoke(self, agent_type: str, task: str, delegate_id: str = "default") -> str:
        """
        Invoke a sub-agent with a specific task.
        Returns the result string.
        """
        agent = self.get_agent(agent_type)
        if not agent:
            return f"Error: Unknown agent type '{agent_type}'"
        
        try:
            result = agent.invoke({
                "messages": [],
                "task_description": task,
                "result": "",
                "delegate_id": delegate_id
            })
            return result.get("result", "No result returned")
        except Exception as e:
            return f"Error executing {agent_type} agent: {str(e)}"


# Global registry instance
sub_agent_registry = SubAgentRegistry()
