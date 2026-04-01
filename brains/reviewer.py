"""
Researcher Agent - With Mock Mode (No API Key Required)
"""
import os
from brains.filetools import write_file

# Check if mock mode
USE_MOCK_MODE = os.getenv('ANTHROPIC_API_KEY') is None or os.getenv('USE_MOCK_MODE', 'false').lower() == 'true'

if not USE_MOCK_MODE:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

def create_researcher():
    """Create researcher agent with mock mode support"""
    
    if USE_MOCK_MODE:
        print("⚠️ RESEARCHER: Running in MOCK MODE (no API calls)")
    
    def researcher_node(state):
        """Researcher agent node"""
        user_task = state.get("user_task", "Research topic")
        current_step = state.get("current_step", 1)
        
        print(f"[RESEARCHER] Processing step {current_step}")
        
        if USE_MOCK_MODE:
            # Generate mock research content
            content = f"""Research Report - Step {current_step}
            
Task: {user_task}

MOCK RESEARCH FINDINGS:

This is simulated research content generated without API calls. In a real scenario, this would contain:

1. **Background Information**
   The topic "{user_task}" is an important area of study that has gained significant attention in recent years. Research shows that understanding this subject requires examining multiple perspectives and considering various factors.

2. **Key Findings**
   - Primary Factor: Recent studies indicate substantial growth and development in this area
   - Secondary Consideration: Multiple stakeholders are involved in shaping outcomes
   - Emerging Trends: New approaches are being developed to address challenges
   - Data Points: Quantitative analysis suggests positive trajectory

3. **Current State Analysis**
   Current implementations demonstrate varying degrees of success. Best practices include systematic approaches, stakeholder engagement, and continuous evaluation. Organizations that adopt comprehensive strategies tend to achieve better results.

4. **Research Methodology**
   This analysis draws from multiple sources including academic journals, industry reports, and expert interviews. The triangulation of data provides robust insights into the subject matter.

5. **Implications**
   The findings suggest that continued attention to this area will yield benefits. Stakeholders should consider both short-term actions and long-term strategic planning.

Note: This is SIMULATED content for demonstration. Replace with real API calls when key is available.
---
Generated: Step {current_step} of 3
Word Count: ~300
"""
        else:
            # Real API mode
            llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.7,
                max_tokens=1500
            )
            
            system_message = """You are a research agent. Your job is to gather information and create research content.

For each research task:
1. Analyze the topic thoroughly
2. Generate detailed, factual content (300-400 words)
3. Create a research file with your findings

Keep your research focused and well-structured. Write in clear paragraphs."""

            prompt = f"""Research Task: {user_task}

Step {current_step} of 3: Provide detailed research findings.

Generate 300-400 words of well-researched content on this topic."""
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            content = response.content
        
        # Write research file
        filename = f"research_step{current_step}.txt"
        write_file(filename, content)
        
        print(f"[RESEARCHER] Created: {filename}")
        
        # Update state
        created_files = state.get("created_files", [])
        created_files.append(filename)
        
        return {
            **state,
            "created_files": created_files,
            "messages": state.get("messages", [])
        }
    
    return researcher_node
