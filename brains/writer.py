"""
Writer Agent - With Mock Mode (No API Key Required)
"""
import os
from brains.filetools import write_file, read_file, list_files

# Check if mock mode
USE_MOCK_MODE = os.getenv('ANTHROPIC_API_KEY') is None or os.getenv('USE_MOCK_MODE', 'false').lower() == 'true'

if not USE_MOCK_MODE:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

def create_writer():
    """Create writer agent with mock mode support"""
    
    if USE_MOCK_MODE:
        print("⚠️ WRITER: Running in MOCK MODE (no API calls)")
    
    def writer_node(state):
        """Writer agent node"""
        user_task = state.get("user_task", "Write report")
        
        print(f"[WRITER] Creating final report")
        
        if USE_MOCK_MODE:
            # Generate mock report
            content = f"""COMPREHENSIVE RESEARCH REPORT
============================================

Title: {user_task}
Date: 2026-04-01
Status: SIMULATION MODE (No API Key)

============================================

EXECUTIVE SUMMARY
-----------------
This report presents a comprehensive analysis of "{user_task}". The findings are based on simulated research data generated for demonstration purposes. In production mode with an API key, this would contain real, detailed analysis.

INTRODUCTION
------------
The purpose of this report is to provide stakeholders with actionable insights regarding the subject matter. This analysis synthesizes information from multiple research phases to present a cohesive understanding of the topic.

METHODOLOGY
-----------
This report integrates findings from three research phases:
- Phase 1: Background research and context establishment
- Phase 2: Detailed analysis and data collection
- Phase 3: Synthesis and trend identification

FINDINGS
--------

1. Primary Insights
   The research reveals several key factors that influence outcomes in this domain. Analysis indicates that systematic approaches yield the most consistent results. Organizations implementing best practices demonstrate measurable improvements in performance metrics.

2. Supporting Evidence
   Data gathered across multiple sources confirms that strategic planning combined with tactical execution creates optimal conditions for success. Stakeholder engagement emerges as a critical success factor.

3. Comparative Analysis
   When compared to alternative approaches, the recommended methodology shows superior outcomes in efficiency, scalability, and sustainability. Cost-benefit analysis supports investment in comprehensive solutions.

4. Implementation Considerations
   Successful deployment requires attention to organizational readiness, resource allocation, and change management. Phased rollout strategies minimize risk while maximizing adoption.

RECOMMENDATIONS
---------------

Based on the research findings, the following recommendations are proposed:

1. SHORT-TERM (0-6 months)
   - Conduct stakeholder assessment
   - Develop implementation roadmap
   - Allocate initial resources

2. MEDIUM-TERM (6-18 months)
   - Execute pilot programs
   - Gather performance metrics
   - Adjust strategies based on feedback

3. LONG-TERM (18+ months)
   - Scale successful initiatives
   - Establish continuous improvement processes
   - Monitor industry developments

CONCLUSION
----------
The analysis demonstrates that "{user_task}" represents a significant opportunity for value creation. Strategic investment in this area, coupled with systematic execution, positions organizations for sustained success. Continued monitoring and adaptation will ensure relevance as conditions evolve.

NEXT STEPS
----------
1. Review and validate findings with key stakeholders
2. Develop detailed action plans for recommended initiatives
3. Establish metrics for tracking progress and outcomes
4. Schedule regular review cycles to assess effectiveness

============================================
NOTE: This is SIMULATED content for demonstration purposes.
For real analysis, configure ANTHROPIC_API_KEY in .env file.
============================================

Report Length: ~600 words
Confidence: SIMULATION MODE
"""
        else:
            # Real API mode
            llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.7,
                max_tokens=2000
            )
            
            # Read all research files
            files = list_files()
            research_files = [f for f in files if f.startswith('research_')]
            
            research_content = ""
            for filename in research_files:
                content = read_file(filename)
                research_content += f"\n\n--- {filename} ---\n{content}"
            
            system_message = """You are a professional writer. Create comprehensive, well-structured reports.

Your report should:
1. Have a clear title and introduction
2. Organize findings into logical sections
3. Include all research gathered
4. Be 500-800 words
5. End with a conclusion

Write in professional, engaging prose."""

            prompt = f"""Task: {user_task}

Research gathered:
{research_content}

Create a comprehensive report that synthesizes all this research. 
Title: "{user_task}" Report
Length: 500-800 words
Format: Professional report with sections"""
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            content = response.content
        
        # Write final report
        filename = "final_report.txt"
        write_file(filename, content)
        
        print(f"[WRITER] Created: {filename}")
        
        # Update state
        created_files = state.get("created_files", [])
        created_files.append(filename)
        
        return {
            **state,
            "created_files": created_files,
            "final_output": content,
            "messages": state.get("messages", [])
        }
    
    return writer_node
