"""
Reviewer Agent - With Mock Mode (No API Key Required)
"""
import os
from brains.filetools import write_file, read_file

# Check if mock mode
USE_MOCK_MODE = os.getenv('ANTHROPIC_API_KEY') is None or os.getenv('USE_MOCK_MODE', 'false').lower() == 'true'

if not USE_MOCK_MODE:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

def create_reviewer():
    """Create reviewer agent with mock mode support"""
    
    if USE_MOCK_MODE:
        print("⚠️ REVIEWER: Running in MOCK MODE (no API calls)")
    
    def reviewer_node(state):
        """Reviewer agent node"""
        
        print(f"[REVIEWER] Reviewing final report")
        
        # Read final report
        try:
            report = read_file("final_report.txt")
        except:
            report = "No report found"
        
        if USE_MOCK_MODE:
            # Generate mock review
            content = f"""QUALITY ASSURANCE REVIEW
========================

Review Date: 2026-04-01
Reviewer: Automated QA System (SIMULATION MODE)
Status: DEMONSTRATION REVIEW

========================

REPORT ASSESSMENT:

1. COMPLETENESS: ✅ PASS
   The report covers all required sections and addresses the core research question. All three research phases have been incorporated into the final deliverable.

2. STRUCTURE: ✅ PASS
   The document follows a logical flow from introduction through findings to recommendations. Section organization is clear and facilitates reader comprehension.

3. CONTENT QUALITY: ⚠️ SIMULATION MODE
   Note: This review is based on simulated content. In production mode with API access, content would undergo rigorous quality validation including:
   - Factual accuracy verification
   - Citation validation
   - Clarity and coherence assessment
   - Professional tone evaluation

4. RECOMMENDATIONS CLARITY: ✅ PASS
   Action items are specific and organized by timeframe. Implementation guidance is practical and actionable.

5. FORMATTING: ✅ PASS
   Document structure is professional and presentation is clean. Headers and sections are well-organized.

STRENGTHS:
----------
• Comprehensive coverage of topic
• Clear structure and organization
• Actionable recommendations
• Professional presentation

AREAS FOR ENHANCEMENT:
---------------------
• Add real data when API key is configured
• Include specific citations and references
• Incorporate quantitative metrics where applicable
• Add visual elements (charts/graphs) for key data points

FINAL VERDICT:
-------------
STATUS: ✅ APPROVED FOR SIMULATION
CONFIDENCE: DEMONSTRATION MODE

This report meets the structural and organizational requirements for a professional deliverable. To generate production-quality content with real research and analysis, please configure your ANTHROPIC_API_KEY in the .env file.

NEXT STEPS:
-----------
1. Configure API key for production use
2. Re-run analysis with real LLM-powered research
3. Validate all data points and citations
4. Add stakeholder-specific customizations

========================
NOTE: This is a SIMULATED review for demonstration.
Real reviews require API access for detailed analysis.
========================

Reviewer Signature: QA System (Mock Mode)
Review Complete: Yes
"""
        else:
            # Real API mode
            llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.3,
                max_tokens=1000
            )
            
            system_message = """You are a quality reviewer. Review the final report and provide feedback.

Check for:
1. Completeness - does it cover the topic well?
2. Accuracy - is the information correct?
3. Clarity - is it well-written?
4. Structure - is it organized logically?

Provide a brief review (200-300 words) with:
- What's good
- Any improvements needed
- Final verdict (Approved/Needs Revision)"""

            prompt = f"""Review this report:

{report}

Provide your quality review."""
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            content = response.content
        
        # Write review
        filename = "review.txt"
        write_file(filename, content)
        
        print(f"[REVIEWER] Created: {filename}")
        
        # Update state
        created_files = state.get("created_files", [])
        created_files.append(filename)
        
        return {
            **state,
            "created_files": created_files,
            "messages": state.get("messages", [])
        }
    
    return reviewer_node
