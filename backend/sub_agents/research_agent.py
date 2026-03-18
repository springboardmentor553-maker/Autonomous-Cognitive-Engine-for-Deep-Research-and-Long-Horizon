from backend.tools.file_system_tools import write_file


def research_agent(task_name):

    print(f"SUBAGENT → Research Agent processing {task_name}")

    if "Imaging" in task_name:
        return """
APPLICATION: Medical Imaging Analysis

Real-World Applications:
AI detects tumors in X-rays, MRIs, and CT scans.

Benefits:
- High diagnostic accuracy
- Faster detection

Limitations:
- Expensive equipment
- Requires labeled data
"""

    elif "Decision" in task_name:
        return """
APPLICATION: Clinical Decision Support Systems

Real-World Applications:
AI helps doctors choose treatments based on patient data.

Benefits:
- Better decisions
- Reduced medical errors

Limitations:
- Depends on data quality
- Trust issues among doctors
"""

    elif "Chatbots" in task_name:
        return """
APPLICATION: AI Chatbots in Healthcare

Real-World Applications:
Used for patient queries, appointment booking, symptom checking.

Benefits:
- 24/7 availability
- Reduces hospital workload

Limitations:
- Limited understanding
- Cannot replace doctors
"""

    elif "Predictive" in task_name:
        return """
APPLICATION: Predictive Analytics

Real-World Applications:
Predicts disease outbreaks and patient risks.

Benefits:
- Early intervention
- Cost reduction

Limitations:
- Needs large datasets
- Privacy concerns
"""

    elif "Telemedicine" in task_name:
        return """
APPLICATION: Telemedicine Platforms

Real-World Applications:
Remote consultations using AI-based diagnostics.

Benefits:
- Accessible healthcare
- Saves time

Limitations:
- Internet dependency
- Less physical examination
"""

    else:
        return f"General research about {task_name}"