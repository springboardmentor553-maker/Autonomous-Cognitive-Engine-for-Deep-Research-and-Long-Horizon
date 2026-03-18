def summarizer_agent(input_data):

    print("SUBAGENT → Summarizer Agent processing")

    # -------------------------------
    # ANALYSIS MODE
    # -------------------------------
    if "APPLICATION:" in input_data:

        if "Imaging" in input_data:
            return """
Analysis:

Key Insights:
- AI improves medical image diagnosis accuracy

Important Findings:
- Works well with MRI and CT scan data

Recommendations:
- Improve dataset quality for better results
"""

        elif "Chatbots" in input_data:
            return """
Analysis:

Key Insights:
- Chatbots reduce hospital workload

Important Findings:
- Useful for basic queries, not complex diagnosis

Recommendations:
- Improve NLP understanding
"""

        elif "Predictive" in input_data:
            return """
Analysis:

Key Insights:
- AI helps predict diseases early

Important Findings:
- Requires large datasets for accuracy

Recommendations:
- Improve data collection and privacy
"""

        elif "Telemedicine" in input_data:
            return """
Analysis:

Key Insights:
- Enables remote healthcare access

Important Findings:
- Depends on internet connectivity

Recommendations:
- Improve internet infrastructure
- Enhance remote diagnostic tools
"""

        elif "Clinical" in input_data or "Decision" in input_data:
            return """
Analysis:

Key Insights:
- AI supports doctors in treatment decisions

Important Findings:
- Reduces medical errors

Recommendations:
- Increase trust and adoption among doctors
"""

        else:
            return """
Analysis:

Key Insights:
- AI improves healthcare efficiency

Important Findings:
- Depends on structured medical data

Recommendations:
- Improve data handling and privacy
"""

    # -------------------------------
    # SUMMARY MODE
    # -------------------------------
    else:
        data = input_data.lower()

        # FINAL SUMMARY
        if "FINAL" in input_data:
            return """Final Summary:
- Medical imaging improves diagnosis accuracy
- Decision systems assist doctors in treatment planning
- Chatbots provide continuous patient interaction
- Predictive analytics enables early disease detection
- Telemedicine increases healthcare accessibility

Overall, AI is transforming healthcare by improving efficiency, accuracy, and accessibility.
"""
    
    
        elif "x-ray" in data or "mri" in data or "imaging" in data:
            return """Summary:
- AI helps detect diseases using X-rays and scans
- Improves diagnostic accuracy
"""

        elif "chatbot" in data:
            return """Summary:
- AI chatbots provide 24/7 patient support
- Reduce hospital workload
"""

        elif "predict" in data:
            return """Summary:
- AI predicts diseases early
- Helps in preventive healthcare
- Reduces treatment cost
"""

        elif "telemedicine" in data or "remote" in data:
            return """Summary:
- AI enables remote consultations
- Improves access to healthcare in rural areas
- Saves time and cost for patients
"""
        elif "clinical" in data or "decision" in data:
            return """Summary:
- AI assists doctors in treatment decisions
- Reduces medical errors
"""

        else:
            return """Summary:
- AI improves overall healthcare efficiency
"""