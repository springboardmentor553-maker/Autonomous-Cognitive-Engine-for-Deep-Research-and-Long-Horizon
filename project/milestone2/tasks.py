"""
Milestone 2 Task Definitions - direct imperative commands for the agent to execute.
"""

TASKS = [
    {
        "id": "task_01",
        "label": "Climate Change: 3-Doc Summary + Final Synthesis",
        "description": (
            "Execute this task using your tools:\n\n"
            "You have 3 climate change paragraphs. Summarize each and store summaries, then combine.\n\n"
            "Paragraph 1: Rising global temperatures caused by greenhouse gas emissions are melting polar ice caps. Sea levels could rise over a meter by 2100, threatening coastal cities.\n\n"
            "Paragraph 2: Extreme weather events like hurricanes, droughts, and wildfires are becoming more frequent. Agricultural systems face stress causing food insecurity in Africa and South Asia.\n\n"
            "Paragraph 3: The Paris Accord aims to limit warming to 1.5C. Renewable energy adoption, carbon pricing, and reforestation are key policy tools deployed globally.\n\n"
            "Use tools in this order:\n"
            "1. write_todos with this task\n"
            "2. write_file('para1_summary.txt|<summary of paragraph 1>')\n"
            "3. write_file('para2_summary.txt|<summary of paragraph 2>')\n"
            "4. write_file('para3_summary.txt|<summary of paragraph 3>')\n"
            "5. ls('') to verify files\n"
            "6. read_file('para1_summary.txt') then read_file('para2_summary.txt') then read_file('para3_summary.txt')\n"
            "7. write_file('final_climate_summary.txt|<combined summary of all 3>')"
        ),
    },
    {
        "id": "task_02",
        "label": "Renewable Energy: 5 Policy Docs -> Differences + Framework",
        "description": (
            "Execute this task using your tools:\n\n"
            "Summarize 5 renewable energy policies, identify differences, propose a framework.\n\n"
            "USA: Tax credits, market-driven solar and wind adoption.\n"
            "Germany: 80% renewable by 2030, offshore wind, grid modernization.\n"
            "India: 500 GW target, rural solar electrification, climate finance.\n"
            "China: State-led solar manufacturing and hydropower, coal phase-down.\n"
            "Brazil: Hydropower base, biofuels, offshore wind, mixed funding.\n\n"
            "Use tools:\n"
            "1. write_todos with this task\n"
            "2. write_file('doc1_summary.txt|<USA policy summary>')\n"
            "3. write_file('doc2_summary.txt|<Germany policy summary>')\n"
            "4. write_file('doc3_summary.txt|<India policy summary>')\n"
            "5. write_file('doc4_summary.txt|<China policy summary>')\n"
            "6. write_file('doc5_summary.txt|<Brazil policy summary>')\n"
            "7. ls('') to verify 5 files\n"
            "8. read_file each summary\n"
            "9. write_file('policy_differences.txt|<key differences>')\n"
            "10. write_file('consolidated_framework.txt|<improvement framework>')"
        ),
    },
    {
        "id": "task_03",
        "label": "Selective Policy Comparison: Germany vs India Only",
        "description": (
            "Execute this task using your tools:\n\n"
            "Store 5 country policy summaries, then compare ONLY Germany and India.\n\n"
            "Use tools:\n"
            "1. write_todos with this task\n"
            "2. write_file('USA.txt|USA: Tax incentives, market-driven solar and wind.')\n"
            "3. write_file('Germany.txt|Germany: 80% renewable by 2030, offshore wind, grid modernization.')\n"
            "4. write_file('India.txt|India: 500 GW renewable target, rural solar, climate finance.')\n"
            "5. write_file('China.txt|China: State-led solar manufacturing and hydropower.')\n"
            "6. write_file('Brazil.txt|Brazil: Hydropower base, biofuels, offshore wind.')\n"
            "7. ls('') to list files\n"
            "8. read_file('Germany.txt') — only Germany needed\n"
            "9. read_file('India.txt') — only India needed\n"
            "10. write_file('germany_india_comparison.txt|<comparison of Germany vs India only>')"
        ),
    },
    {
        "id": "task_04",
        "label": "Draft Refinement: write_file -> read_file -> edit_file",
        "description": (
            "Execute this task using your tools:\n\n"
            "Create a policy draft, read it, analyze gaps, then refine it with edit_file.\n\n"
            "Use tools:\n"
            "1. write_todos with this task\n"
            "2. write_file('draft.txt|Countries should invest in renewable energy to reduce carbon emissions. Solar and wind are most cost-effective.')\n"
            "3. read_file('draft.txt') to retrieve the draft\n"
            "4. write_file('analysis.txt|Draft gaps: missing battery storage, grid infrastructure, just transition for fossil fuel workers.')\n"
            "5. edit_file('draft.txt|Countries should invest in renewable energy focusing on solar, wind, and battery storage. Grid modernization and just transition policies for fossil fuel workers are essential.')\n"
            "6. read_file('draft.txt') to verify the update"
        ),
    },
    {
        "id": "task_05",
        "label": "AI Ethics: 4 Frameworks -> Unified Model -> Sustainability Refinement",
        "description": (
            "Execute this task using your tools:\n\n"
            "Summarize 4 AI ethics frameworks, compare, build unified model, then refine.\n\n"
            "Framework A (EU AI Act): Risk-based classification, human oversight, mandatory audits.\n"
            "Framework B (IEEE): Human well-being, accountability, transparency, engineer responsibility.\n"
            "Framework C (Google): Socially beneficial AI, bias avoidance, privacy by design, voluntary.\n"
            "Framework D (UNESCO): Human rights, digital inclusion, environmental sustainability, governance.\n\n"
            "Use tools:\n"
            "1. write_todos with this task\n"
            "2. write_file('A.txt|<EU AI Act summary>')\n"
            "3. write_file('B.txt|<IEEE summary>')\n"
            "4. write_file('C.txt|<Google summary>')\n"
            "5. write_file('D.txt|<UNESCO summary>')\n"
            "6. ls('') to verify 4 files\n"
            "7. read_file('A.txt'), read_file('B.txt'), read_file('C.txt'), read_file('D.txt')\n"
            "8. write_file('comparison.txt|<comparison of all 4>')\n"
            "9. read_file('comparison.txt')\n"
            "10. write_file('unified_model.txt|<unified AI ethics model>')\n"
            "11. read_file('unified_model.txt')\n"
            "12. edit_file('unified_model.txt|<unified model refined with sustainability considerations>')"
        ),
    },
    {
        "id": "task_06",
        "label": "Scaling Test: 4-Doc Cybersecurity Chain",
        "description": (
            "Execute this task using your tools:\n\n"
            "Summarize 4 cybersecurity frameworks, compare, build unified policy, refine.\n\n"
            "Doc A (NIST): Identify, Protect, Detect, Respond, Recover. Voluntary US framework.\n"
            "Doc B (ISO 27001): International ISMS standard. Risk assessment and controls.\n"
            "Doc C (EU NIS2): Mandatory for critical infrastructure. 24h incident reporting.\n"
            "Doc D (UK Cyber Essentials): 5 controls: firewalls, config, access, malware, patching.\n\n"
            "Use tools:\n"
            "1. write_todos with this task\n"
            "2. write_file('A_summary.txt|<NIST summary>')\n"
            "3. write_file('B_summary.txt|<ISO 27001 summary>')\n"
            "4. write_file('C_summary.txt|<NIS2 summary>')\n"
            "5. write_file('D_summary.txt|<Cyber Essentials summary>')\n"
            "6. ls('') to verify 4 files\n"
            "7. read_file each summary\n"
            "8. write_file('cyber_comparison.txt|<comparison of all 4>')\n"
            "9. read_file('cyber_comparison.txt')\n"
            "10. write_file('cyber_unified_policy.txt|<unified cybersecurity policy>')\n"
            "11. read_file('cyber_unified_policy.txt')\n"
            "12. edit_file('cyber_unified_policy.txt|<policy refined with implementation roadmap>')"
        ),
    },
    {
        "id": "task_07",
        "label": "Full Eval Pattern: write->write->ls->read->read->write->read->edit",
        "description": (
            "Execute this task using your tools in this exact sequence:\n\n"
            "Compare two AI governance approaches: Regulatory vs Self-Regulatory.\n\n"
            "Approach A (Regulatory): Government legislation, mandatory compliance, audits, penalties. Ensures accountability but may slow innovation.\n\n"
            "Approach B (Self-Regulatory): Industry voluntary codes and ethics boards. Flexible but no binding enforcement.\n\n"
            "Use tools in this exact order:\n"
            "1. write_todos with this task\n"
            "2. write_file('A_summary.txt|<summary of regulatory approach>')\n"
            "3. write_file('B_summary.txt|<summary of self-regulatory approach>')\n"
            "4. ls('') to verify both files\n"
            "5. read_file('A_summary.txt')\n"
            "6. read_file('B_summary.txt')\n"
            "7. write_file('comparison.txt|<comparison of A vs B>')\n"
            "8. read_file('comparison.txt')\n"
            "9. edit_file('comparison.txt|<refined comparison with final recommendation>')"
        ),
    },
]
