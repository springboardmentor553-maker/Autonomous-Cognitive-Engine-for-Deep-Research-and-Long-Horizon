from uuid import uuid4

import streamlit as st

from app.config import DEFAULT_THREAD_ID
from app.supervisor import Supervisor


st.set_page_config(page_title="Milestone 4 Agent", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f3ea 0%, #f4f8fb 100%);
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #13315c 0%, #0b6e4f 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 14px 35px rgba(19, 49, 92, 0.18);
    }
    .card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(19, 49, 92, 0.08);
        padding: 1rem 1.1rem;
        border-radius: 16px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }
    .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.6rem;
        color: #17324d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0 0 0.4rem 0;">Autonomous Research Agent</h1>
        <p style="margin:0; font-size:1rem;">
            LangGraph supervisor with planning, delegation, synthesis, evaluation, checkpointing, and optional live search.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 0.65], gap="large")

with left:
    st.markdown('<div class="section-title">User Request</div>', unsafe_allow_html=True)
    user_request = st.text_area(
        "Enter a task for the agent",
        height=190,
        label_visibility="collapsed",
        placeholder="Example: Research the role of AI in healthcare and generate a final report with recommendations.",
    )

with right:
    st.markdown(
        """
        <div class="card">
            <h3 style="margin-top:0;">Demo Tips</h3>
            <p style="margin-bottom:0;">
                Use prompts with words like research, analyze, compare, and summarize for stronger multi-step runs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run_clicked = st.button("Run Workflow", use_container_width=True, type="primary")

if run_clicked:
    if not user_request.strip():
        st.warning("Please enter a request.")
    else:
        with st.spinner("Running LangGraph workflow..."):
            thread_id = f"{DEFAULT_THREAD_ID}-{uuid4().hex[:8]}"
            result = Supervisor().run(user_request.strip(), thread_id=thread_id)

        eval_result = result["evaluation"]
        score = eval_result["score"]
        passed = eval_result["passed"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Evaluation Score", f"{score}/10")
        col2.metric("Pass Status", "Passed" if passed else "Needs Work")
        col3.metric("Files Created", len(result["files"]))

        tab1, tab2, tab3 = st.tabs(["Final Report", "Evaluation", "TODO Plan"])

        with tab1:
            st.markdown("### Final Report")
            st.write(result["final_report"])

        with tab2:
            st.markdown("### Evaluation Summary")
            st.write(eval_result["summary"])

            inner1, inner2 = st.columns(2)
            with inner1:
                st.markdown("#### Strengths")
                for item in eval_result["strengths"]:
                    st.write(f"- {item}")

                st.markdown("#### Weaknesses")
                for item in eval_result["weaknesses"]:
                    st.write(f"- {item}")

            with inner2:
                st.markdown("#### Improvements")
                for item in eval_result["improvements"]:
                    st.write(f"- {item}")

                with st.expander("Raw Evaluation JSON"):
                    st.json(eval_result)

        with tab3:
            st.markdown("### TODO Plan")
            st.dataframe(result["todos"], use_container_width=True)
