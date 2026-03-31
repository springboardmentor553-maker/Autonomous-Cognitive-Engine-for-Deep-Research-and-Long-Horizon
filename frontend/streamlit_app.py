import streamlit as st
import sys
import os
import time
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# PATH FIX
# =========================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from backend.core.executor import run_agent

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Autonomous AI Agent", layout="wide")

# =========================
# LIGHT UI FIX (NO DARK THEME)
# =========================
st.markdown("""
<style>
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px 15px;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #E8F0FE;
    color: black;
}

[data-testid="stChatMessage"][data-testid*="assistant"] {
    background-color: #FFFFFF;
    color: black;
    border: 1px solid #E5E7EB;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🧠 Autonomous Cognitive Engine")
st.write("⚡ AI-powered research assistant")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.markdown("## 🎛️ Choose Output Mode")

    mode = st.selectbox(
        "",
        ["summary", "detailed"],
        index=1  # default = detailed
    )

    st.markdown("---")

    st.markdown("## ⚙️ Settings")

    st.markdown("**Summary →** Fast + Low API usage")
    st.markdown("**Detailed →** Full research pipeline")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# SHOW CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# SMART CLEAN FUNCTION
# =========================
def clean_and_format(text):
    text = text.replace("**", "")
    text = text.replace("***", "")
    text = text.replace("..", ".")

    lines = []
    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Add spacing to bullets
        if line.startswith("•"):
            lines.append("\n" + line)
        else:
            lines.append(line)

    return "\n\n".join(lines)

# =========================
# SMART SPLIT FUNCTION
# =========================
def split_sections(text):

    overview = ""
    key_points = []
    challenges = []
    future = []
    takeaway = ""

    current = None

    for line in text.split("\n"):
        l = line.lower().strip()

        if "overview" in l:
            current = "overview"
            continue
        elif "key points" in l:
            current = "key"
            continue
        elif "challenges" in l:
            current = "challenges"
            continue
        elif "future scope" in l:
            current = "future"
            continue
        elif "final takeaway" in l or "summary" in l:
            current = "takeaway"
            continue

        if not line.strip():
            continue

        if current == "overview":
            overview += line + " "

        elif current == "key":
            key_points.append(line)

        elif current == "challenges":
            challenges.append(line)

        elif current == "future":
            future.append(line)

        elif current == "takeaway":
            takeaway += line + " "

    # Construct sections
    research = f"{overview}\n\n" + "\n".join(key_points)
    analysis = "\n".join(challenges + future)
    summary = takeaway

    return research.strip(), analysis.strip(), summary.strip()


# =========================
# PDF FUNCTION
# =========================
def create_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    story = []
    for line in text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================
# INPUT
# =========================
user_input = st.chat_input("Enter your query ")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("⏳ Thinking..."):
            try:
                result = run_agent(user_input, output_format=mode)
            except Exception as e:
                result = f"Error: {str(e)}"

        result = clean_and_format(result)
        research, analysis, summary = split_sections(result)

        # =========================
        # OUTPUT UI (CLEAN)
        # =========================
        # =========================
        # HEADER
        # =========================
        st.markdown(f"### 🔎 Query: {user_input}")

        # =========================
        # TABS (SIDE-BY-SIDE)
        # =========================
        tab1, tab2, tab3 = st.tabs(["📌 Research", "🧠 Analysis", "📊 Summary"])

        with tab1:
            st.markdown(research if research else "No data")

        with tab2:
            st.markdown(analysis if analysis else "No major challenges identified.")

        with tab3:
            st.markdown(summary if summary else "Summary not available.")

        # =========================
        # DETAILED VIEW
        # =========================
        st.markdown("## 📌 Detailed Research")
        st.markdown(result)

        final_text = result

        # =========================
        # PREVIEW FIRST (FIXED ORDER)
        # =========================
        with st.expander("📝 Preview Export Content"):
            st.code(final_text)

        # =========================
        # EXPORT OPTIONS AFTER PREVIEW
        # =========================
        st.markdown("## 📥 Export Options")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button("📄 Download TXT", final_text, "report.txt")

        with col2:
            pdf_file = create_pdf(final_text)
            st.download_button("📑 Download PDF", pdf_file, "report.pdf")

        # =========================
        # METRICS
        # =========================
        st.markdown("## 📊 Output Metrics")

        word_count = len(final_text.split())
        char_count = len(final_text)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Words", word_count)

        with col2:
            st.metric("Characters", char_count)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })