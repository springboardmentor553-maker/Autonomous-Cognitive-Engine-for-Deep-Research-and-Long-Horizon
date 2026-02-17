import streamlit as st
from backend.core.graph_builder import run_agent
st.title("Autonomous Cognitive Engine")
goal = st.text_input("Enter a complex goal:")
if st.button("Run Agent"):
    state = run_agent(goal)
    st.write("### Completed Tasks")
    st.write(state.completed)
    st.write("### Memory")
    st.write(state.memory)