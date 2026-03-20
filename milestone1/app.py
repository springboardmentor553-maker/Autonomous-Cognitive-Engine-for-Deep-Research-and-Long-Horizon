import json
import sys
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from tools.planning.write_todos import write_todos_tool

def create_planning_agent():
    llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0, groq_api_key='your_api_key_here'
    system_prompt = 'You are a planning assistant. You MUST use write_todos_tool for every task.'
    return create_react_agent(llm, tools=[write_todos_tool], state_modifier=system_prompt)

def run_agent(agent, input_text):
    print(f'\n🤖 Processing: {input_text}')
    try:
        return agent.invoke({'messages': [('user', input_text)]})
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == '__main__':
    print('🚀 --- STARTING ENGINE ---')
    agent = create_planning_agent()
    tasks = ['Task 1', 'Task 2', 'Task 3']
    for i, t in enumerate(tasks, 1):
        print(f'[PROGRESS {i}/3]')
        run_agent(agent, t)
    print('\n🏁 DONE')