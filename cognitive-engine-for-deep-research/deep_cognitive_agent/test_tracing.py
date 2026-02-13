import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Simple test: Send a message and get a response
response = model.invoke([HumanMessage(content="Hello, how are you?")])

print("Response:", response.content)
print("Tracing test completed. Check LangSmith dashboard for traces.")
