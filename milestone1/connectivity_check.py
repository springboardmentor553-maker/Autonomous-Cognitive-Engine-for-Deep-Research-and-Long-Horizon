import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

print("--- 1. Testing Environment Loading ---")
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: No API Key found in .env file.")
else:
    print(f"✅ API Key found (starts with: {api_key[:6]}...)")

print("\n--- 2. Testing Groq Connection ---")
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    response = llm.invoke("Say 'Connection Successful'")
    print(f"✅ Response from Groq: {response.content}")
except Exception as e:
    print(f"❌ Connection Failed: {e}")