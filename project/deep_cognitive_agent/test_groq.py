import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
key = os.getenv("GROQ_API_KEY")

print(f"Checking Key: {key[:6]}..." if key else "? No Key Found in .env")

try:
    llm = ChatGroq(model="llama-3.1-8b-instant")
    response = llm.invoke("Hello, are you there?")
    print("? Groq Connection: SUCCESS")
    print(f"?? AI Response: {response.content}")
except Exception as e:
    print(f"? Groq Connection: FAILED\nError: {e}")
