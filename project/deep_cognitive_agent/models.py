import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    exit()

genai.configure(api_key=api_key)

print(f"Checking models for key: {api_key[:5]}...")

try:
    print("\n✅ AVAILABLE MODELS:")
    found = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
            found = True
    
    if not found:
        print("⚠️ No models found. Your API key might be invalid or has no access.")
        
except Exception as e:
    print(f"\n❌ CONNECTION ERROR: {e}")