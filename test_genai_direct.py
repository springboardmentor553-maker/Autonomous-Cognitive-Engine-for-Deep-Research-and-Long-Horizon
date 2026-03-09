import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("No JAVA_HOME... no wait, GOOGLE_API_KEY found.")
    exit(1)

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content("Hello")
    print("Generate content successful")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
