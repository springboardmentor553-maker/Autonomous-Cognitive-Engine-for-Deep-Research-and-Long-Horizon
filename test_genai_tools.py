import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def write_todos(todos: list[str]):
    """Create a list of TODOs."""
    return f"Created {len(todos)} todos."

try:
    tools = [write_todos]
    model = genai.GenerativeModel('gemini-1.5-pro', tools=tools)
    
    chat = model.start_chat()
    response = chat.send_message("Plan a study schedule for Python in 2 steps.")
    
    print("Response received")
    print(response.parts[0])
    
    # Check if function call exists
    if response.parts[0].function_call:
        fc = response.parts[0].function_call
        print(f"Function call: {fc.name}")
        print(f"Args: {fc.args}")

except Exception as e:
    import traceback
    traceback.print_exc()
