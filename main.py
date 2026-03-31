import dotenv
import os
import re
# Load environment variables before importing modules that depend on them
dotenv.load_dotenv()

from src.agent import app
from langchain_core.messages import HumanMessage

def sanitize_filename(text):
    """Clean string to be safe for filenames."""
    # Keep alphanumeric characters and replace spaces with underscores
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    clean = re.sub(r'\s+', '_', clean.strip())
    # Limit to 30 characters
    return clean[:30]

def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found. Please set it in .env file.")
        return

    print("Agent initialized. specific task? (or press Enter for default 'Plan a research report on AI Agents')")
    user_input = input("> ").strip()
    if not user_input:
        user_input = "Plan a research report on AI Agents"
        
    print(f"\nProcessing request: {user_input}...\n")
    
    # Add constraint to prevent model repetition loops
    guided_input = f"{user_input} Max 5 tasks."
    initial_state = {"messages": [HumanMessage(content=guided_input)], "todos": []}
    
    # Run the graph
    try:
        final_state = app.invoke(initial_state)
        
        todos = final_state.get("todos", [])
        if todos:
            # save todos to file
            todo_dir = "todo"
            if not os.path.exists(todo_dir):
                os.makedirs(todo_dir)
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_prefix = sanitize_filename(user_input)
            filename = os.path.join(todo_dir, f"TODOS_{prompt_prefix}_{timestamp}.md")
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Prompt: {user_input}\n\n")
                f.write("## Tasks\n\n")
                for i, todo in enumerate(todos, 1):
                    f.write(f"{i}. {todo}\n")
            
            print(f"TODOs generated successfully and saved to: {filename}")
            
            # Optionally print the file contents instead of everything
            print("\nOutput:")
            for i, todo in enumerate(todos, 1):
                print(f"{i}. {todo}")
        else:
            print("No TODOs generated.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
