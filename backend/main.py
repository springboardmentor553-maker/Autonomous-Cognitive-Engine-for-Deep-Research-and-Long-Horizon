from backend.core.executor import run_agent
from dotenv import load_dotenv
import os

# Load environment variables (.env)
load_dotenv(dotenv_path=".env")

print("API Loaded:", os.getenv("GOOGLE_API_KEY") is not None)

def main():
    print("🚀 Autonomous Cognitive Engine Started...\n")

    while True:
        query = input("Enter query (or 'exit'): ")

        if query.lower() == "exit":
            print("👋 Exiting... Goodbye!")
            break

        choice = input("Choose output (summary/detailed): ").strip().lower()

        if choice not in ["summary", "detailed"]:
            print("⚠️ Invalid choice. Defaulting to detailed.\n")
            choice = "detailed"

        print("\n⏳ Processing your request...\n")

        try:
            result = run_agent(query,output_format=choice)

            print("\n" + "="*60)
            print("📊 FINAL OUTPUT")
            print("="*60)
            print(result)

            # ✅ Save output to file
            with open("output.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\nQUERY: {query}\n")
                f.write(f"OUTPUT:\n{result}\n")
                f.write("="*60 + "\n")

        except Exception as e:
            print("❌ Error occurred:", str(e))


if __name__ == "__main__":
    main()