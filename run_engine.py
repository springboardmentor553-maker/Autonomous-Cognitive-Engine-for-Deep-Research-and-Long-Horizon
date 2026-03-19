import subprocess
import os

print("\n🚀 STARTING AUTONOMOUS COGNITIVE ENGINE\n")

print("🔹 Running Milestone 1\n")
subprocess.run(["python", "milestone1/simple_app.py"])

print("\n✅ Milestone 1 Completed\n")

print("🔹 Running Milestone 2\n")

os.chdir("milestone2/cognitive_engine")
subprocess.run(["python", "main.py"])

print("\n✅ Milestone 2 Completed\n")

print("\n🏁 ENGINE FINISHED\n")