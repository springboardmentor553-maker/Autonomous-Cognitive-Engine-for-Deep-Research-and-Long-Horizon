import subprocess
import os

print("\n🚀 AUTONOMOUS COGNITIVE ENGINE (MERGED TEST)\n")

print("----- PHASE 1: AGENT EXECUTION -----\n")

subprocess.run(["python", "../milestone1/simple_app.py"])


print("\n----- PHASE 2: MEMORY SYSTEM -----\n")

os.chdir("../milestone2/cognitive_engine")

subprocess.run(["python", "main.py"])


print("\n🏁 MERGED ENGINE TEST COMPLETE\n")