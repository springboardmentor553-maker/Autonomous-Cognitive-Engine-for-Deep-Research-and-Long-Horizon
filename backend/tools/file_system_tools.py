import os
def save_to_file(filename: str, content: str):
    with open(filename, "w") as f:
        f.write(content)
def read_file(filename: str):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.read()
    return "File not found."