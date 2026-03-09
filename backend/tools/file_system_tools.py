import os

BASE_DIR = "memory"


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def write_file(path, content):

    full_path = os.path.join(BASE_DIR, path)

    ensure_directory(os.path.dirname(full_path))

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Stored research in {full_path}"


def read_file(path):

    full_path = os.path.join(BASE_DIR, path)

    if not os.path.exists(full_path):
        return "File not found"

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()
