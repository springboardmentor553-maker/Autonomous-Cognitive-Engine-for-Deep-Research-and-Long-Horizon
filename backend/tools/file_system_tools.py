file_store = {}


def write_file(filename, content):
    file_store[filename] = content
    print(f"WRITE FILE → {filename}")


def read_file(filename):
    return file_store.get(filename, "")


def list_files():
    return list(file_store.keys())