# project/deep_cognitive_agent/tools/filesystem/storage.py

class VirtualFileSystem:
    def __init__(self):
        # Stores {"filename.txt": "content"}
        self.files = {}

    def write(self, filename: str, content: str):
        self.files[filename] = content
        return f"File '{filename}' written successfully."

    def read(self, filename: str):
        return self.files.get(filename, f"File '{filename}' not found.")

    def ls(self):
        return list(self.files.keys())

# Global instance to persist during the agent's run
vfs = VirtualFileSystem()