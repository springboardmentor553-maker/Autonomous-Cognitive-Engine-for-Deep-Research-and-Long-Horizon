# Simple in-memory file system

file_system = {}


def write_file(filename, content):
    file_system[filename] = content


def read_file(filename):
    return file_system.get(filename, "")


def ls():
    return list(file_system.keys())