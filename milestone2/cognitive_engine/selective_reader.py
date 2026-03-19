from tools.file_tools import read_file


def load_required_files(state, files_needed):

    combined_content = ""

    for file in files_needed:
        combined_content += read_file(state, file) + "\n"

    return combined_content