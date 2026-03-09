from .write_file import write_file
from .read_file import read_file, set_vfs_reference as set_read_vfs
from .edit_file import edit_file
from .ls import ls, set_vfs_reference as set_ls_vfs


def bind_vfs(vfs: dict) -> None:
    """Bind the live VFS dict to tools that need to read from it."""
    set_read_vfs(vfs)
    set_ls_vfs(vfs)
