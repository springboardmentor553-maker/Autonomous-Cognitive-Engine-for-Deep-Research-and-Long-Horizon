"""
File System Tools for Context Offloading - Milestone 2
Implements selective retrieval, meaningful naming, and edit operations
"""
from langchain_core.tools import tool
from pathlib import Path
import json

# Virtual file system directory
FILE_SYSTEM_DIR = Path("virtual_fs")
FILE_SYSTEM_DIR.mkdir(exist_ok=True)


def _write_file_impl(filename: str, content: str) -> str:
    """Internal implementation of write_file."""
    filepath = FILE_SYSTEM_DIR / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = len(content)
        word_count = len(content.split())
        
        return json.dumps({
            "status": "success",
            "message": f"Wrote {word_count} words ({file_size} chars) to {filename}",
            "filepath": str(filepath),
            "filename": filename,
            "size_bytes": file_size,
            "word_count": word_count
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to write file: {str(e)}",
            "filename": filename
        })


@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the virtual file system.
    Use for storing SUMMARIES and PROCESSED data, NOT raw input.
    
    Args:
        filename: Meaningful filename (e.g., "climate_causes_summary.txt")
        content: PROCESSED content (summary/analysis, not raw data)
        
    Returns:
        JSON confirmation with file metadata
    """
    return _write_file_impl(filename, content)


def _read_file_impl(filename: str) -> str:
    """Internal implementation of read_file."""
    filepath = FILE_SYSTEM_DIR / filename
    
    try:
        if not filepath.exists():
            return json.dumps({
                "status": "error",
                "message": f"File not found: {filename}",
                "filename": filename
            })
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        word_count = len(content.split())
        
        return json.dumps({
            "status": "success",
            "message": f"Read {word_count} words from {filename}",
            "filename": filename,
            "content": content,
            "word_count": word_count
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to read file: {str(e)}",
            "filename": filename
        })


@tool
def read_file(filename: str) -> str:
    """
    Read content from a file - SELECTIVE RETRIEVAL.
    Only load files you actually need for the current step.
    
    Args:
        filename: Specific file to read (e.g., "climate_causes_summary.txt")
        
    Returns:
        JSON with file content and metadata
    """
    return _read_file_impl(filename)


def _edit_file_impl(filename: str, search_text: str, replace_text: str) -> str:
    """Internal implementation of edit_file."""
    filepath = FILE_SYSTEM_DIR / filename
    
    try:
        if not filepath.exists():
            return json.dumps({
                "status": "error",
                "message": f"File not found: {filename}",
                "filename": filename
            })
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if search_text not in content:
            return json.dumps({
                "status": "error",
                "message": f"Search text not found in {filename}",
                "filename": filename,
                "search_text": search_text[:50]
            })
        
        new_content = content.replace(search_text, replace_text)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return json.dumps({
            "status": "success",
            "message": f"Edited {filename}: replaced {len(search_text)} chars with {len(replace_text)} chars",
            "filename": filename,
            "old_length": len(content),
            "new_length": len(new_content)
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to edit file: {str(e)}",
            "filename": filename
        })


@tool
def edit_file(filename: str, search_text: str, replace_text: str) -> str:
    """
    Edit existing file by replacing text - NO DUPLICATION.
    Use this to update/refine content without creating new files.
    
    Args:
        filename: File to edit
        search_text: Text to find and replace
        replace_text: New text to insert
        
    Returns:
        JSON confirmation of edit operation
    """
    return _edit_file_impl(filename, search_text, replace_text)


def _list_files_impl(pattern: str = None) -> str:
    """Internal implementation of list_files."""
    try:
        all_files = [f for f in FILE_SYSTEM_DIR.iterdir() if f.is_file()]
        
        if pattern:
            files = [f for f in all_files if pattern.lower() in f.name.lower()]
        else:
            files = all_files
        
        file_info = []
        for f in files:
            size = f.stat().st_size
            file_info.append({
                "filename": f.name,
                "size_bytes": size,
                "size_kb": round(size / 1024, 2)
            })
        
        return json.dumps({
            "status": "success",
            "message": f"Found {len(file_info)} files" + (f" matching '{pattern}'" if pattern else ""),
            "total_files": len(all_files),
            "matched_files": len(file_info),
            "files": file_info
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to list files: {str(e)}"
        })


@tool
def list_files(pattern: str = None) -> str:
    """
    List files in virtual FS - supports filtering by pattern.
    
    Args:
        pattern: Optional filter (e.g., "summary" to find all summary files)
        
    Returns:
        JSON list of matching files with metadata
    """
    return _list_files_impl(pattern)


def clear_virtual_fs():
    """Clear all files from virtual file system (for testing)."""
    if FILE_SYSTEM_DIR.exists():
        for file in FILE_SYSTEM_DIR.iterdir():
            if file.is_file():
                file.unlink()


def get_fs_stats():
    """Get file system statistics for evaluation."""
    if not FILE_SYSTEM_DIR.exists():
        return {
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_kb": 0.0,
            "files": []
        }
    
    files = list(FILE_SYSTEM_DIR.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    return {
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_kb": round(total_size / 1024, 2),
        "files": [f.name for f in files]
    }
