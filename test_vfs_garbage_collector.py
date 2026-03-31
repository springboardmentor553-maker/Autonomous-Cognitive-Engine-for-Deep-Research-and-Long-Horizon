import sys
sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
import random
import json
from typing import TypedDict, Dict

# ---------------------------------------------------------------------------------
# INNOVATION: Autonomous VFS Garbage Collection & Context Pruning
# ---------------------------------------------------------------------------------
# Why this is cutting-edge:
# Long-horizon agents inevitably hit LLM Context Windows and Context Degradation
# ("lost in the middle" problems) due to bloated Virtual File Systems filled with
# obsolete, hallucinated, or tangent data. 
# This Autonomous Garbage Collector (GC) Agent reads the system passively,
# evaluates Relevance/Decay metrics, and aggressively COMPRESSES or DELETES VFS 
# state files to keep the agent's memory lean, accurate, and cheap token-wise.
# ---------------------------------------------------------------------------------

class GCState(TypedDict):
    objective: str
    vfs: Dict[str, dict] # Added metadata to vfs files
    memory_usage_bytes: int
    gc_runs: int
    pruned_files: int

def write_vfs_file(state: GCState, filename: str, content: str, initial_relevance_guess: int):
    """Simulate a Sub-Agent dumping data into the File System."""
    print(f"   [SUB-AGENT] \u2795 Adding {filename} to VFS. ({len(content)} bytes)")
    
    # Store not just content, but GC metadata tracking
    state["vfs"][filename] = {
        "content": content,
        "size_bytes": len(content),
        "relevance_score": initial_relevance_guess, # 1-100
        "compressions": 0
    }
    state["memory_usage_bytes"] += len(content)
    return state

def run_vfs_garbage_collector(state: GCState):
    """
    The Active Pruning Agent.
    It evaluates all files against the core objective.
    Rules:
      - Score > 80: Perfect. Leave it alone.
      - Score 40-79: Compress it to preserve context window.
      - Score < 40: Tangent/Junk. Delete it completely.
    """
    print(f"\n============================================================")
    print(f"\u26a0\ufe0f MEMORY BLOAT DETECTED: {state['memory_usage_bytes']} Bytes.")
    print(f"\u2699\ufe0f INITIATING AUTONOMOUS VFS GARBAGE COLLECTION...")
    print(f"============================================================\n")
    
    state["gc_runs"] += 1
    files_to_delete = []
    
    for filename, metadata in list(state["vfs"].items()):
        # Simulate LLM evaluating the content against the Objective
        print(f"  [GC-AGENT] Evaluating relevance of '{filename}'...")
        score = metadata["relevance_score"]
        
        if score >= 80:
            print(f"      \u2192 Status: High Relevance ({score}%). Kept pristine.\n")
            continue
            
        elif score >= 40:
            # COMPRESS TO SAVE TOKENS
            original_size = metadata["size_bytes"]
            compressed_content = metadata["content"][:20] + "... [COMPRESSED SUMMARY]"
            new_size = len(compressed_content)
            
            # Update state
            state["vfs"][filename]["content"] = compressed_content
            state["vfs"][filename]["size_bytes"] = new_size
            state["vfs"][filename]["compressions"] += 1
            
            bytes_saved = original_size - new_size
            state["memory_usage_bytes"] -= bytes_saved
            print(f"      \u2192 Status: Bloated ({score}%). Compressible.")
            print(f"      \u21b3 Action: Aggressively summarized. \u2b07\ufe0f Freed {bytes_saved} bytes.\n")
            
        else:
            # DELETE HALLUCINATIONS / TANGENTS
            print(f"      \u2192 Status: Tangent/Hallucination ({score}%). Irrelevant to Objective.")
            print(f"      \u21b3 Action: FILE PERMANENTLY DELETED. \u2b07\ufe0f Freed {metadata['size_bytes']} bytes.\n")
            
            files_to_delete.append(filename)
            state["memory_usage_bytes"] -= metadata["size_bytes"]
            state["pruned_files"] += 1
            
    # Cleanup dictionary keys
    for d_file in files_to_delete:
        del state["vfs"][d_file]
        
    print(f"\n[GC-AGENT] Sweeping Complete. Memory Optimized \u2705")
    return state


if __name__ == "__main__":
    print(f"\n============================================================")
    print(f"INNOVATION: VFS GARBAGE COLLECTOR & ACTIVE MEMORY PRUNING")
    print(f"============================================================\n")
    
    initial_state: GCState = {
        "objective": "Build a secure AI-driven drug discovery pipeline.",
        "vfs": {},
        "memory_usage_bytes": 0,
        "gc_runs": 0,
        "pruned_files": 0
    }
    
    state = initial_state
    
    print(f"Objective: {state['objective']}\n")
    print("--- SIMULATING AGENTS GENERATING DATA OVER 100 HOURS ---")
    
    # 1. High Quality File
    content_1 = "Detailed biochemical bindings of Compound X. Strict security compliance mapped out for pipelines..." * 5
    state = write_vfs_file(state, "core_research.txt", content_1, initial_relevance_guess=95)
    
    # 2. Bloated File
    content_2 = "History of AI in the 1990s. Some good points on security, but mostly fluff and historical context that is irrelevant..." * 5
    state = write_vfs_file(state, "historical_context.txt", content_2, initial_relevance_guess=60)
    
    # 3. Complete Tangent / Hallucination
    content_3 = "The user might also want a pizza delivery app built into the drug discovery pipeline with CSS animations..." * 5
    state = write_vfs_file(state, "ui_ux_pizza_delivery.txt", content_3, initial_relevance_guess=15)
    
    # Run the Garbage Collector!
    state = run_vfs_garbage_collector(state)
    
    print(f"============================================================")
    print(f"FINAL SYSTEM MEMORY HEALTH METRICS")
    print(f"============================================================")
    print(f"Active Files in VFS : {len(state['vfs'])}")
    print(f"Current Token Space : {state['memory_usage_bytes']} Bytes (Highly Optimized)")
    print(f"Files Pruned/Deleted: {state['pruned_files']}")
    
    print(f"\n[CURRENT VFS CONTENTS]")
    for filename, md in state["vfs"].items():
        print(f"  \U0001f4c4 {filename}")
        print(f"     Size: {md['size_bytes']} Bytes | Compressions: {md['compressions']}")
        print(f"     Preview: {md['content'][:50]}...\n")
