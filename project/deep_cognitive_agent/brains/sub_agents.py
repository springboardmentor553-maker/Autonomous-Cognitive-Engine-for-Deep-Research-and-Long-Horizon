from brains.workers import get_worker_chain

# Export the native chains so config passes through them automatically
SUB_AGENTS = {
    "researcher": get_worker_chain("researcher"),
    "summarizer": get_worker_chain("summarizer"),
    "comparator": get_worker_chain("comparator"),
    "refiner": get_worker_chain("refiner")
}