from brains.workers import worker_logic
from langchain_core.runnables import RunnableLambda

# Create Runnable objects for the Supervisor to invoke
SUB_AGENTS = {
    "researcher": RunnableLambda(lambda x: worker_logic("researcher", x)),
    "summarizer": RunnableLambda(lambda x: worker_logic("summarizer", x)),
    "comparator": RunnableLambda(lambda x: worker_logic("comparator", x)),
    "refiner": RunnableLambda(lambda x: worker_logic("refiner", x))
}