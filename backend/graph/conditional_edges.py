def planning_router(state: dict) -> str:
    """
    Routes flow after validation.
    If validated → go to synthesis.
    Else → retry planning.
    """

    if state["planning_meta"]["validated"]:
        return "synthesis"
    else:
        return "planning"