def evaluate(state):
    total = len(state.completed)
    success = total
    score = success / max(total, 1)
    state.score = score
    return score