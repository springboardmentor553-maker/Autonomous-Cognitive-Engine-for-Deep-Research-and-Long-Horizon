from duckduckgo_search import DDGS

def search_web(query):

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r['body'])

    # Remove duplicates
    unique_results = list(set(results))

    return "\n".join(unique_results)