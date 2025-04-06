import requests
from tqdm import tqdm
from utils import load_test,get_scores

BASE_URL = "http://localhost:8002"

def _search(query, limit=10):
    url = f"{BASE_URL}/search"
    payload = {
        "query": query,
        "limit": limit,
        "types": ["bm25","rerank"]
    }
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def get_chunks(ids, fields):
    url = f"{BASE_URL}/get_chunks"
    payload = {
        "ids": ids,
        "fields": fields
    }
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def search(query, limit=10):
    search_results = _search(query, limit)["results"]
    ids = [result["docid"] for result in search_results]
    fields = ["id", "title"]
    chunk_results = get_chunks(ids, fields)["chunks"]
    # Merge search results with chunk results
    for result in search_results:
        for chunk in chunk_results:
            if result["docid"] == chunk["id"]:
                result.update(chunk)
                break
    # Remove chunk results from search results
    search_results = [{
        "id": result["docid"],
        "title": result["title"],
        "score": result["score"]
    } for result in search_results]

    return search_results

if __name__ == "__main__":
    golds = load_test()
    preds = []
    limit = 50
    for item in tqdm(golds):
        query = item["question"]
        search_results = search(query, limit)
        preds.append(search_results)

    scores = get_scores(preds, golds)
    print("Scores:", scores)