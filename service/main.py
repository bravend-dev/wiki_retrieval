from fastapi import FastAPI,Depends
from pydantic import BaseModel, field_validator
from pyserini.search.lucene import LuceneSearcher
from .utils import basic_tokenizer, connect_db
from typing import List

app = FastAPI()

class Searcher:
    def __init__(self):
        self.bm25_index = LuceneSearcher('D:\\DungFolder\\workspace\\wiki_retrieval\\bm25\\indexes\\wiki')
        self.collection_path = "D:\\DungFolder\\workspace\\wiki_retrieval\\notebooks\\wiki.db"

    def _search_bm25(self, query: str, limit = 10):
        _query = basic_tokenizer(query)
        hits = self.bm25_index.search(_query, k=limit)
        results = []
        for hit in hits:
            results.append({
                'docid': int(hit.docid),
                'score': hit.score
            })

        return results

    def _get_chunks(self, ids: list[int], fields: list[str]):
        if not ids:
            return []

        conn, cursor = connect_db(self.collection_path)

        placeholders = ','.join(['?'] * len(ids))  # tạo chuỗi ?,?,?...
        placefield = ','.join(fields)
        query = f"SELECT {placefield} FROM wiki WHERE id IN ({placeholders})"

        cursor.execute(query, ids)

        return [
            {
                k:v for k,v in zip(fields, values)
            } for values in cursor.fetchall()
        ]

@app.on_event("startup")
async def startup_event():
    """Loads the model during application startup."""
    searcher = Searcher()
    app.state.searcher = searcher

def get_searcher():
    """Dependency to get the loaded model."""
    return app.state.searcher

VALID_SEARCH_TYPES = {"bm25", "dense", "rerank"}
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    types: List[str] = ["bm25"]

    @field_validator('types')
    def validate_search_types(cls, v):
        invalid = [t for t in v if t not in VALID_SEARCH_TYPES]
        if invalid:
            raise ValueError(f"Kiểu search không hợp lệ: {', '.join(invalid)}. Cho phép: {', '.join(VALID_SEARCH_TYPES)}")
        return v

VALID_FIELDS = {"id", "title", "text"}
class ChunkRequest(BaseModel):
    ids: List[int]
    fields: List[str] = ["id", "title", "text"]

    @field_validator('fields')
    def validate_search_fields(cls, v):
        invalid = [t for t in v if t not in VALID_FIELDS]
        if invalid:
            raise ValueError(f"Kiểu field không hợp lệ: {', '.join(invalid)}. Cho phép: {', '.join(VALID_FIELDS)}")
        return v

@app.post("/search")
async def search(req: SearchRequest, searcher: Searcher = Depends(get_searcher)):
    output = searcher._search_bm25(req.query, limit=req.limit)

    # Lúc này types đã được validate rồi
    return {
        "query": req.query,
        "limit": req.limit,
        "search_methods": req.types,
        "results": output  # xử lý sau
    }

@app.post("/get_chunks")
def get_chunks(req: ChunkRequest, searcher: Searcher = Depends(get_searcher)):
    results = searcher._get_chunks(req.ids, req.fields)
    return {"chunks": results}
