from fastapi import FastAPI,Depends
from pydantic import BaseModel, field_validator
from pyserini.search.lucene import LuceneSearcher
from .utils import basic_tokenizer, connect_db
from typing import List
from .encoder import Encoder
from .reranker import PairwiseModel as Reranker
from faiss import read_index
import torch

app = FastAPI()

class Searcher:
    def __init__(self):
        self.bm25_index = LuceneSearcher('/home/uyen/workspace/nlp_project/data/cache/bm25_index')
        self.dense_index = read_index("/home/uyen/workspace/nlp_project/data/cache/dense_index/bert.index")
        self.collection_path = "/home/uyen/workspace/nlp_project/data/cache/database/wiki.db"
        self.encoder = Encoder("bkai-foundation-models/vietnamese-bi-encoder", cache_dir="/home/uyen/workspace/nlp_project/cache")

        self.reranker = Reranker("nguyenvulebinh/vi-mrc-base").half()
        state_dict = torch.load("/home/uyen/workspace/nlp_project/data/zac-data/pairwise_v2.bin")
        self.reranker.load_state_dict(state_dict,  strict=False)
        self.reranker.eval()
    
    def _search_dense(self, query: str, limit = 10):
        query_vector = self.encoder.encode([query])
        
        # Tìm kiếm trong index
        D, I = self.dense_index.search(query_vector, k=limit)

        results = []
        for score, index in zip(D[0].tolist(), I[0].tolist()):
            results.append({
                'docid': int(index),
                'score': score
            })
        return results

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

    def _rerank(self, query: str, texts: list[str]):
        # Sử dụng mô hình để dự đoán điểm số
        scores = self.reranker.stage1_ranking(query, texts)
        return scores.tolist()

    
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
    results = {}
    if "bm25" in req.types:
        bm25_results = searcher._search_bm25(req.query, limit=req.limit)
        for rank, result in enumerate(bm25_results):
            results[result["docid"]] = rank

    if "dense" in req.types:
        dense_results = searcher._search_dense(req.query, limit=req.limit)
        for rank, result in enumerate(dense_results):
            if result["docid"] in results:
                results[result["docid"]] = (results[result["docid"]] + rank) / 2
            else:
                results[result["docid"]] = rank

    results = sorted([{
        "docid": k,
        "score": v
    } for k, v in results.items()], key=lambda x: x["score"])
    results = results[:req.limit]

    if "rerank" in req.types:
        # Lấy danh sách các văn bản từ các kết quả đã tìm kiếm

        docids = list([result["docid"] for result in results])
        chunks = searcher._get_chunks(docids, ["id","text"])

        # Sử dụng mô hình để dự đoán điểm số
        scores = searcher._rerank(req.query, [chunk["text"] for chunk in chunks])
        results = sorted([{
            "docid": chunk["id"],
            "score": score
        } for chunk, score in zip(chunks, scores)], key=lambda x: -x["score"])

    # Lúc này types đã được validate rồi
    return {
        "query": req.query,
        "limit": req.limit,
        "search_methods": req.types,
        "results": results
    }

@app.post("/get_chunks")
def get_chunks(req: ChunkRequest, searcher: Searcher = Depends(get_searcher)):
    results = searcher._get_chunks(req.ids, req.fields)
    return {"chunks": results}
