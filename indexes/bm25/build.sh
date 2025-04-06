python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/uyen/workspace/nlp_project/data/cache/bm25_collection \
  --index  /home/uyen/workspace/nlp_project/data/cache/bm25_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw