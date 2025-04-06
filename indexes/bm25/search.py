from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('/home/uyen/workspace/nlp_project/data/cache/bm25_index')
hits = searcher.search('tiếng việt')

print("Result:")
for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')