import json
import os

def load_test():
    MAILONG_DATA_DIR = '/home/uyen/workspace/nlp_project/repos/bert-vietnamese-question-answering/dataset'
    mailong_test_data = json.load(open(os.path.join(MAILONG_DATA_DIR, 'test_IR.json')))

    test_data = [
        {
            '__id__': item['__id__'],
            'question': item['question'],
            'title': item['title'],
        }
        for item in mailong_test_data
    ]

    return test_data

# def get_scores(preds, golds):
#     ranks = []
#     for i, gold in enumerate(golds):
#         rank = 0
#         for j, pred in enumerate(preds[i]):
#             if gold['title'].lower() == pred['title'].lower():
#                 rank = j + 1
#                 break
#         ranks.append(rank)
    
#     marks = [1, 5, 10, 50]
#     scores = {f'recall@{i}': 0 for i in marks}
#     for rank in ranks:
#         if rank > 0:
#             for i in marks:
#                 if rank <= i:
#                     scores[f'recall@{i}'] += 1
                    
#     scores = {k: v / len(ranks) for k, v in scores.items()}
#     return scores

import math

def get_scores(preds, golds, ks=[1, 5, 10, 50]):
   
    recalls = {f'recall@{k}': 0 for k in ks}
    rr_total = 0  # Reciprocal Rank

    ranks = []
    for i, gold in enumerate(golds):
        gold_title = gold['title'].lower()
        pred_titles = [p['title'].lower() for p in preds[i]]
        hits = []
        for j, title in enumerate(pred_titles):
            if gold_title == title:
                hits.append(j + 1)
        ranks.append(hits)

    # For Precision@k and Recall@k
    for hits in ranks:

        if len(hits) > 0:
            # Mean Reciprocal Rank (MRR)
            rr_total += 1 / hits[0]

            for k in ks:
                if hits[0] <= k:
                    recalls[f'recall@{k}'] += 1

    total = len(golds)
    recall_scores = {k: v / total for k, v in recalls.items()}
    mrr = rr_total / total

    scores =  {
        'mrr@10': mrr,
        **recall_scores,
        
    }

    scores = {k: round(v, 4)*100 for k, v in scores.items()}
    return scores
