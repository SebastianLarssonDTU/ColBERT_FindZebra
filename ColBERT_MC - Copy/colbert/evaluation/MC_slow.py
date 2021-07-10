import os
import numpy as np

def MC_slow_rerank(args, query, passages):
    colbert = args.colbert
    inference = args.inference

    Q = inference.queryFromText([query])
    D_ = inference.docFromText(passages, bsize=args.bsize)
    scores = colbert.score(Q, D_).cpu()

    unranked_scores = scores.tolist()
    if max(unranked_scores) == -args.query_maxlen:
        unranked_scores.append(0.0)
    else:
        unranked_scores.append(-1000.0)

    scores = scores.sort(descending=True)
    ranked = scores.indices.tolist()

    ranked_scores = scores.values.tolist()
    # ranked_pids = [pids[position] for position in ranked]
    ranked_passages = [passages[position] for position in ranked]

    if ranked_scores[0] == -args.query_maxlen:
        ranked_scores.insert(0, 0.0)
        ranked_passages.insert(0, "Uncertain")
    else:
        ranked_scores.append(-1000.0)
        ranked_passages.append("Uncertain")
    # assert len(ranked_pids) == len(set(ranked_pids))

    # return list(zip(ranked_scores, ranked_passages))
    return ranked_scores, ranked_passages, unranked_scores
