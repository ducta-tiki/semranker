import math

def ndcg(pos, relevance):
    log2_pos = list(map(lambda x: math.log2(x+1), pos))
    rel_div_log = [rel/lp for lp, rel in zip(log2_pos, relevance)]
    dcg = sum(rel_div_log)

    sorted_relevance = sorted(relevance, reverse=True)
    idcg = 0.

    for i, sr in enumerate(sorted_relevance):
        idcg += sr/math.log2(i+2)
    if idcg == 0.:
        return 0.
    return dcg/idcg
