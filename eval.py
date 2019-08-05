import csv
from predict import SemRankerPredict
from ndcg import ndcg

file_paths = [
    "product_impressions/product_impressions_20190803_000000000000.csv",
    "product_impressions/product_impressions_20190804_000000000000.csv"
]

predictor = SemRankerPredict()
interaction = {}

count = 0

for file_path in file_paths:
    with open(file_path, 'r') as fobj:
        for r in csv.DictReader(fobj):
            if r.get('product_id'):
                query = r.get('keyword')
                product_id = int(r.get('product_id'))
                action = r.get('action')
                rel = 1
                if action == "buy":
                    rel = 2
                if action == "" or action is None:
                    rel = 0

                if (query, product_id) in interaction:
                    interaction[(query, product_id)] = max(interaction[(query, product_id)], rel)
                else:
                    interaction[(query, product_id)] = rel

dict_q = {}
for k, v in interaction.items():
    query = k[0]
    pid = k[1]
    rel = v
    if query in dict_q:
        dict_q[query].append((pid, rel))
    else:
        dict_q[query] = [(pid, rel)]

cum_ndcg = 0.
ndcg_value = 0.
total_queries = len(dict_q)
print("Total queries: %d" % total_queries)
count = 0
for query, z in dict_q.items():
    if len(z) == 0:
        continue
    try:
        score, ret_products = predictor.fit(query, [x[0] for x in z])
    except:
        continue
    

    pids = list(map(lambda x: str(x.get('product_id')), ret_products))
    nz = []
    for p in z:
        if str(p[0]) in pids:
            nz.append(p)
    assert(len(score)==len(ret_products))
    assert(len(nz)== len(ret_products))


    for p, v, s in zip(ret_products, nz, score):
        p['rel'] = v[1]
        p['score'] = s
    
    #for p in ret_products:
    #    if p.get('score') is None:
    #        print(p)
    sorted_products = sorted(ret_products, key=lambda x: x.get('score'), reverse=True)

    ret_ndcg = ndcg(range(1, len(sorted_products)+1), [x['rel'] for x in sorted_products])
    if ret_ndcg > 0.:
        count += 1
        cum_ndcg += ret_ndcg
        if count % 200 == 0 :
            print("Processed %d queries, ndcg: %0.4f" % (count, cum_ndcg/count))

    if count > 1000:
        break
print("NDCG: %0.4f" % (cum_ndcg/count))