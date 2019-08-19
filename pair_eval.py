import os
import csv
from predict import SemRankerPredict
from ndcg import ndcg
from vn_lang import query_preprocessing
import re

predictor = SemRankerPredict(using_gpu=False)

f1 = open("greater_0.7.txt", "w")
f2 = open("between_0.6-0.7.txt", "w")
f3 = open("lower_0.6.txt", "w")

count = 0
cum_ndcg = 0.
for f in os.listdir("eval_transform_impressions"):
    fpath = os.path.join("eval_transform_impressions", f)
    with open(fpath, 'r') as fobj:
        for j, r in enumerate(csv.reader(fobj)):
            query = r[0]
            pk = r[1].split("|")
            pnk = [z.split("#") for z in pk]
            rel = []
            pids = []
            negs = []
            for p in pnk:
                pids.append(p[0])
                if p[1] == '2':
                    rel.append(2)
                elif p[1] == '1':
                    rel.append(1)
                else:
                    rel.append(0)

            try:
                score, ret_products = predictor.fit(query, pids)
            except:
                continue
            

            ret_pids = list(map(lambda x: str(x.get('product_id')), ret_products))
            nz = []
            for i, p in enumerate(pids):
                if str(p) in ret_pids:
                    nz.append(rel[i])
            assert(len(score)==len(ret_products))
            assert(len(nz)== len(ret_products))


            for p, v, s in zip(ret_products, nz, score):
                p['rel'] = v
                p['score'] = s
            
            sorted_products = sorted(ret_products, key=lambda x: x.get('score'), reverse=True)

            ret_ndcg = ndcg(
                range(1, len(sorted_products)+1)[:100], [x['rel'] for x in sorted_products][:100])
            
            if ret_ndcg > 0.:
                cum_ndcg += ret_ndcg
                count += 1
                if count == 1000:
                    break
                if ret_ndcg > 0.7:
                    f1.write(query + ":%0.4f" % ret_ndcg + "\n")
                if ret_ndcg <= 0.7 and ret_ndcg > 0.6:
                    f2.write(query + ":%0.4f" % ret_ndcg + "\n")
                if ret_ndcg <= 0.6:
                    f3.write(query + ":%0.4f" % ret_ndcg + "\n")
            if count % 200 == 0:
                f1.flush()
                f2.flush()
                f3.flush()
                print("Processed %d queries, ndcg: %0.4f" % (count, cum_ndcg/count))
    if count >= 1000:
        break

print("NDCG: %0.4f" % (cum_ndcg/count))
f1.close()
f2.close()
f3.close()