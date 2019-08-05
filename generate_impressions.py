import csv
import os
import random
from vn_lang import query_preprocessing
import re

interactions = {}
negatives_pool = set()
count = 0
for f in sorted(list(os.listdir('product_impressions'))):
    if f.endswith('.csv'):
        print("Processing file: " + f)
        fobj = open(os.path.join('product_impressions', f))
        reader = csv.DictReader(fobj)

        for row in reader:
            keyword = row.get('keyword')
            product_id = row.get('product_id')
            action = row.get('action')

            if re.match(r'\d{6,}', keyword):
                continue
            rel = 1
            if action == "buy":
                rel = 2
            if action == "" or action is None:
                rel = 0

            if keyword is None or product_id is None:
                continue
            if random.random() < 0.4 and rel == 0: #drop 40% negatives samples
                continue
            keyword = query_preprocessing(keyword)
            if len(negatives_pool) < 5e6:
                negatives_pool.add((keyword, product_id))
            if not len(keyword):
                continue
            count += 1
            if count % 10000 == 0:
                print("Processed %d pair (keyword, product)" % count)
            interactions[(keyword, product_id)] = max(interactions.get((keyword, product_id), 0), rel)
        fobj.close()

negatives_pool = list(negatives_pool)
queries = {}
keep_q = {}
for (keyword, product_id), v in interactions.items():
    if not keyword in queries:
        queries[keyword] = [(product_id, v)]
        keep_q[keyword] = bool(v)
    else:
        queries[keyword].append((product_id, v))
        keep_q[keyword] = keep_q[keyword] or bool(v)

with open("pairs.csv", "w") as fobj:
    csv_writer = csv.writer(fobj)
    for k, v in keep_q.items():
        if not v:
            continue
        negs = random.sample(negatives_pool, 20)
        l = ["#".join([str(p), str(x)]) for p, x in queries[k]]
        z = ["#".join([str(p), "0"]) for kk, p in negs if kk != k]
        csv_writer.writerow([k, "|".join(l+z)])