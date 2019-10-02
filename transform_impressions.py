import threading
import csv
import os
import re
from multiprocessing import Process
from multiprocessing import Queue
import random

q = Queue()

input_dir = 'split_queries'
output_dir = 'transform_impressions'
eval_output_dir = 'eval_transform_impressions'

def worker():
    while True:
        d = q.get()
        if d is None:
            break
        # if d != 263:
        #     continue
        print("Dealing with q-%d.csv" % d)
        fobj = open(os.path.join(input_dir, "q-%d.csv" %d))
        reader = csv.reader(fobj)
        interactions = {}
        max_interactions = {}
        for r in reader:
            if len(r) < 2:
                continue
            keyword = r[0]
            product_id = r[1]
            rel = int(r[2])
            if re.match(r'\d{6,}', keyword):
                continue
            if len(keyword) == 0:
                continue
            # interactions[(keyword, product_id)] = max(interactions.get((keyword, product_id), 0), rel)
            interactions[(keyword, product_id)] = interactions.get((keyword, product_id), 0) + rel
            max_interactions[(keyword, product_id)] = max(max_interactions.get((keyword, product_id), 0), rel)
        fobj.close()
        queries = {}
        keep_q = {}
        sum_k = {}
        for (keyword, product_id), v in interactions.items():
            if not keyword in queries:
                queries[keyword] = [(product_id, max_interactions[(keyword, product_id)], v)]
                keep_q[keyword] = bool(v)
                sum_k[keyword] = v
            else:
                queries[keyword].append((product_id, max_interactions[(keyword, product_id)], v))
                keep_q[keyword] = keep_q[keyword] or bool(v)
                sum_k[keyword] += v
        
        ftrain = open(os.path.join(output_dir, "q-%d.csv" % d), "w")
        feval = open(os.path.join(eval_output_dir, "q-%d.csv" % d), "w")
        
        # with open(os.path.join(zoutput_dir, "q-%d.csv" % d), "w") as fobj:
        train_csv_writer = csv.writer(ftrain)
        eval_csv_writer = csv.writer(feval)
        for k, v in keep_q.items():
            if not v:
                continue

            l = []
            for p, m, x in queries[k]:
                score = float(x)/sum_k[k]
                if score < 0.01 and score > 0.001:
                    m = min(m, 1)
                if score <= 0.001:
                    m = 0
                l.append("#".join([str(p), str(m)]))
            eps = random.random()
            if eps < 0.2:
                csv_writer = eval_csv_writer
            else:
                csv_writer = train_csv_writer
            csv_writer.writerow([k, "|".join(l)])
        ftrain.close()
        feval.close()
        print("Done with q-%d.csv" % d)

procs = []
for i in range(16):
    t = Process(target=worker)
    t.start()
    procs.append(t)

for i in range(512):
    q.put(i)

for i in range(16):
    q.put(None)

while not q.empty():
    continue

print("Completed!")