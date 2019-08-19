import threading
import csv
import os
import re
import queue

q = queue.Queue()

input_dir = 'split_queries'
output_dir = 'transform_impressions'

def worker():
    while True:
        d = q.get()
        if d is None:
            break
        print("Dealing with q-%d.csv" % d)
        fobj = open(os.path.join(input_dir, "q-%d.csv" %d))
        reader = csv.reader(fobj)

        interactions = {}
        for r in reader:
            keyword = r[0]
            product_id = r[1]
            rel = int(r[2])
            if re.match(r'\d{6,}', keyword):
                continue
            if len(keyword) == 0:
                continue
            interactions[(keyword, product_id)] = max(interactions.get((keyword, product_id), 0), rel)
        fobj.close()
        queries = {}
        keep_q = {}
        for (keyword, product_id), v in interactions.items():
            if not keyword in queries:
                queries[keyword] = [(product_id, v)]
                keep_q[keyword] = bool(v)
            else:
                queries[keyword].append((product_id, v))
                keep_q[keyword] = keep_q[keyword] or bool(v)

        with open(os.path.join(output_dir, "q-%d.csv" % d), "w") as fobj:
            csv_writer = csv.writer(fobj)
            for k, v in keep_q.items():
                if not v:
                    continue
                l = ["#".join([str(p), str(x)]) for p, x in queries[k]]
                csv_writer.writerow([k, "|".join(l)])
        q.task_done()

threads = []
for i in range(16):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for i in range(512):
    q.put(i)
q.join()

for i in range(16):
    q.put(None)
for t in threads:
    t.join()