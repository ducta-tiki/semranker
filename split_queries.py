import csv
import os
import random
from vn_lang import query_preprocessing
import re
import pyhash

count = 0

dfobjs = {}
hasher = pyhash.murmur3_32()


def hash_kw(query, bins):
    return hasher(query) % bins

input_dir = 'product_impressions'
output_dir = 'split_queries'

def fix_nulls(s):
    for line in s:
        yield line.replace('\0', '')

for f in sorted(list(os.listdir(input_dir))):
    if f.endswith('.csv'):
        print("Processing file: " + f)
        fobj = open(os.path.join(input_dir, f))
        reader = csv.DictReader(fix_nulls(fobj))

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

            keyword = query_preprocessing(keyword)
            hkw = hash_kw(keyword, 512)
            
            if not hkw in dfobjs:
                fp = open(os.path.join(output_dir, "q-%d.csv" % hkw), "w")
                dfobjs[hkw] = [
                    fp,
                    csv.writer(fp)
                ]
            else:
                dfobjs[hkw][1].writerow([keyword, product_id, str(rel)])
            count += 1
            if count % 100000 == 0:
                print("Processed %d pair (keyword, product)" % count)


for k, v in dfobjs.items():
    v[0].close()