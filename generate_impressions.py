import csv
import os
from vn_lang import query_preprocessing

positives = set()
impressions = set()
negatives = set()


count = 0
for f in os.listdir('product_impressions'):
    if f.endswith('.csv'):
        print("Processing file: " + f)
        fobj = open(os.path.join('product_impressions', f))
        reader = csv.DictReader(fobj, fieldnames=[
            'fullVisitorId', 'keyword', 'product_id', 'position', 'score', 'action'
        ])

        for row in reader:
            keyword = row.get('keyword')
            product_id = row.get('product_id')
            action = row.get('action')

            if keyword is None or product_id is None:
                continue
            
            keyword = query_preprocessing(keyword)
            if not len(keyword):
                continue
            count += 1
            if count % 10000 == 0:
                print("Processed %d pair (keyword, product)" % count)
            pair = (keyword, product_id) 
            inp = pair not in positives
            ini = pair not in impressions
            inn = pair not in negatives
            if action == 'buy':
                if inp:
                    positives.add(pair)
                    if not ini: impressions.remove(pair)
                    if not inn: negatives.remove(pair)
            else:
                if action is None or action == '':
                    if inp and ini and inn:
                        negatives.add(pair)
                else:
                    if inp and ini:
                        impressions.add(pair)
                        if not inn:
                            negatives.remove(pair)
        fobj.close()


for fname, ptup in zip(['positives', 'impressions', 'negatives'], 
                        [positives, impressions, negatives]):
    fobj = open('triples/%s.csv' % fname, 'w')
    writer = csv.DictWriter(fobj, fieldnames=['keyword', 'product_id'])

    for pair in ptup:
        writer.writerow({"keyword": pair[0], "product_id": pair[1]})

    fobj.close()