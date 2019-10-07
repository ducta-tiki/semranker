import os
import csv
import time
file_path = "data/product.csv"
output_dir = "modulo/%d" % (int(time.time() * 1000))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fobjs = []
writers = []

fields = [
    'name','brand','author','attributes','categories','reviews','rating','sales_monthly','sales_yearly','support_p2h_delivery'
]

def fix_nulls(s):
    for line in s:
        yield line.replace('\0', '')

for i in range(100):
    fobj = open(output_dir + "/%d.csv" % i, "w")
    fobjs.append(fobj)
    writers.append(csv.writer(fobj, delimiter="|"))
    writers[i].writerow(['product_id',] + fields)



count = 0
with open(file_path, "r") as fin:
    reader = csv.DictReader(fix_nulls(fin))

    for r in reader:
        product_id = r.get("product_id")
        modulo = int(product_id) % 100
        count += 1
        if count % 10000 == 0:
            print("Processed %d products" % count)
        writer = writers[modulo]
        row = [product_id,] + [r.get(f) for f in fields]
        writer.writerow(row)

for f in fobjs:
    f.close()
