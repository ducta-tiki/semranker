import os
import csv
import json
from vn_lang import query_preprocessing

# Make vocabulary
count_unigrams = 0
count_bigrams = 0
count_char_trigrams = 0
unigrams = {}
bigrams = {}
char_trigrams = {}

def add_to_vocab(s):
    global count_unigrams, count_bigrams, count_char_trigrams, \
            unigrams, bigrams, char_trigrams
    tokens = s.split()
    for t in tokens:
        if len(t):
            if t in unigrams:
                unigrams[t] += 1
            else:
                unigrams[t] = 1
            count_unigrams += 1
            
            z = "#" + t +"#"
            for i in range(0, max(len(z)-2, 1)):
                v = z[i:i+3]
                if v in char_trigrams:
                    char_trigrams[v] += 1
                else:
                    char_trigrams[v] = 1
                count_char_trigrams += 1

    for i in range(0, max(len(tokens) - 1, 0)):
        t = "%s#%s" % (tokens[i], tokens[i+1])
        if t in bigrams:
            bigrams[t] += 1
        else:
            bigrams[t] = 1
        count_bigrams += 1

fobj = open("data/catalog.csv")
reader = csv.reader(fobj)

for r in reader:
    product_name = query_preprocessing(r[1])
    if not product_name or len(product_name) == 0:
        continue
    
    add_to_vocab(product_name)

fobj.close()

for f in os.listdir("triples"):
    if f.endswith(".csv"):
        fobj = open("data/catalog.csv")
        reader = csv.reader(fobj)
        for r in reader:
            query = query_preprocessing(r[0])
            if not query or len(query) == 0:
                continue
            add_to_vocab(query)
        fobj.close()


del_keys = []
for k in unigrams:
    if unigrams[k] <= 5:
        del_keys.append(k)
for k in del_keys:
    del unigrams[k]

del_keys = []
for k in bigrams:
    if bigrams[k] <= 5:
        del_keys.append(k)
for k in del_keys:
    del bigrams[k]

del_keys = []
for k in char_trigrams:
    if char_trigrams[k] <= 5:
        del_keys.append(k)
for k in del_keys:
    del char_trigrams[k]

vocab = unigrams.copy()
vocab.update(bigrams)
vocab.update(char_trigrams)
vocab = sorted(vocab.keys())

fobj = open("meta/vocab.txt", "w")
fobj.write("\n".join(vocab))
fobj.close()

# Make category tokens and attr tokens
features = {
    'reviews':[10000., 0.],
    'rating':[10000., 0.],
    'sales_monthly':[1000000., 0.],
    'sales_yearly':[1000000., 0.],
    'support_p2h_delivery':[2., 0.]
}

fobj = open("data/product.csv")
reader = csv.DictReader(fobj)
attr_token = set()
cat_token = set()

for r in reader:
    pattr = r.get('attributes')
    pcat = r.get('categories')

    ta = pattr.split("|")
    for tta in ta:
        z = tta.split("#")
        if len(z) > 0:
            attr_token.add("#".join(z[:2]))
    tc = pcat.split("|")
    for ttc in tc:
        z = ttc.split("#")
        if len(z) > 0:
            cat_token.add("#".join(z[:2]))
    
    for f in features:
        if r.get(f):
            try:
                fv = float(r.get(f))
                features[f][0] = min(features[f][0], fv)
                features[f][1] = max(features[f][1], fv)
            except:
                pass


fobj = open("meta/attrs.txt", "w")
fobj.write("\n".join(sorted(list(attr_token))))
fobj.close()

fobj = open("meta/cats.txt", "w")
fobj.write("\n".join(sorted(list(cat_token))))
fobj.close()

fobj = open("meta/precomputed.json", "w")
json.dump(features, fobj)
fobj.close()