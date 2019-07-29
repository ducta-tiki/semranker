import os
import json
import csv
from vn_lang import query_preprocessing
from pymongo import MongoClient


class Category(object):
    def __init__(self, _id, name):
        self._id = _id
        self.name = name

    def to_dict(self):
        return {
            "id": self._id,
            "name": self.name
        }

    def __repr__(self):
        return str(self.to_dict())

    def __eq__(self, other):
        return self._id == other._id

    def __hash__(self):
        return hash(self._id)


def none_or_bool(b):
    if not b:
        return False
    if bool(int(b)):
        return True
    return False

class AggregationFields(object):
    def __init__(self,
            db, attr_collection,
            category_collection,
            product_collection):
        self.db = MongoClient(db)['talaria-browser']
        self.attr_collection = self.db[attr_collection]
        self.attr_last_id = None

        self.product_collection = self.db[product_collection]
        self.product_last_id = None
        self.discard_attrs = {
            'support_subscription',
            'brand',
            'nha_mang',
            'subtitles',
            'only_ship_to',
            'phong_cach',
            'filter_washer_drum',
            'filter_water_heater_capability',
            'menh_gia',
            'product_feature'
        }

        self.category_collection = self.db[category_collection]
        self.category_last_id = None

    def normalize_attr(self, doc):
        return {
            "id": doc.get("id"),
            "name": doc.get("code"),
            "display_name": doc.get("display_name"),
            "input_type": doc.get("input_type"),
            "data_type": doc.get("data_type")
        }

    def fetch_attr(self, batch_size=100):
        cond = {}
        if self.attr_last_id:
            cond = {"_id": {"$gt": self.attr_last_id}}
        results = list(self.attr_collection.find(
            cond, no_cursor_timeout=True).limit(batch_size))
        if len(results):
            self.attr_last_id = results[-1]["_id"]
        results = filter(lambda x: none_or_bool(x.get("is_filterable")), results)
        results = map(lambda x: self.normalize_attr(x), results)

        return list(results)

    def fetch_categories(self, batch_size=100):
        cond = {}
        if self.category_last_id:
            cond = {"_id": {"$gt": self.category_last_id}}
        results = list(self.category_collection.find(
                        cond, {'_id':1, 'id':1, 'path':1, 'include_in_menu':1, 'is_active': 1, 'level': 1}, no_cursor_timeout=True)
                            .limit(batch_size))
        if len(results):
            self.category_last_id = results[-1]["_id"]
        
        results = filter(lambda x: none_or_bool(x.get("include_in_menu")) and none_or_bool(x.get('is_active')) and (int(x.get('level')) <= 5), results)
        return list(results)

    def make_product_projection(self):
        projs = {'_id':1, 'id':1, 'name': 1, 'brand': 1, 
        'author': 1, 'category': 1, 'entity_type': 1, 'reviews_count':1, 
        'rating_summary': 1, 'sales_volume': 1, 'support_p2h_delivery': 1}
        for attr in self.attrs:
            projs[attr] = 1

        self.product_projs = projs
        

    def fetch_product(self, batch_size=100):
        cond = {}
        if self.product_last_id:
            cond = {"_id": {"$gt": self.product_last_id}}
        results = list(self.product_collection.find(
                        cond, self.product_projs, no_cursor_timeout=True).limit(batch_size))
        if len(results):
            self.product_last_id = results[-1]["_id"]

        return results

    def fetch_all_attr(self):
        fields = []
        results = self.fetch_attr()
        while len(results) > 0:
            fields += results
            results = self.fetch_attr()
        df = {}
        for v in fields:
            df[v['name']] = v
        return df
    
    def fetch_all_cats(self):
        fields = []
        results = self.fetch_categories()
        count = 0
        while len(results) > 0:
            count += 1
            fields += results
            if count % 20 == 0:
                print("Fetched %d categories" % len(fields))
            results = self.fetch_categories()

        df = {}
        for v in fields:
            df[v['id']] = v
        return df

    def gather(self):
        results = {}
        categories = {}
        attrs = self.fetch_all_attr()
        cats = self.fetch_all_cats()
        
        self.attrs = attrs.keys()
        self.make_product_projection()
        truncate = {}
        new_cats = set()
        for k, v in cats.items():
            path = v.get('path')
            skip = False
            if path:
                for scid in path.split("/"):
                    if len(scid):
                        z = int(scid)
                        if z <= 2:
                            continue
                        if not z in cats:
                            skip = True
                            break
            if not skip:
                new_cats.add(k)
        try:
            new_cats.remove(2)
            new_cats.remove(1)
        except:
            pass

        f_product = open("data/product.csv", "w")
        csv_writer = csv.DictWriter(f_product, [
            'product_id', 'name', 'brand', 'author', 'attributes', 'categories',
            'reviews', 'rating', 'sales_monthly', 'sales_yearly', 'support_p2h_delivery'])
        csv_writer.writeheader()
        ps = self.fetch_product()
        count = 0
        while len(ps) > 0:
            for p in ps:
                if not p.get('entity_type') in {'master_simple', 'master_configurable', 'master_virtual'}:
                    continue
                product_name =  p.get('name')
                if product_name:
                    product_name = product_name.strip()
                else:
                    continue
                brand = ""
                if p.get("brand"):
                    brand = p.get("brand").get("value").strip()
                authors = []
                for a in p.get("author", []):
                    if a.get("name"):
                        authors.append(a.get("name").strip())

                reviews_count = 0
                if p.get('reviews_count'):
                    reviews_count = int(p.get('reviews_count'))
                rating_summary = 0
                if p.get('rating_summary'):
                    rating_summary = int(p.get('rating_summary'))
                sales_monthly = 0
                sales_yearly = 0
                if p.get('sales_volume'):
                    sales_monthly = int(p.get('sales_volume').get('sales_monthly', 0))
                    sales_yearly = int(p.get('sales_volume').get('sales_yearly', 0))
                support_p2h_delivery = 0
                if p.get('support_p2h_delivery'):
                    support_p2h_delivery = int(p.get('support_p2h_delivery'))

                count += 1
                if count % 500 == 0:
                    print("Processed %d products" % count)
                pcats = p.get("category", [])
                write_cats = []
                write_attrs = []

                consider_cid = []
                for cat in pcats:
                    if not cat['id'] in new_cats:
                        continue
                    path = cat.get('path')
                    skip = False
                    if path:
                        for scid in path.split("/"):
                            if len(scid):
                                z = int(scid)
                                if z <= 2:
                                    continue
                                if not z in new_cats:
                                    skip = True
                                    break
                    if skip:
                        continue          
                    cid = cat.get("id")
                    cname = cat.get("name")
                    level = cat.get("level")
                    consider_cid.append(cat)
                    write_cats.append("#".join([str(cid), str(level), cname]))

                for attr_name, attr in attrs.items():
                    if attr_name in self.discard_attrs:
                        continue
                    value = p.get(attr_name)

                    if isinstance(value, list):
                        value = list(map(lambda x: x.get("value"), value))
                    elif isinstance(value, dict):
                        value = value.get("value")
                        value = [value]
                    else:
                        value = None
                    if value:
                        for v in value:
                            if not isinstance(v, int) and not isinstance(v, float):
                                write_attrs.append("#".join([str(attr['id']), attr['name'], str(v)]))

                    for cat in consider_cid:
                        cid = cat.get("id")
                        cname = cat.get("name")
                        level = cat.get("level")

                        if cid not in categories:
                            categories[cid] = {"__name__": cname, "__id__": cid, "__level__": level}

                        if not value is None:
                            if attr_name not in categories[cid]:
                                categories[cid][attr_name] = set()
                            for vs in value:
                                categories[cid][attr_name].add(vs)
                                
                csv_writer.writerow({
                    'product_id': str(p.get('id')),
                    'name': product_name,
                    'brand': brand,
                    'author': ";".join(authors),
                    'attributes': "|".join(write_attrs),
                    'categories': '|'.join(sorted(write_cats, key=lambda x:x.split("#")[1])),
                    'reviews': reviews_count,
                    'rating': rating_summary,
                    'sales_monthly': sales_monthly,
                    'sales_yearly': sales_yearly,
                    'support_p2h_delivery': support_p2h_delivery
                })

            ps = self.fetch_product()

        for cat, v in categories.items():
            keys = v.keys()
            for k in keys:
                if k != "__id__" and k != "__name__" and k != "__level__":
                    if isinstance(v[k], str) and len(v[k]) == 0:
                        del v[k]
                        continue
                    v[k] = list(v[k])

        f_product.close()
        return categories, attrs

    def save(self, categories, attrs):
        cat_save_path = "data/cat_aggs.json"
        attrs_save_path = "data/attrs.json"

        with open(cat_save_path, mode='w') as fobj:
            json.dump(categories, fobj, ensure_ascii=False)

        with open(attrs_save_path, mode='w') as fobj:
            json.dump(attrs, fobj, ensure_ascii=False)


if __name__ == "__main__":
    MONGO_PORT = int(os.environ.get("MONGO_PORT", "27017"))
    # MONGO_HOST = "mongodb://%s:%d/" % (os.environ.get("MONGO_HOST", "discovery-mongodb-1.svr.tiki.services"), MONGO_PORT)
    MONGO_HOST = "mongodb://%s:%d/" % (os.environ.get("MONGO_HOST", "10.20.40.142"), MONGO_PORT)

    aggf = AggregationFields(
        MONGO_HOST, "catalog_attribute", "category", "product")
    categories, attrs = aggf.gather()
    aggf.save(categories, attrs)