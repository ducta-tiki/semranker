import tensorflow as tf
from tf_csv_reader import CsvSemRankerReader


class CsvSemRankerReaderTest(tf.test.TestCase):
    def setUp(self):
        self.reader = CsvSemRankerReader(
            pos_path="../triples/positives.csv",
            zero_path="../triples/negatives.csv",
            neg_path="../triples/impressions.csv",
            precomputed_path="../meta/precomputed.json",
            product_db="../db/tiki-products.db",
            vocab_path="../meta/vocab.txt",
            cat_tokens_path="../meta/cats.txt",
            attr_tokens_path="../meta/attrs.txt",
            maximums_query=[25, 25, 125],#for unigram, bigram, character trigrams
            maximums_product_name=[50, 50, 250], #for unigram, bigram, character trigrams
            maximums_brand=[10, 10, 50],
            maximums_author=[10, 10, 50],
            maximums_cat=[10, 10, 50], #for unigram, bigram, character trigrams
            maximums_attr=[10, 10, 50], #for unigram, bigram, character trigrams
        )

    def tearDown(self):
        self.reader.end()

    def testGenerateBatch(self):
        tensors = self.reader.get_batch(batch_size=4)

        with self.cached_session() as sess:
            results = sess.run(tensors)
            print(results[2])


if __name__ == "__main__":
    tf.test.main()