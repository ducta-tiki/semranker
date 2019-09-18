import tensorflow as tf
from tf_csv_reader import CsvSemRankerReader


class CsvSemRankerReaderTest(tf.test.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testGenerateBatch(self):
        self.reader = CsvSemRankerReader(
            pair_paths=["../overfit/q-1.csv"],
            precomputed_path="../meta/precomputed.json",
            product_db="../data/product.csv",
            vocab_path="../meta/vocab.txt",
            cat_tokens_path="../meta/cats.txt",
            attr_tokens_path="../meta/attrs.txt",
            maximums_query=[25, 25, 125],#for unigram, bigram, character trigrams
            maximums_product_name=[50, 50, 250], #for unigram, bigram, character trigrams
            maximums_brand=[10, 10, 50],
            maximums_author=[10, 10, 50],
            maximums_cat=[10, 10, 20], #for unigram, bigram, character trigrams
            maximums_attr=[10, 10, 20], #for unigram, bigram, character trigrams
        )
        tensors = self.reader.get_batch(batch_size=2)

        with self.cached_session() as sess:
            for _ in range(1):
                results = sess.run(tensors)

                inputs = results[0]
                labels = results[1]
                # print(inputs['product_unigram_indices'])
                print(results)
    # def testVerifyTrainingData(self):
    #     for i in range(1):
    #         tf.reset_default_graph()
    #         reader = CsvSemRankerReader(
    #             pair_paths=["../overfit_transform_impressions/1_samples.csv"],
    #             precomputed_path="../meta/precomputed.json",
    #             product_db="../db/tiki-products.db",
    #             vocab_path="../meta/vocab.txt",
    #             cat_tokens_path="../meta/cats.txt",
    #             attr_tokens_path="../meta/attrs.txt",
    #             maximums_query=[25, 25, 125],#for unigram, bigram, character trigrams
    #             maximums_product_name=[50, 50, 250], #for unigram, bigram, character trigrams
    #             maximums_brand=[10, 10, 50],
    #             maximums_author=[10, 10, 50],
    #             maximums_cat=[10, 10, 50], #for unigram, bigram, character trigrams
    #             maximums_attr=[10, 10, 50], #for unigram, bigram, character trigrams
    #         )
    #         print("\nTesting /q-%d.csv" % i)
    #         tensors = reader.get_batch(batch_size=1)

    #         with self.cached_session() as sess:
    #             for _ in range(1):
    #                 print("range: %d" % _)
    #                 results = sess.run(tensors)


if __name__ == "__main__":
    tf.test.main()