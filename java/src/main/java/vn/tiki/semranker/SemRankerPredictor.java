package vn.tiki.semranker;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;


public class SemRankerPredictor {
    String QUERY_UNIGRAM_INDICES = "query_unigram_indices";
    String QUERY_BIGRAM_INDICES = "query_bigram_indices";
    String QUERY_CHAR_TRIGRAM_INDICES = "query_char_trigram_indices";
    String PRODUCT_UNIGRAM_INDICES = "product_unigram_indices";
    String PRODUCT_BIGRAM_INDICES = "product_bigram_indices";
    String PRODUCT_CHAR_TRIGRAM_INDICES = "product_char_trigram_indices";
    String BRAND_UNIGRAM_INDICES = "brand_unigram_indices";
    String BRAND_BIGRAM_INDICES = "brand_bigram_indices";
    String BRAND_CHAR_TRIGRAM_INDICES = "brand_char_trigram_indices";
    String AUTHOR_UNIGRAM_INDICES = "author_unigram_indices";
    String AUTHOR_BIGRAM_INDICES = "author_bigram_indices";
    String AUTHOR_CHAR_TRIGRAM_INDICES = "author_char_trigram_indices";
    String CAT_UNIGRAM_INDICES = "cat_unigram_indices";
    String CAT_BIGRAM_INDICES = "cat_bigram_indices";
    String CAT_CHAR_TRIGRAM_INDICES = "cat_char_trigram_indices";
    String CAT_TOKENS = "cat_tokens";
    String CATS_IN_PRODUCT = "cats_in_product";
    String ATTR_UNIGRAM_INDICES = "attr_unigram_indices";
    String ATTR_BIGRAM_INDICES = "attr_bigram_indices";
    String ATTR_CHAR_TRIGRAM_INDICES = "attr_char_trigram_indices";
    String ATTR_TOKENS = "attr_tokens";
    String ATTRS_IN_PRODUCT = "attrs_in_product";
    String FREE_FEATURES = "free_features";
    String SCORE = "score:0";

    Map<String, Integer> vocab = null;
    Map<String, Integer> catTokens = null;
    Map<String, Integer> attrTokens = null;

    private void loadTokens(String filePath, Map<String, Integer> mapst){
        BufferedReader reader;
        try{
            reader = new BufferedReader(new FileReader(filePath));
            int idx = 0;
            while(true){
                String word = reader.readLine();
                if(word == null) break;
                if(word.trim().length() > 0){
                    mapst.put(word, idx);
                    idx += 1;
                }

            }
        }catch (IOException e){
            e.printStackTrace();
        }

    }

    public SemRankerPredictor(String checkpoint){
        this.vocab = new HashMap<String, Integer>();
        this.catTokens = new HashMap<String, Integer>();
        this.attrTokens = new HashMap<String, Integer>();

        String vocabPath = Paths.get(checkpoint, "vocab.txt").toString();
        this.loadTokens(vocabPath, this.vocab);

        String catTokensPath = Paths.get(checkpoint, "cats.txt").toString();
        this.loadTokens(catTokensPath, this.catTokens);

        String attrTokensPath = Paths.get(checkpoint, "attrs.txt").toString();
        this.loadTokens(attrTokensPath, this.catTokens);

        JSONParser parser = new JSONParser();
        Path metaPath = Paths.get(checkpointPath, "meta.json");
    }
}
