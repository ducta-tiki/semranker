package org.tiki.vn;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
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


public class SemRankerPredict {
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
    Map<String, Integer[]> precomputed = null;
    int maxQueryLength = 0;
    int maxProductNameLength = 0;
    int maxBrandLength = 0;
    int maxAuthorLength = 0;
    int maxCatLength = 0;
    int maxAttrLength = 0;
    int unknownBin = 0;

    Session session = null;

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

    private class InputTensors{
        public Tensor<?> queryUnigramIndicesTensor;
        public Tensor<?> queryBigramIndicesTensor;
        public Tensor<?> queryCharTrigramIndicesTensor;
        public Tensor<?> productUnigramIndicesTensor;
        public Tensor<?> productBigramIndicesTensor;
        public Tensor<?> productCharTrigramIndicesTensor;
        public Tensor<?> brandUnigramIndicesTensor;
        public Tensor<?> brandBigramIndicesTensor;
        public Tensor<?> brandCharTrigramIndicesTensor;
        public Tensor<?> authorUnigramIndicesTensor;
        public Tensor<?> authorBigramIndicesTensor;
        public Tensor<?> authorCharTrigramIndicesTensor;
        public Tensor<?> catUnigramIndicesTensor;
        public Tensor<?> catBigramIndicesTensor;
        public Tensor<?> catCharTrigramIndicesTensor;
        public Tensor<?> catTokensTensor;
        public Tensor<?> catInProductTensor;
        public Tensor<?> attrUnigramIndicesTensor;
        public Tensor<?> attrBigramIndicesTensor;
        public Tensor<?> attrCharTrigramIndicesTensor;
        public Tensor<?> attrTokensTensor;
        public Tensor<?> attrInProductTensor;
        public Tensor<?> freeFeaturesTensor;

        public InputTensors(
                Tensor<?> queryUnigramIndicesTensor,
                 Tensor<?> queryBigramIndicesTensor,
                 Tensor<?> queryCharTrigramIndicesTensor,
                 Tensor<?> productUnigramIndicesTensor,
                 Tensor<?> productBigramIndicesTensor,
                 Tensor<?> productCharTrigramIndicesTensor,
                 Tensor<?> brandUnigramIndicesTensor,
                 Tensor<?> brandBigramIndicesTensor,
                 Tensor<?> brandCharTrigramIndicesTensor,
                 Tensor<?> authorUnigramIndicesTensor,
                 Tensor<?> authorBigramIndicesTensor,
                 Tensor<?> authorCharTrigramIndicesTensor,
                 Tensor<?> catUnigramIndicesTensor,
                 Tensor<?> catBigramIndicesTensor,
                 Tensor<?> catCharTrigramIndicesTensor,
                 Tensor<?> catTokensTensor,
                 Tensor<?> catInProductTensor,
                 Tensor<?> attrUnigramIndicesTensor,
                 Tensor<?> attrBigramIndicesTensor,
                 Tensor<?> attrCharTrigramIndicesTensor,
                 Tensor<?> attrTokensTensor,
                 Tensor<?> attrInProductTensor,
                 Tensor<?> freeFeaturesTensor) {
            this.queryUnigramIndicesTensor = queryUnigramIndicesTensor;
            this.queryBigramIndicesTensor = queryBigramIndicesTensor;
            this.queryCharTrigramIndicesTensor = queryCharTrigramIndicesTensor;
            this.productUnigramIndicesTensor = productUnigramIndicesTensor;
            this.productBigramIndicesTensor = productBigramIndicesTensor;
            this.productCharTrigramIndicesTensor = productCharTrigramIndicesTensor;
            this.brandUnigramIndicesTensor = brandUnigramIndicesTensor;
            this.brandBigramIndicesTensor = brandBigramIndicesTensor;
            this.brandCharTrigramIndicesTensor = brandCharTrigramIndicesTensor;
            this.authorUnigramIndicesTensor = authorUnigramIndicesTensor;
            this.authorBigramIndicesTensor = authorBigramIndicesTensor;
            this.authorCharTrigramIndicesTensor = authorCharTrigramIndicesTensor;
            this.catUnigramIndicesTensor = catUnigramIndicesTensor;
            this.catBigramIndicesTensor = catBigramIndicesTensor;
            this.catCharTrigramIndicesTensor = catCharTrigramIndicesTensor;
            this.catTokensTensor = catTokensTensor;
            this.catInProductTensor = catInProductTensor;
            this.attrUnigramIndicesTensor = attrUnigramIndicesTensor;
            this.attrBigramIndicesTensor = attrBigramIndicesTensor;
            this.attrCharTrigramIndicesTensor = attrCharTrigramIndicesTensor;
            this.attrTokensTensor = attrTokensTensor;
            this.attrInProductTensor = attrInProductTensor;
            this.freeFeaturesTensor = freeFeaturesTensor;
        }
    }

    public SemRankerPredict(String checkpoint, boolean verbose){
        this.vocab = new HashMap<>();
        this.catTokens = new HashMap<>();
        this.attrTokens = new HashMap<>();

        String vocabPath = Paths.get(checkpoint, "vocab.txt").toString();
        this.loadTokens(vocabPath, this.vocab);

        String catTokensPath = Paths.get(checkpoint, "cats.txt").toString();
        this.loadTokens(catTokensPath, this.catTokens);

        String attrTokensPath = Paths.get(checkpoint, "attrs.txt").toString();
        this.loadTokens(attrTokensPath, this.catTokens);

        JSONParser parser = new JSONParser();
        String precomputedPath = Paths.get(checkpoint, "precomputed.json").toString();
        this.precomputed = new HashMap<>();

        try{
            Object obj = parser.parse(new FileReader(precomputedPath));
            JSONObject jsonObject = (JSONObject) obj;

            for(Object k: jsonObject.keySet()){
                this.precomputed.put((String)k, new Integer[2]);
                this.precomputed.get(k)[0] = (Integer)((JSONArray)jsonObject.get(k)).get(0);
                this.precomputed.get(k)[1] = (Integer)((JSONArray)jsonObject.get(k)).get(1);
            }
        }catch (Exception e){
            e.printStackTrace();
        }

        String hyperParamsPath = Paths.get(checkpoint, "hyperparams.json").toString();
        try{
            Object obj = parser.parse(new FileReader(hyperParamsPath));
            JSONObject hyperParams = (JSONObject) obj;
            this.maxQueryLength = (int)hyperParams.get("max_query_length");
            this.maxProductNameLength = (int)hyperParams.get("max_product_name_length");
            this.maxBrandLength = (int)hyperParams.get("max_brand_length");
            this.maxAuthorLength = (int)hyperParams.get("max_author_length");
            this.maxCatLength = (int)hyperParams.get("max_cat_length");
            this.maxAttrLength = (int)hyperParams.get("max_attr_length");
            this.unknownBin = (int)hyperParams.get("unknown_bin");
        }catch (Exception e){
            e.printStackTrace();
        }

        try{
            String modelPath = Paths.get(checkpoint, "model").toString();
            SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");
            if (verbose){
                Utils.printSignature(model);
            }
            this.session = model.session();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    private InputTensors createInputTensors(){
        return null;
    }

    public float[] fit(String query, JSONObject products){
        return null;
    }
}
