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
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class SemRankerPredict {
    String QUERY_UNIGRAM_INDICES = "query_unigram_indices:0";
    String QUERY_BIGRAM_INDICES = "query_bigram_indices:0";
    String QUERY_CHAR_TRIGRAM_INDICES = "query_char_trigram_indices:0";
    String PRODUCT_UNIGRAM_INDICES = "product_unigram_indices:0";
    String PRODUCT_BIGRAM_INDICES = "product_bigram_indices:0";
    String PRODUCT_CHAR_TRIGRAM_INDICES = "product_char_trigram_indices:0";
    String BRAND_UNIGRAM_INDICES = "brand_unigram_indices:0";
    String BRAND_BIGRAM_INDICES = "brand_bigram_indices:0";
    String BRAND_CHAR_TRIGRAM_INDICES = "brand_char_trigram_indices:0";
    String AUTHOR_UNIGRAM_INDICES = "author_unigram_indices:0";
    String AUTHOR_BIGRAM_INDICES = "author_bigram_indices:0";
    String AUTHOR_CHAR_TRIGRAM_INDICES = "author_char_trigram_indices:0";
    String CAT_UNIGRAM_INDICES = "cat_unigram_indices:0";
    String CAT_BIGRAM_INDICES = "cat_bigram_indices:0";
    String CAT_CHAR_TRIGRAM_INDICES = "cat_char_trigram_indices:0";
    String CAT_TOKENS = "cat_tokens:0";
    String CATS_IN_PRODUCT = "cats_in_product:0";
    String ATTR_UNIGRAM_INDICES = "attr_unigram_indices:0";
    String ATTR_BIGRAM_INDICES = "attr_bigram_indices:0";
    String ATTR_CHAR_TRIGRAM_INDICES = "attr_char_trigram_indices:0";
    String ATTR_TOKENS = "attr_tokens:0";
    String ATTRS_IN_PRODUCT = "attrs_in_product:0";
    String FREE_FEATURES = "free_features:0";
    String SCORE = "score:0";

    Map<String, Integer> vocab = null;
    Map<String, Integer> catTokens = null;
    Map<String, Integer> attrTokens = null;
    Map<String, float[]> precomputed = null;
    String[] featureKeys = null;
    float[] precomputedMins = null;
    float[] precomputedMaxs = null;
    int maxQueryLength = 0;
    int maxProductNameLength = 0;
    int maxBrandLength = 0;
    int maxAuthorLength = 0;
    int maxCatLength = 0;
    int maxAttrLength = 0;
    int unknownBin = 0;
    int tokenZeroIdx = 0;
    int catZeroIdx = 0;
    int attrZeroIdx = 0;
    Converter converter =null;

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
        public Tensor<?> queryUnigramIndicesTensor = null;
        public Tensor<?> queryBigramIndicesTensor = null;
        public Tensor<?> queryCharTrigramIndicesTensor = null;
        public Tensor<?> productUnigramIndicesTensor = null;
        public Tensor<?> productBigramIndicesTensor = null;
        public Tensor<?> productCharTrigramIndicesTensor = null;
        public Tensor<?> brandUnigramIndicesTensor = null;
        public Tensor<?> brandBigramIndicesTensor = null;
        public Tensor<?> brandCharTrigramIndicesTensor = null;
        public Tensor<?> authorUnigramIndicesTensor = null;
        public Tensor<?> authorBigramIndicesTensor = null;
        public Tensor<?> authorCharTrigramIndicesTensor = null;
        public Tensor<?> catUnigramIndicesTensor = null;
        public Tensor<?> catBigramIndicesTensor = null;
        public Tensor<?> catCharTrigramIndicesTensor = null;
        public Tensor<?> catTokensTensor = null;
        public Tensor<?> catInProductTensor = null;
        public Tensor<?> attrUnigramIndicesTensor = null;
        public Tensor<?> attrBigramIndicesTensor = null;
        public Tensor<?> attrCharTrigramIndicesTensor = null;
        public Tensor<?> attrTokensTensor = null;
        public Tensor<?> attrInProductTensor = null;
        public Tensor<?> freeFeaturesTensor = null;

        public InputTensors(){}
    }

    private InputTensors createInputTensors(
            String[] queries, String[] productNames,
            String[] brands, String[] authors, String[] categories,
            String[] attributes, float[][] freeFeatures){

        Converter.NGramIndices queriesNGramIndices = converter.convertStrings(
                queries, this.maxQueryLength, this.maxQueryLength, this.maxQueryLength*5);
        Converter.NGramIndices productNamesNGramIndices = converter.convertStrings(
                productNames, this.maxProductNameLength, this.maxProductNameLength,
                this.maxProductNameLength*5);
        Converter.NGramIndices brandsNGramIndices = converter.convertStrings(
                brands, this.maxBrandLength, this.maxBrandLength, this.maxBrandLength*5);
        Converter.NGramIndices authorsNGramIndices = converter.convertStrings(
                authors, this.maxAuthorLength, this.maxAuthorLength, this.maxAuthorLength*5);
        Converter.CategoryIndices categoryIndices = converter.convertCats(
                categories, this.maxCatLength, this.maxCatLength, this.maxCatLength*5);
        Converter.AttributeIndices attributeIndices = converter.convertAttrs(
                attributes, this.maxAttrLength, this.maxAttrLength, this.maxAttrLength*5);
        float[][] normalizedFreeFeatures = converter.convertFreeFeatures(freeFeatures, precomputedMins, precomputedMaxs);

        InputTensors inputTensors = new InputTensors();

        inputTensors.queryUnigramIndicesTensor = Tensor.create(
                queriesNGramIndices.unigramsIndices, Integer.class);
        inputTensors.queryBigramIndicesTensor = Tensor.create(
                queriesNGramIndices.bigramsIndices, Integer.class);
        inputTensors.queryCharTrigramIndicesTensor = Tensor.create(
                queriesNGramIndices.charTrigramsIndices, Integer.class);

        inputTensors.productUnigramIndicesTensor = Tensor.create(
                productNamesNGramIndices.unigramsIndices, Integer.class);
        inputTensors.productBigramIndicesTensor = Tensor.create(
                productNamesNGramIndices.bigramsIndices, Integer.class);
        inputTensors.productCharTrigramIndicesTensor = Tensor.create(
                productNamesNGramIndices.charTrigramsIndices, Integer.class);

        inputTensors.brandUnigramIndicesTensor =
                Tensor.create(brandsNGramIndices.unigramsIndices, Integer.class);
        inputTensors.brandBigramIndicesTensor =
                Tensor.create(brandsNGramIndices.bigramsIndices, Integer.class);
        inputTensors.brandCharTrigramIndicesTensor =
                Tensor.create(brandsNGramIndices.charTrigramsIndices, Integer.class);

        inputTensors.authorUnigramIndicesTensor =
                Tensor.create(authorsNGramIndices.unigramsIndices, Integer.class);
        inputTensors.authorBigramIndicesTensor =
                Tensor.create(authorsNGramIndices.bigramsIndices, Integer.class);
        inputTensors.authorCharTrigramIndicesTensor =
                Tensor.create(authorsNGramIndices.charTrigramsIndices, Integer.class);

        inputTensors.catUnigramIndicesTensor = Tensor.create(categoryIndices.catUnigramIndices, Integer.class);
        inputTensors.catBigramIndicesTensor = Tensor.create(categoryIndices.catBigramIndices, Integer.class);
        inputTensors.catCharTrigramIndicesTensor = Tensor.create(categoryIndices.catCharTrigramIndices, Integer.class);
        inputTensors.catTokensTensor = Tensor.create(categoryIndices.catIndices, Integer.class);
        inputTensors.catInProductTensor = Tensor.create(categoryIndices.catInProduct, Integer.class);

        inputTensors.attrUnigramIndicesTensor = Tensor.create(attributeIndices.attrUnigramIndices, Integer.class);
        inputTensors.attrBigramIndicesTensor = Tensor.create(attributeIndices.attrBigramIndices, Integer.class);
        inputTensors.attrCharTrigramIndicesTensor = Tensor.create(attributeIndices.attrCharTrigramIndices, Integer.class);
        inputTensors.attrTokensTensor = Tensor.create(attributeIndices.attrIndices, Integer.class);
        inputTensors.attrInProductTensor = Tensor.create(attributeIndices.attrInProduct, Integer.class);

        inputTensors.freeFeaturesTensor = Tensor.create(normalizedFreeFeatures, Float.class);

        return inputTensors;
    }

    public SemRankerPredict(String checkpoint, boolean verbose){
        this.vocab = new HashMap<>();
        this.catTokens = new HashMap<>();
        this.attrTokens = new HashMap<>();


        String vocabPath = Paths.get(checkpoint, "vocab.txt").toString();
        this.loadTokens(vocabPath, this.vocab);
        tokenZeroIdx = this.vocab.size() + unknownBin;

        String catTokensPath = Paths.get(checkpoint, "cats.txt").toString();
        this.loadTokens(catTokensPath, this.catTokens);
        catZeroIdx = this.catTokens.size();

        String attrTokensPath = Paths.get(checkpoint, "attrs.txt").toString();
        this.loadTokens(attrTokensPath, this.catTokens);
        attrZeroIdx = this.attrTokens.size();

        JSONParser parser = new JSONParser();
        String precomputedPath = Paths.get(checkpoint, "precomputed.json").toString();
        this.precomputed = new HashMap<>();

        try{
            Object obj = parser.parse(new FileReader(precomputedPath));
            JSONObject jsonObject = (JSONObject) obj;
            featureKeys = new String[jsonObject.size()-1];

            int i = 0;
            for(Object v: (JSONArray)jsonObject.get("__key_order__")){
                featureKeys[i] = v.toString();
                i += 1;
            }

            precomputedMaxs = new float[featureKeys.length];
            precomputedMins = new float[featureKeys.length];

            for(Object k: jsonObject.keySet()){
                if(k.equals("__key_order__")) continue;
                this.precomputed.put((String)k, new float[2]);
                this.precomputed.get(k)[0] = ((Double)((JSONArray)jsonObject.get(k)).get(0)).floatValue();
                this.precomputed.get(k)[1] = ((Double)((JSONArray)jsonObject.get(k)).get(1)).floatValue();
            }

            for(i=0;i<featureKeys.length;i++){
                precomputedMins[i] = this.precomputed.get(featureKeys[i])[0];
                precomputedMaxs[i] = this.precomputed.get(featureKeys[i])[1];
            }

        }catch (Exception e){
            e.printStackTrace();
        }

        String hyperParamsPath = Paths.get(checkpoint, "hyperparams.json").toString();
        try{
            Object obj = parser.parse(new FileReader(hyperParamsPath));
            JSONObject hyperParams = (JSONObject) obj;
            this.maxQueryLength = ((Long)hyperParams.get("max_query_length")).intValue();
            this.maxProductNameLength = ((Long)hyperParams.get("max_product_name_length")).intValue();
            this.maxBrandLength = ((Long)hyperParams.get("max_brand_length")).intValue();
            this.maxAuthorLength = ((Long)hyperParams.get("max_author_length")).intValue();
            this.maxCatLength = ((Long)hyperParams.get("max_cat_length")).intValue();
            this.maxAttrLength = ((Long)hyperParams.get("max_attr_length")).intValue();
            this.unknownBin = ((Long)hyperParams.get("unknown_bin")).intValue();
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

        converter = new Converter(
                this.vocab, this.catTokens, this.attrTokens,
                tokenZeroIdx, catZeroIdx, attrZeroIdx, unknownBin);

    }


    public float[] fit(String query, JSONArray products){
        String preprocessedQuery = QueryPreprocessing.preprocess(query);
        String[] cloneQueries = new String[products.size()];
        String[] productNames = new String[products.size()];
        String[] brands = new String[products.size()];
        String[] authors = new String[products.size()];
        String[] categories = new String[products.size()];
        String[] attributes = new String[products.size()];
        float[][] freeFeatures = new float[products.size()][];

        for(int i=0; i<products.size();i++){
            JSONObject p = (JSONObject)products.get(i);
            cloneQueries[i] = preprocessedQuery;
            productNames[i] = QueryPreprocessing.preprocess(p.get("name").toString());
            brands[i] = QueryPreprocessing.preprocess(p.get("brand").toString());
            authors[i] = QueryPreprocessing.preprocess(p.get("author").toString());
            categories[i] = QueryPreprocessing.preprocess(p.get("categories").toString());
            attributes[i] = QueryPreprocessing.preprocess(p.get("attributes").toString());

            freeFeatures[i] = new float[featureKeys.length];
            for(int j=0; j < featureKeys.length; j++){
                freeFeatures[i][j] = ((Double)p.get(featureKeys[i])).floatValue();
            }
        }

        InputTensors inputs = createInputTensors(
                cloneQueries, productNames, brands, authors, categories, attributes, freeFeatures);
        List<Tensor<?>> outputs = null;

        outputs = session.runner()
                        .feed(QUERY_UNIGRAM_INDICES, inputs.queryUnigramIndicesTensor)
                        .feed(QUERY_BIGRAM_INDICES, inputs.queryBigramIndicesTensor)
                        .feed(QUERY_CHAR_TRIGRAM_INDICES, inputs.queryCharTrigramIndicesTensor)
                        .feed(PRODUCT_UNIGRAM_INDICES, inputs.productUnigramIndicesTensor)
                        .feed(PRODUCT_BIGRAM_INDICES, inputs.productBigramIndicesTensor)
                        .feed(PRODUCT_CHAR_TRIGRAM_INDICES, inputs.productCharTrigramIndicesTensor)
                        .feed(BRAND_UNIGRAM_INDICES, inputs.brandUnigramIndicesTensor)
                        .feed(BRAND_BIGRAM_INDICES, inputs.brandBigramIndicesTensor)
                        .feed(BRAND_CHAR_TRIGRAM_INDICES, inputs.brandCharTrigramIndicesTensor)
                        .feed(AUTHOR_UNIGRAM_INDICES, inputs.authorUnigramIndicesTensor)
                        .feed(AUTHOR_BIGRAM_INDICES, inputs.authorBigramIndicesTensor)
                        .feed(AUTHOR_CHAR_TRIGRAM_INDICES, inputs.authorCharTrigramIndicesTensor)
                        .feed(CAT_UNIGRAM_INDICES, inputs.catUnigramIndicesTensor)
                        .feed(CAT_BIGRAM_INDICES, inputs.catBigramIndicesTensor)
                        .feed(CAT_CHAR_TRIGRAM_INDICES, inputs.catCharTrigramIndicesTensor)
                        .feed(CAT_TOKENS, inputs.catTokensTensor)
                        .feed(CATS_IN_PRODUCT, inputs.catInProductTensor)
                        .feed(ATTR_UNIGRAM_INDICES, inputs.attrUnigramIndicesTensor)
                        .feed(ATTR_BIGRAM_INDICES, inputs.attrBigramIndicesTensor)
                        .feed(ATTR_CHAR_TRIGRAM_INDICES, inputs.attrCharTrigramIndicesTensor)
                        .feed(ATTR_TOKENS, inputs.attrTokensTensor)
                        .feed(ATTRS_IN_PRODUCT, inputs.attrInProductTensor)
                        .feed(FREE_FEATURES, inputs.freeFeaturesTensor)
                        .fetch(SCORE)
                        .run();


        Tensor<Float> predScores = (Tensor<Float>) outputs.get(0);
        float[] scores = predScores.copyTo(new float[products.size()]);

        return scores;
    }
}
