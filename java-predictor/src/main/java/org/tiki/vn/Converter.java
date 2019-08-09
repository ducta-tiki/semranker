package org.tiki.vn;

import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


public class Converter {
    private Map<String, Integer>token2Idx = null;
    private Map<String, Integer>cat2Idx = null;
    private Map<String, Integer>attr2Idx = null;
    private int zeroIdx = 0;
    private int catZeroIdx = 0;
    private int attrZeroIdx = 0;
    private int unknownBin = 0;

    public Converter(Map<String, Integer>token2Idx, Map<String, Integer>cat2Idx,
                     Map<String, Integer>attr2Idx,
                     int zeroIdx, int catZeroIdx, int attrZeroIdx, int unknownBin ){
        this.token2Idx = token2Idx;
        this.cat2Idx = cat2Idx;
        this.attr2Idx = attr2Idx;
        this.zeroIdx = zeroIdx;
        this.catZeroIdx = catZeroIdx;
        this.attrZeroIdx = attrZeroIdx;
        this.unknownBin = unknownBin;
    }

    public int[][] convertTokens(String[][] arrTokens, int maxSeqLen){
        List<int[]> arrIndices = new ArrayList<>();
        int baseUnknown = token2Idx.size();

        for(String[] tokens: arrTokens){
            List<Integer> indices = new ArrayList<>();
            String[] paddedTokens = new String[maxSeqLen];
            Arrays.fill(paddedTokens, "<zero>");

            System.arraycopy(tokens, 0, paddedTokens, 0, Math.min(tokens.length, maxSeqLen));

            for(String t: paddedTokens){
                if(this.token2Idx.containsKey(t)){
                    indices.add(this.token2Idx.get(t));
                }else if(t.equals("<zero>")){
                    indices.add(this.zeroIdx);
                }else {
                    indices.add(baseUnknown + Hash.token2UnknownIdx(t, this.unknownBin));
                }
            }
            arrIndices.add(ArrayUtils.toPrimitive(
                    indices.toArray(new Integer[0])));
        }

        return arrIndices.toArray(new int[0][]);
    }

    public static class NGrams{
        public String[] unigrams;
        public String[] bigrams;
        public String[] charTrigrams;

        NGrams(String[] unigrams, String[] bigrams, String[] charTrigrams){
            this.unigrams = unigrams;
            this.bigrams = bigrams;
            this.charTrigrams = charTrigrams;
        }
    }

    public NGrams createNGrams(String s){
        List<String> unigrams = new ArrayList<>();
        List<String> biggrams = new ArrayList<>();
        List<String> charTrigrams = new ArrayList<>();
        String[] tokens = s.split(" ");
        for(String t: tokens){
            if(t.length() > 0){
                unigrams.add(t);
                String z = "#" + t + "#";
                for(int i=0; i<Math.max(z.length()-2, 1); i++){
                    String v = z.substring(i,i+3);
                    charTrigrams.add(v);
                }
            }
        }
        for(int i=0; i<Math.max(tokens.length-1, 0); i++){
            String t = String.format("%s#%s", tokens[i], tokens[i+1]);
            biggrams.add(t);
        }

        return new NGrams(
                unigrams.toArray(new String[0]),
                biggrams.toArray(new String[0]),
                charTrigrams.toArray(new String[0])
        );
    }

    public static class NGramIndices{
        public int[][] unigramsIndices;
        public int[][] bigramsIndices;
        public int[][] charTrigramsIndices;

        NGramIndices(
                int[][] unigramsIndices,
                int[][] bigramsIndices,
                int[][] charTrigramsIndices){
            this.unigramsIndices = unigramsIndices;
            this.bigramsIndices = bigramsIndices;
            this.charTrigramsIndices = charTrigramsIndices;
        }
    }
    public NGramIndices convertStrings(
            String[] arrStrings, int unigramMaxSeqLen, int bigramMaxSeqLen, int charTrigramMaxSeqlLen){
        List<String[]> unigramTokens = new ArrayList<>();
        List<String[]> bigramTokens = new ArrayList<>();
        List<String[]> charTrigramTokens = new ArrayList<>();

        for(String s: arrStrings){
            NGrams nGrams = createNGrams(s);
            unigramTokens.add(nGrams.unigrams);
            bigramTokens.add(nGrams.bigrams);
            charTrigramTokens.add(nGrams.charTrigrams);
        }

        int[][] unigramIndices = convertTokens(
                unigramTokens.toArray(new String[0][]), unigramMaxSeqLen);
        int[][] bigramIndices = convertTokens(
                bigramTokens.toArray(new String[0][]), bigramMaxSeqLen);
        int[][] charTrigramIndices = convertTokens(
                charTrigramTokens.toArray(new String[0][]), charTrigramMaxSeqlLen);
        return new NGramIndices(unigramIndices, bigramIndices, charTrigramIndices);
    }

    public static class CategoryIndices{
        public int[] catIndices;
        public int[] catInProduct;
        public int[][] catUnigramIndices;
        public int[][] catBigramIndices;
        public int[][] catCharTrigramIndices;

        public CategoryIndices(
                int[] catIndices, int[] catInProduct,
                int[][] catUnigramIndices,
                int[][] catBigramIndices,
                int[][] catCharTrigramIndices){
            this.catIndices = catIndices;
            this.catInProduct = catInProduct;
            this.catUnigramIndices = catUnigramIndices;
            this.catBigramIndices = catBigramIndices;
            this.catCharTrigramIndices = catCharTrigramIndices;
        }
    }

    public CategoryIndices convertCats(
            String[] arrCats, int catUnigramMaxSeqLen,
            int catBigramMaxSeqLen, int catCharTrigramMaxSeqLen){
        List<Integer> catIndices = new ArrayList<>();
        List<Integer> catInProduct = new ArrayList<>();
        List<int[]> unigramIndices = new ArrayList<>();
        List<int[]> bigramIndices = new ArrayList<>();
        List<int[]> charTrigramIndices = new ArrayList<>();

        for(String catStr: arrCats){
            String[] zz = catStr.split("\\|");
            if(catStr.trim().length() == 0){
                catInProduct.add(1);
                catIndices.add(catZeroIdx);
                int[] z1 = new int[catUnigramMaxSeqLen];
                Arrays.fill(z1, zeroIdx);
                unigramIndices.add(z1);
                int[] z2 = new int[catBigramMaxSeqLen];
                Arrays.fill(z2, zeroIdx);
                bigramIndices.add(z2);
                int[] z3 = new int[catCharTrigramMaxSeqLen];
                Arrays.fill(z3, zeroIdx);
                charTrigramIndices.add(z3);
            }
            int count = 0;
            for(String t: zz){
                String[] spt = t.split("#");
                String catToken = spt[0] + "#" + spt[1];
                if(cat2Idx.containsKey(catToken)){
                    count += 1;
                    catIndices.add(cat2Idx.get(catToken));
                    String catName = QueryPreprocessing.preprocess(spt[spt.length-1]);
                    NGramIndices nGramIndices = convertStrings(
                            new String[]{catName}, catUnigramMaxSeqLen, catBigramMaxSeqLen, catCharTrigramMaxSeqLen);
                    unigramIndices.add(nGramIndices.unigramsIndices[0]);
                    bigramIndices.add(nGramIndices.bigramsIndices[0]);
                    charTrigramIndices.add(nGramIndices.charTrigramsIndices[0]);
                }

            }
            catInProduct.add(count);
        }

        return new CategoryIndices(
                ArrayUtils.toPrimitive(catIndices.toArray(new Integer[0])),
                ArrayUtils.toPrimitive(catInProduct.toArray(new Integer[0])),
                unigramIndices.toArray(new int[0][]),
                bigramIndices.toArray(new int[0][]),
                charTrigramIndices.toArray(new int[0][])
        );
    }

    public static class AttributeIndices{
        public int[] attrIndices;
        public int[] attrInProduct;
        public int[][] attrUnigramIndices;
        public int[][] attrBigramIndices;
        public int[][] attrCharTrigramIndices;

        public AttributeIndices(
                int[] attrIndices, int[] attrInProduct,
                int[][] attrUnigramIndices,
                int[][] attrBigramIndices,
                int[][] attrCharTrigramIndices){
            this.attrIndices = attrIndices;
            this.attrInProduct = attrInProduct;
            this.attrUnigramIndices = attrUnigramIndices;
            this.attrBigramIndices = attrBigramIndices;
            this.attrCharTrigramIndices = attrCharTrigramIndices;
        }
    }

    public AttributeIndices convertAttrs(
            String[] arrAttrs, int attrUnigramMaxSeqLen,
            int attrBigramMaxSeqLen, int attrCharTrigramMaxSeqLen){
        List<Integer> attrIndices = new ArrayList<>();
        List<Integer> attrInProduct = new ArrayList<>();
        List<int[]> unigramIndices = new ArrayList<>();
        List<int[]> bigramIndices = new ArrayList<>();
        List<int[]> charTrigramIndices = new ArrayList<>();

        for(String attrStr: arrAttrs){
            String[] zz = attrStr.split("\\|");
            if(attrStr.trim().length() == 0){
                attrInProduct.add(1);
                attrIndices.add(attrZeroIdx);
                int[] z1 = new int[attrUnigramMaxSeqLen];
                Arrays.fill(z1, zeroIdx);
                unigramIndices.add(z1);
                int[] z2 = new int[attrBigramMaxSeqLen];
                Arrays.fill(z2, zeroIdx);
                bigramIndices.add(z2);
                int[] z3 = new int[attrCharTrigramMaxSeqLen];
                Arrays.fill(z3, zeroIdx);
                charTrigramIndices.add(z3);
            }
            int count = 0;
            for(String t: zz){
                String[] spt = t.split("#");
                String catToken = spt[0] + "#" + spt[1];
                if(attr2Idx.containsKey(catToken)){
                    count += 1;
                    attrIndices.add(attr2Idx.get(catToken));
                    String catName = QueryPreprocessing.preprocess(spt[spt.length-1]);
                    NGramIndices nGramIndices = convertStrings(
                            new String[]{catName}, attrUnigramMaxSeqLen, attrBigramMaxSeqLen, attrCharTrigramMaxSeqLen);
                    unigramIndices.add(nGramIndices.unigramsIndices[0]);
                    bigramIndices.add(nGramIndices.bigramsIndices[0]);
                    charTrigramIndices.add(nGramIndices.charTrigramsIndices[0]);
                }

            }
            attrInProduct.add(count);
        }

        return new AttributeIndices(
                ArrayUtils.toPrimitive(attrIndices.toArray(new Integer[0])),
                ArrayUtils.toPrimitive(attrInProduct.toArray(new Integer[0])),
                unigramIndices.toArray(new int[0][]),
                bigramIndices.toArray(new int[0][]),
                charTrigramIndices.toArray(new int[0][])
        );
    }
}
