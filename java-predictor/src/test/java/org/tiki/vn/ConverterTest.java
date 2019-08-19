package org.tiki.vn;

import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class ConverterTest {
    Converter converter = null;
    int unknownBin = 4;
    Map<String, Integer> token2Idx = null;
    Map<String, Integer> cat2Idx = null;
    Map<String, Integer> attr2Idx = null;
    int zeroIdx = 0;

    int fiveUknownIdx = 0;
    int threeFiveUnknownIdx = 0;

    @Before
    public void setUp(){
        token2Idx = new HashMap<>();
        cat2Idx = new HashMap<>();
        attr2Idx = new HashMap<>();

        token2Idx.put("zero", 0);
        token2Idx.put("one", 1);
        token2Idx.put("two", 2);
        token2Idx.put("three", 3);
        token2Idx.put("four", 4);

        token2Idx.put("zero#one", 5);
        token2Idx.put("one#two", 6);
        token2Idx.put("two#three", 7);
        token2Idx.put("three#four", 8);

        token2Idx.put("#ze", 9);
        token2Idx.put("zer", 10);
        token2Idx.put("ero", 11);
        token2Idx.put("ro#", 12);
        token2Idx.put("#on", 13);
        token2Idx.put("ne#", 14);
        token2Idx.put("#tw", 15);
        token2Idx.put("wo#", 16);
        token2Idx.put("#th", 17);
        token2Idx.put("thr", 18);
        token2Idx.put("hre", 19);
        token2Idx.put("ree", 20);
        token2Idx.put("ee#", 21);
        token2Idx.put("#fo", 22);
        token2Idx.put("fou", 23);
        token2Idx.put("our", 24);
        token2Idx.put("ur#", 25);

        cat2Idx.put("1#2", 0);
        cat2Idx.put("2#3", 1);
        cat2Idx.put("3#4", 2);
        cat2Idx.put("5#6", 3);

        attr2Idx.put("1#filter_1", 0);
        attr2Idx.put("2#filter_2", 1);
        attr2Idx.put("3#filter_3", 2);
        attr2Idx.put("4#filter_4", 3);
        attr2Idx.put("5#filter_5", 4);

        converter = new Converter(
                token2Idx, cat2Idx, attr2Idx,
                token2Idx.size()+unknownBin,
                cat2Idx.size(), attr2Idx.size(),
                unknownBin);
        zeroIdx = token2Idx.size() + unknownBin;

        fiveUknownIdx = toUnknownIdx("five");
        threeFiveUnknownIdx = toUnknownIdx("three#five");
    }

    private int toUnknownIdx(String s){
        return token2Idx.size()+Hash.token2UnknownIdx(s, unknownBin);
    }

    @Test
    public void testConvertTokens() {
        String[][] arrTokens = {
                {"zero", "one", "four"},
                {"zero", "one", "two", "three"},
                {"zero", "one", "two", "three", "five"}
        };

        int maxSeqLen = 7;

        int[][] indices = converter.convertTokens(arrTokens, maxSeqLen);

        assertEquals(indices.length, 3);
        assertEquals(indices[0].length, maxSeqLen);
        assertArrayEquals(indices[0], new int[]{
                token2Idx.get("zero"), token2Idx.get("one"), token2Idx.get("four"),
                zeroIdx, zeroIdx, zeroIdx, zeroIdx});
        assertEquals(indices[1].length, maxSeqLen);
        assertArrayEquals(indices[1], new int[]{
                token2Idx.get("zero"), token2Idx.get("one"), token2Idx.get("two"), token2Idx.get("three"),
                zeroIdx, zeroIdx, zeroIdx});
        assertEquals(indices[2].length, maxSeqLen);
        assertArrayEquals(indices[2],
                new int[]{token2Idx.get("zero"), token2Idx.get("one"), token2Idx.get("two"), token2Idx.get("three"),
                        fiveUknownIdx, zeroIdx, zeroIdx});

    }

    @Test
    public void testCreateNGrams() {
        Converter.NGrams nGrams = converter.createNGrams("hello world"); // #he, hel, ell, llo, lo#, #wo, wor, orl, rld, ld#

        assertEquals(nGrams.unigrams.length, 2);
        assertEquals(nGrams.unigrams[0], "hello");
        assertEquals(nGrams.unigrams[1], "world");
        assertEquals(nGrams.bigrams.length, 1);
        assertEquals(nGrams.bigrams[0], "hello#world");
        assertEquals(nGrams.charTrigrams.length, 10);
        String[] actualCharTrigrams = {
                "#he", "hel", "ell", "llo", "lo#", "#wo", "wor", "orl", "rld", "ld#"
        };
        assertArrayEquals(nGrams.charTrigrams, actualCharTrigrams);
    }

    @Test
    public void testConvertStrings() {
        Converter.NGramIndices nGramIndices = converter.convertStrings(
                new String[]{"zero one two three four five", "zero one two"},
                5, 5, 25);

        assertEquals(nGramIndices.unigramsIndices.length, 2);
        assertArrayEquals(nGramIndices.unigramsIndices[0], new int[]{
                token2Idx.get("zero"), token2Idx.get("one"), token2Idx.get("two"), token2Idx.get("three"), token2Idx.get("four")});
        assertArrayEquals(nGramIndices.unigramsIndices[1], new int[]{
                token2Idx.get("zero"), token2Idx.get("one"), token2Idx.get("two"), zeroIdx, zeroIdx
        });

        assertEquals(nGramIndices.bigramsIndices.length, 2);
        assertArrayEquals(nGramIndices.bigramsIndices[0], new int[]{
                token2Idx.get("zero#one"), token2Idx.get("one#two"), token2Idx.get("two#three"),
                token2Idx.get("three#four"), toUnknownIdx("four#five")
        });
        assertArrayEquals(nGramIndices.bigramsIndices[1], new int[]{
                token2Idx.get("zero#one"), token2Idx.get("one#two"), zeroIdx, zeroIdx, zeroIdx
        });

        assertEquals(nGramIndices.charTrigramsIndices.length, 2);
        assertArrayEquals(nGramIndices.charTrigramsIndices[0], new int[]{
                token2Idx.get("#ze"), token2Idx.get("zer"), token2Idx.get("ero"), token2Idx.get("ro#"),
                token2Idx.get("#on"), token2Idx.get("one"), token2Idx.get("ne#"),
                token2Idx.get("#tw"), token2Idx.get("two"), token2Idx.get("wo#"),
                token2Idx.get("#th"), token2Idx.get("thr"), token2Idx.get("hre"), token2Idx.get("ree"), token2Idx.get("ee#"),
                token2Idx.get("#fo"), token2Idx.get("fou"), token2Idx.get("our"), token2Idx.get("ur#"),
                toUnknownIdx("#fi"), toUnknownIdx("fiv"), toUnknownIdx("ive"), toUnknownIdx("ve#"),
                zeroIdx, zeroIdx
        });

        assertArrayEquals(nGramIndices.charTrigramsIndices[1], new int[]{
                token2Idx.get("#ze"), token2Idx.get("zer"), token2Idx.get("ero"), token2Idx.get("ro#"),
                token2Idx.get("#on"), token2Idx.get("one"), token2Idx.get("ne#"),
                token2Idx.get("#tw"), token2Idx.get("two"), token2Idx.get("wo#"),
                zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx,
                zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx
        });
    }

    @Test
    public void testConvertCats() {
        Converter.CategoryIndices categoryIndices = converter.convertCats(
                new String[]{
                        "1#2#zero|2#3#two",
                        "3#4#three four|5#6#three five"
                }, 5, 5, 10
        );
        int[] actualCatIndices = {0, 1, 2, 3};
        assertArrayEquals(categoryIndices.catIndices, actualCatIndices);

        int[] actualCatInProduct = {2, 2};
        assertArrayEquals(categoryIndices.catInProduct, actualCatInProduct);

        int[][] actualCatUnigramIndices = {
                {token2Idx.get("zero"), zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("two"), zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("three"), token2Idx.get("four"), zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("three"), fiveUknownIdx, zeroIdx, zeroIdx, zeroIdx}
        };
        assertArrayEquals(categoryIndices.catUnigramIndices[0], actualCatUnigramIndices[0]);
        assertArrayEquals(categoryIndices.catUnigramIndices[1], actualCatUnigramIndices[1]);
        assertArrayEquals(categoryIndices.catUnigramIndices[2], actualCatUnigramIndices[2]);
        assertArrayEquals(categoryIndices.catUnigramIndices[3], actualCatUnigramIndices[3]);

        int[][] actualCatBigramIndices = {
                {zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("three#four"), zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {threeFiveUnknownIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx}
        };
        assertArrayEquals(categoryIndices.catBigramIndices[0], actualCatBigramIndices[0]);
        assertArrayEquals(categoryIndices.catBigramIndices[1], actualCatBigramIndices[1]);
        assertArrayEquals(categoryIndices.catBigramIndices[2], actualCatBigramIndices[2]);
        assertArrayEquals(categoryIndices.catBigramIndices[3], actualCatBigramIndices[3]);

        int[][] actualCatCharTrigramIndices = {
                {token2Idx.get("#ze"), token2Idx.get("zer"), token2Idx.get("ero"), token2Idx.get("ro#"),
                        zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("#tw"), token2Idx.get("two"), token2Idx.get("wo#"),
                        zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("#th"), token2Idx.get("thr"), token2Idx.get("hre"), token2Idx.get("ree"), token2Idx.get("ee#"),
                        token2Idx.get("#fo"), token2Idx.get("fou"), token2Idx.get("our"), token2Idx.get("ur#"), zeroIdx},
                {token2Idx.get("#th"), token2Idx.get("thr"), token2Idx.get("hre"), token2Idx.get("ree"), token2Idx.get("ee#"),
                        toUnknownIdx("#fi"), toUnknownIdx("fiv"), toUnknownIdx("ive"), toUnknownIdx("ve#"),
                        zeroIdx
                }
        };

        assertArrayEquals(categoryIndices.catCharTrigramIndices[0], actualCatCharTrigramIndices[0]);
        assertArrayEquals(categoryIndices.catCharTrigramIndices[1], actualCatCharTrigramIndices[1]);
        assertArrayEquals(categoryIndices.catCharTrigramIndices[2], actualCatCharTrigramIndices[2]);
        assertArrayEquals(categoryIndices.catCharTrigramIndices[3], actualCatCharTrigramIndices[3]);
    }

    @Test
    public void testConvertAttrs() {
        Converter.AttributeIndices attributeIndices = converter.convertAttrs(
                new String[]{
                        "1#filter_1#zero|2#filter_2#two",
                        "3#filter_3#three four|4#filter_4#three five"
                }, 5, 5, 10
        );
        int[] actualAttrIndices = {0, 1, 2, 3};
        assertArrayEquals(attributeIndices.attrIndices, actualAttrIndices);

        int[] actualCatInProduct = {2, 2};
        assertArrayEquals(attributeIndices.attrInProduct, actualCatInProduct);

        int[][] actualAttrUnigramIndices = {
                {token2Idx.get("zero"), zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("two"), zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("three"), token2Idx.get("four"), zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("three"), fiveUknownIdx, zeroIdx, zeroIdx, zeroIdx}
        };
        assertArrayEquals(attributeIndices.attrUnigramIndices[0], actualAttrUnigramIndices[0]);
        assertArrayEquals(attributeIndices.attrUnigramIndices[1], actualAttrUnigramIndices[1]);
        assertArrayEquals(attributeIndices.attrUnigramIndices[2], actualAttrUnigramIndices[2]);
        assertArrayEquals(attributeIndices.attrUnigramIndices[3], actualAttrUnigramIndices[3]);

        int[][] actualAttrBigramIndices = {
                {zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("three#four"), zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {threeFiveUnknownIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx}
        };
        assertArrayEquals(attributeIndices.attrBigramIndices[0], actualAttrBigramIndices[0]);
        assertArrayEquals(attributeIndices.attrBigramIndices[1], actualAttrBigramIndices[1]);
        assertArrayEquals(attributeIndices.attrBigramIndices[2], actualAttrBigramIndices[2]);
        assertArrayEquals(attributeIndices.attrBigramIndices[3], actualAttrBigramIndices[3]);

        int[][] actualAttrCharTrigramIndices = {
                {token2Idx.get("#ze"), token2Idx.get("zer"), token2Idx.get("ero"), token2Idx.get("ro#"),
                        zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("#tw"), token2Idx.get("two"), token2Idx.get("wo#"),
                        zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx, zeroIdx},
                {token2Idx.get("#th"), token2Idx.get("thr"), token2Idx.get("hre"), token2Idx.get("ree"), token2Idx.get("ee#"),
                        token2Idx.get("#fo"), token2Idx.get("fou"), token2Idx.get("our"), token2Idx.get("ur#"), zeroIdx},
                {token2Idx.get("#th"), token2Idx.get("thr"), token2Idx.get("hre"), token2Idx.get("ree"), token2Idx.get("ee#"),
                        toUnknownIdx("#fi"), toUnknownIdx("fiv"), toUnknownIdx("ive"), toUnknownIdx("ve#"),
                        zeroIdx
                }
        };

        assertArrayEquals(attributeIndices.attrCharTrigramIndices[0], actualAttrCharTrigramIndices[0]);
        assertArrayEquals(attributeIndices.attrCharTrigramIndices[1], actualAttrCharTrigramIndices[1]);
        assertArrayEquals(attributeIndices.attrCharTrigramIndices[2], actualAttrCharTrigramIndices[2]);
        assertArrayEquals(attributeIndices.attrCharTrigramIndices[3], actualAttrCharTrigramIndices[3]);
    }

    @Test
    public void testConvertFreeFeatures() {
    }
}
