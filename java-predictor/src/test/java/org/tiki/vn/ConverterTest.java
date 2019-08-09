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
        token2Idx.put("two#three", 6);
        token2Idx.put("three#four", 6);

        cat2Idx.put("10#5", 0);
        cat2Idx.put("10068#3", 1);
        cat2Idx.put("10073#4", 2);
        cat2Idx.put("10076#4", 3);
        cat2Idx.put("10077#4", 4);

        attr2Idx.put("1045#filter_tivi_screen_size_range", 0);
        attr2Idx.put("1046#filter_tivi_type", 1);
        attr2Idx.put("1049#filter_tivi_resolution", 2);
        attr2Idx.put("1055#filter_baby_diaper_weight", 3);
        attr2Idx.put("1056#filter_baby_diaper_size", 4);

        converter = new Converter(
                token2Idx, cat2Idx, attr2Idx,
                token2Idx.size()+unknownBin, cat2Idx.size(), attr2Idx.size(),
                unknownBin);
        int zeroIdx = token2Idx.size() + unknownBin;
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
                        token2Idx.size()+Hash.token2UnknownIdx("five", unknownBin), zeroIdx, zeroIdx});

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
        assertArrayEquals(nGramIndices.unigramsIndices[0], new int[]{0, 1, 2, 3, 4});

    }

    @Test
    public void testConvertCats() {
    }

    @Test
    public void testConvertAttrs() {
    }
}