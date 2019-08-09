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

    @Before
    public void setUp(){
        token2Idx = new HashMap<>();

        token2Idx.put("zero", 0);
        token2Idx.put("one", 1);
        token2Idx.put("two", 2);
        token2Idx.put("three", 3);
        token2Idx.put("four", 4);

        converter = new Converter(token2Idx, token2Idx.size()+unknownBin, unknownBin);
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
        assertArrayEquals(indices[0], new int[]{0, 1, 4, 9, 9, 9, 9});
        assertArrayEquals(indices[1], new int[]{0, 1, 2, 3, 9, 9, 9});
        assertArrayEquals(indices[2],
                new int[]{0, 1, 2, 3, token2Idx.size()+Hash.token2UnknownIdx("five", unknownBin), 9, 9});

    }

    @Test
    public void createNGrams() {
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
}