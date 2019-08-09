package org.tiki.vn;

import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class ConverterTest {
    Converter converter = null;

    @Test
    public void testConvertTokens() {
        String[][] arrTokens = {
                {"zero", "one", "four"},
                {"zero", "one", "two", "three"},
                {"zero", "one", "two", "three", "five"}
        };

        int maxSeqLen = 7;

        int[][] indices = converter.convertTokens(arrTokens, maxSeqLen);

        assertArrayEquals(indices[0], new int[]{0, 1, 4, 9, 9, 9, 9});

    }

    @Before
    public void beforeEachTestMethod(){
        Map<String, Integer> token2Idx = new HashMap<>();

        token2Idx.put("zero", 0);
        token2Idx.put("one", 1);
        token2Idx.put("two", 2);
        token2Idx.put("three", 3);
        token2Idx.put("four", 4);

        converter = new Converter(token2Idx, token2Idx.size()+4, 4);
    }
}