package org.tiki.vn;

import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


public class Converter {
    private Map<String, Integer>token2Idx = null;
    private int zeroIdx = 0;
    private int unknownBin = 0;

    public Converter(Map<String, Integer>token2Idx, int zeroIdx, int unknownBin ){
        this.token2Idx = token2Idx;
        this.zeroIdx = zeroIdx;
        this.unknownBin = unknownBin;
    }

    public int[][] convertTokens(String[][] arrTokens, int maxSeqLen){
        List<int[]> arrIndices = new ArrayList<>();
        int baseUnknown = token2Idx.size();

        for(String[] tokens: arrTokens){
            List<Integer> indices = new ArrayList<>();
            String[] truncatedTokens = Arrays.copyOfRange(tokens, 0, maxSeqLen);
            String[] zeroPadding = new String[Math.max(token2Idx.size(), maxSeqLen) - token2Idx.size()];
            Arrays.fill(zeroPadding, "<zero>");
            truncatedTokens = (String[])ArrayUtils.add(truncatedTokens, zeroPadding);
            for(String t: truncatedTokens){
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
}
