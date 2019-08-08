package org.tiki.vn;

import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.common.hash.HashFunction;
import com.google.common.hash.HashCode;


public class Hash {
    private static int murhash3_32(String token){
        HashFunction hf = Hashing.murmur3_32();
        Hasher hasher = hf.newHasher();

        for(int i=0; i< token.length();i++){
            hasher.putChar(token.charAt(i));
        }

        HashCode hc = hasher.hash();
        return hc.asInt();
    }

    public static int token2UnknownIdx(String token, int unknownBin){
        return murhash3_32(token) % unknownBin;
    }
}