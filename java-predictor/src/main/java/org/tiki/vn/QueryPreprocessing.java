package org.tiki.vn;

import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;

public class QueryPreprocessing {
    public QueryPreprocessing() {}

    public static boolean isPrintableChar(char c) {
        Character.UnicodeBlock block = Character.UnicodeBlock.of(c);
        return (!Character.isISOControl(c)) &&
                c != KeyEvent.CHAR_UNDEFINED &&
                block != null &&
                block != Character.UnicodeBlock.SPECIALS;
    }

    public static String removeUnprintableChars(String query) {
        StringBuilder pr = new StringBuilder();
        for(int i=0; i<query.length(); i++) {
            char ch = query.charAt(i);
            if(isPrintableChar(ch) || VnLangUtils.vnLowercases.indexOf(ch) != -1) {
                pr.append(ch);
            }
        }
        return pr.toString();
    }

    public static String removeLongTokens(String query, int limitLength) {
        String[] tokens = query.split(" ");
        List<String> filterTokens = new ArrayList<String>();
        for (String token : tokens) {
            if (token.length() <= limitLength) {
                filterTokens.add(token);
            }
        }
        return Utils.join(filterTokens, " ");
    }

    public static String removeYearTokens(String query) {
        String ne = query.replaceAll("20[0-9][0-9]", "");
        return ne.trim().replaceAll(" +", " ");
    }

    public static String preprocess(String query) {
        String result = query;
        result = result.replaceAll("(\\d+),(\\d+)", "$1.$2");
        result = result.replaceAll("([!?,“”【】\"':/()…\\-])", " $1 ");
        result = result.trim().replaceAll(" +", " ");
        result = VnLangUtils.toLowerCase(result);

        return result;
    }
}
