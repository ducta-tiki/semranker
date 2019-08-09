package org.tiki.vn;

public class VnLangUtils {
    public static String vnLowercases = "aạảàáãâậẩầấẫăặẳằắẵbcdđeẹẻèéẽêệểềếễfghiịỉìíĩjklmnoọỏòóõôộổồốỗơợởờớỡpqrstuụủùúũưựửừứữvwxyỵỷỳýỹz";
    public static String vnUppercases = "AẠẢÀÁÃÂẬẨẦẤẪĂẶẮẰẮẴBCDĐEẸẺÈÉẼÊỆỂỀẾỄFGHIỊỈÌÍĨJKLMNOỌỎÒÓÕÔỘỔỒỐỖƠỢỞỜỚỠPQRSTUỤỦÙÚŨƯỰỬỪỨỮVWXYỴỶỲÝỸZ";
    public static String [][] vnComposite =  {{"à","à"},{"á","á"},{"ã","ã"},{"ả","ả"},{"ạ","ạ"},{"è","è"},{"é","é"},{"ẽ","ẽ"},
            {"ẻ","ẻ"},{"ẹ","ẹ"},{"ì","ì"},{"í","í"},{"ĩ","ĩ"},{"ỉ","ỉ"},{"ị","ị"},{"ò","ò"},
            {"ó","ó"},{"õ","õ"},{"ỏ","ỏ"},{"ọ","ọ"},{"ờ","ờ"},{"ớ","ớ"},{"ỡ","ỡ"},{"ở","ở"},
            {"ợ","ợ"},{"ù","ù"},{"ú","ú"},{"ũ","ũ"},{"ủ","ủ"},{"ụ","ụ"},{"ỳ","ỳ"},{"ý","ý"},
            {"ỹ","ỹ"},{"ỷ","ỷ"},{"ỵ","ỵ"},{"â","â"},{"ầ","ầ"},{"ấ","ấ"},{"ẫ","ẫ"},{"ẩ","ẩ"},
            {"ậ","ậ"},{"ằ","ằ"},{"ắ","ắ"},{"ẵ","ẵ"},{"ẳ","ẳ"},{"ặ","ặ"},{"ừ","ừ"},{"ứ","ứ"},
            {"ữ","ữ"},{"ử","ử"},{"ự","ự"},{"ê","ê"},{"ề","ề"},{"ế","ế"},{"ễ","ễ"},{"ể","ể"},
            {"ệ","ệ"},{"ô","ô"},{"ồ","ồ"},{"ố","ố"},{"ỗ","ỗ"},{"ổ","ổ"},{"ộ","ộ"}};
    public static String [] toneMarks = {"aáảàãạâấẩẫầậăắẳẵằặ", "dđ", "eẹẻèéẽêệểềếễ", "iịỉìíĩ", "oọỏòóõơợởờớỡôổốồỗộ", "uụủùúũưựửừứữ", "yỵỷỳýỹ", "AÁẢÀÃẠÂẤẨẪẦẬĂẮẮẴẰẶ", "DĐ","IỊỈÌÍĨ", "OỌỎÒÓÕƠỢỞỜỚỠÔỔỐỒỖỘ", "UỤỦÙÚŨƯỰỬỪỨỮ","YỴỶỲÝỸ"};

    public static String removeComposite(String text){
        int N = vnComposite.length;
        String temp = text;
        for (String[] strings : vnComposite) {
            temp = temp.replace(strings[0], strings[1]);
        }
        return temp;
    }

    public static String toLowerCase(String text){
        int N=vnLowercases.length();
        String temp = text;
        for (int i=0; i<N; i++){
            temp = temp.replace(vnUppercases.charAt(i),vnLowercases.charAt(i));
        }
        return temp;
    }

    public static String toUpperCase(String text){
        int N=vnLowercases.length();
        String temp = text;
        for (int i=0; i<N; i++){
            temp = temp.replace(vnLowercases.charAt(i), vnUppercases.charAt(i));
        }
        return temp;
    }

    public static String removeToneMarks(String text){
        int N = toneMarks.length;
        String temp = text;
        for (String toneMark : toneMarks) {
            for (int j = 1; j < toneMark.length(); j++) {
                temp = temp.replace(toneMark.charAt(j), toneMark.charAt(0));
            }
        }
        return temp;
    }

    public static String strip(String text, String chars) {
        int i = 0;
        int j = text.length() - 1;

        while(chars.indexOf(text.charAt(i)) != -1) {i++;}
        while(chars.indexOf(text.charAt(j)) != -1) {j--;}

        return text.substring(i, j);
    }
}
