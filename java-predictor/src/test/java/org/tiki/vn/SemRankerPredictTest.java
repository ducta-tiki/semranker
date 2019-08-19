package org.tiki.vn;

import org.junit.Before;
import org.junit.Test;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.io.File;


import static org.junit.Assert.*;

public class SemRankerPredictTest {
    SemRankerPredict predictor = null;
    String[] actualFreeFeatures = {
            "reviews", "rating", "sales_monthly", "sales_yearly", "support_p2h_delivery"
    };
    String[] queries = {
            "thủy hử liên hoàn truyện",
            "not giving a fuck",
            "nồi cơm điện tư toshiba 1.8l"
    };

    JSONArray q1Products = null;
    JSONArray q2Products = null;

    @Before
    public void setUp(){
        try {
            File file = new File(this.getClass().getResource("/ranker").getFile());
            predictor = new SemRankerPredict(file.getAbsolutePath(), true);


            q1Products = new JSONArray();
            q2Products = new JSONArray();
            JSONObject p = new JSONObject();

            p.put("name", "Thủy hử liên hoàn họa truyện");
            p.put("brand", "");
            p.put("author", "Thi Nại Am");
            p.put("attributes", "");
            p.put("categories", "8322#2#Nhà Sách Tiki|316#3#Sách tiếng Việt |839#4#Sách văn học|842#5#Tác phẩm kinh điển");
            p.put("reviews", 3.);
            p.put("rating", 80.);
            p.put("sales_monthly", 58.);
            p.put("sales_yearly", 0.);
            p.put("support_p2h_delivery", 1.);
            q1Products.add(p);

            p = new JSONObject();
            p.put("name", "Thủy Hử (Tập 1)");
            p.put("brand", "");
            p.put("author", "Thi Nại Am");
            p.put("attributes", "");
            p.put("categories", "8322#2#Nhà Sách Tiki|316#3#Sách tiếng Việt |839#4#Sách văn học|842#5#Tác phẩm kinh điển");
            p.put("reviews", 3.);
            p.put("rating", 80.);
            p.put("sales_monthly", 7.);
            p.put("sales_yearly", 0.);
            p.put("support_p2h_delivery", 1.);
            q1Products.add(p);

            p = new JSONObject();
            p.put("name", "The Subtle Art of Not Giving A F*ck");
            p.put("brand", "");
            p.put("author", "Mark Manson");
            p.put("attributes", "");
            p.put("categories", "8322#2#Nhà Sách Tiki|320#3#English Books|614#4#How-to - Self Help|9902#5#Motivational");
            p.put("reviews", 26.);
            p.put("rating", 90.);
            p.put("sales_monthly", 586);
            p.put("sales_yearly", 762);
            p.put("support_p2h_delivery", 1);
            q2Products.add(p);

            p = new JSONObject();
            p.put("name", "The Subtle Art of Not Giving A F*ck");
            p.put("brand", "");
            p.put("author", "Mark Manson");
            p.put("attributes", "");
            p.put("categories", "320#3#English Books|870#4#Sách kỹ năng sống");
            p.put("reviews", 0.);
            p.put("rating", 0.);
            p.put("sales_monthly", 3.);
            p.put("sales_yearly", 0.);
            p.put("support_p2h_delivery", 1.);
            q2Products.add(p);


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testLoadPrecomputed() {
        assertArrayEquals(predictor.featureKeys, actualFreeFeatures);
    }

    @Test
    public void testFit() {
        float[] scores = predictor.fit(queries[0], q1Products);
        System.out.println(scores[0]);
        System.out.println(scores[1]);

        assertTrue(scores[0] > scores[1]);
        scores = predictor.fit(queries[1], q2Products);
        System.out.println(scores[0]);
        System.out.println(scores[1]);
        assertTrue(scores[0] > scores[1]);
    }
}