package org.tiki.vn;

import org.junit.Before;
import org.junit.Test;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.io.File;


import static org.junit.Assert.*;

public class SemRankerPredictTest {
    String resourcesDirectory = "resources";
    SemRankerPredict predictor = null;
    String[] actualFreeFeatures = {
            "reviews", "rating", "sales_monthly", "sales_yearly", "support_p2h_delivery"
    };
    String[] queries = {
            "thủy hử liên hoàn truyện",
            "not giving a fuck",
            "nồi cơm điện tư toshiba 1.8l"
    };

    JSONArray products = null;

    @Before
    public void setUp(){
        try {
            File file = new File(this.getClass().getResource("/ranker").getFile());
            predictor = new SemRankerPredict(file.getAbsolutePath(), true);


            products = new JSONArray();

            JSONObject p1 = new JSONObject();
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
    }
}