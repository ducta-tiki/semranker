package org.tiki.vn;

import org.junit.Before;
import org.junit.Test;

import java.io.File;


import static org.junit.Assert.*;

public class SemRankerPredictTest {

    String resourcesDirectory = "resources";
    SemRankerPredict predictor = null;
    String[] actualFreeFeatures = {
            "reviews", "rating", "sales_monthly", "sales_yearly", "support_p2h_delivery"
    };
    @Before
    public void setUp(){
        try {
            File file = new File(this.getClass().getResource("/ranker").getFile());
            predictor = new SemRankerPredict(file.getAbsolutePath(), true);
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