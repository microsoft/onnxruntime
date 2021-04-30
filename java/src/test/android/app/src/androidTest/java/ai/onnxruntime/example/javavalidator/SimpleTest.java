package ai.onnxruntime.example.javavalidator;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import android.content.res.AssetManager;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.util.*;


@RunWith(AndroidJUnit4.class)
public class SimpleTest {

    @Test
    public void runSigmoidModelTest() throws OrtException, IOException {

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {

            try (OrtSession session = env.createSession(readModel("sigmoid.ort"), opts)) {
                String inputName = session.getInputNames().iterator().next();
                float[][][] testdata = new float[3][4][5];
                for (float[][] array2d : testdata) {
                    for (float[] array : array2d) {
                        Arrays.fill(array, (float) 0);
                    }
                }
                try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, testdata);
                    OrtSession.Result output = session.run(Collections.singletonMap(inputName, inputTensor))) {
                    float[][][] rawOutput = (float[][][]) output.get(0).getValue();
                    //expected sigmoid output is y = 1.0 / (1.0 + exp(-x))
                    float expected = (float) 0.5;
                    for (float[][] array2d : rawOutput) {
                        for (float[] array : array2d) {
                            for (float actual : array) {
                                Assert.assertEquals(actual, expected, 1e-15);
                            }
                        }
                    }
                }
            }
        }
    }


    private byte[] readModel(String fileName) throws IOException {
        AssetManager assetManager =
                InstrumentationRegistry.getInstrumentation().getContext().getAssets();
        InputStream inputStream = null;
        try {
           inputStream = assetManager.open(fileName);
        } catch (IOException e) {
            Log.e("Test", "Cannot load model from assets");
        }
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[16384];
        while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        return buffer.toByteArray();
    }
}
