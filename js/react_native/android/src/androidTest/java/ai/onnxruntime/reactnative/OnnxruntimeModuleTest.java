// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import static com.android.dx.mockito.inline.extended.ExtendedMockito.mockitoSession;
import static org.mockito.Mockito.when;

import ai.onnxruntime.TensorInfo;
import android.util.Base64;
import androidx.test.platform.app.InstrumentationRegistry;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.CatalystInstance;
import com.facebook.react.bridge.JavaOnlyArray;
import com.facebook.react.bridge.JavaOnlyMap;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.blob.BlobModule;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoSession;

public class OnnxruntimeModuleTest {
  private ReactApplicationContext reactContext =
      new ReactApplicationContext(InstrumentationRegistry.getInstrumentation().getContext());

  private FakeBlobModule blobModule;

  @Before
  public void setUp() {
    blobModule = new FakeBlobModule(reactContext);
  }

  @Test
  public void getName() throws Exception {
    OnnxruntimeModule ortModule = new OnnxruntimeModule(reactContext);
    ortModule.blobModule = blobModule;
    String name = "Onnxruntime";
    Assert.assertEquals(ortModule.getName(), name);
  }

  @Test
  public void onnxruntime_module() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OnnxruntimeModule ortModule = new OnnxruntimeModule(reactContext);
      ortModule.blobModule = blobModule;
      String sessionKey = "";

      // test loadModel()
      {
        InputStream modelStream =
            reactContext.getResources().openRawResource(ai.onnxruntime.reactnative.test.R.raw.test_types_float);
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();

        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        int len;
        while ((len = modelStream.read(buffer)) != -1) {
          byteBuffer.write(buffer, 0, len);
        }

        byte[] modelBuffer = byteBuffer.toByteArray();

        JavaOnlyMap options = new JavaOnlyMap();
        try {
          ReadableMap resultMap = ortModule.loadModel(modelBuffer, options);
          sessionKey = resultMap.getString("key");
          ReadableArray inputNames = resultMap.getArray("inputNames");
          ReadableArray outputNames = resultMap.getArray("outputNames");

          Assert.assertEquals(inputNames.size(), 1);
          Assert.assertEquals(inputNames.getString(0), "input");
          Assert.assertEquals(outputNames.size(), 1);
          Assert.assertEquals(outputNames.getString(0), "output");
        } catch (Exception e) {
          Assert.fail(e.getMessage());
        }
      }

      int[] dims = new int[] {1, 5};
      float[] inputData = new float[] {1.0f, 2.0f, -3.0f, Float.MIN_VALUE, Float.MAX_VALUE};

      // test run()
      {
        JavaOnlyMap inputDataMap = new JavaOnlyMap();
        {
          JavaOnlyMap inputTensorMap = new JavaOnlyMap();

          JavaOnlyArray dimsArray = new JavaOnlyArray();
          for (int dim : dims) {
            dimsArray.pushInt(dim);
          }
          inputTensorMap.putArray("dims", dimsArray);

          inputTensorMap.putString("type", TensorHelper.JsTensorTypeFloat);

          ByteBuffer buffer = ByteBuffer.allocate(5 * Float.BYTES).order(ByteOrder.nativeOrder());
          FloatBuffer floatBuffer = buffer.asFloatBuffer();
          for (float value : inputData) {
            floatBuffer.put(value);
          }
          floatBuffer.rewind();
          inputTensorMap.putMap("data", blobModule.testCreateData(buffer.array()));

          inputDataMap.putMap("input", inputTensorMap);
        }

        JavaOnlyArray outputNames = new JavaOnlyArray();
        outputNames.pushString("output");

        JavaOnlyMap options = new JavaOnlyMap();
        options.putBoolean("encodeTensorData", true);

        try {
          ReadableMap resultMap = ortModule.run(sessionKey, inputDataMap, outputNames, options);

          ReadableMap outputMap = resultMap.getMap("output");
          for (int i = 0; i < 2; ++i) {
            Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
          }
          Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeFloat);
          ReadableMap data = outputMap.getMap("data");
          FloatBuffer buffer =
              ByteBuffer.wrap(blobModule.testGetData(data)).order(ByteOrder.nativeOrder()).asFloatBuffer();
          for (int i = 0; i < 5; ++i) {
            Assert.assertEquals(buffer.get(i), inputData[i], 1e-6f);
          }
        } catch (Exception e) {
          Assert.fail(e.getMessage());
        }
      }

      // test dispose
      ortModule.dispose(sessionKey);
    } finally {
      mockSession.finishMocking();
    }
  }
}
