// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import android.util.Base64;

import androidx.test.platform.app.InstrumentationRegistry;

import static com.android.dx.mockito.inline.extended.ExtendedMockito.mockitoSession;
import static org.mockito.Mockito.when;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.JavaOnlyArray;
import com.facebook.react.bridge.JavaOnlyMap;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoSession;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import ai.onnxruntime.TensorInfo;

public class OnnxruntimeModuleTest {
  private ReactApplicationContext reactContext = new ReactApplicationContext(InstrumentationRegistry.getInstrumentation().getContext());

  @Before
  public void setUp() {}

  @Test
  public void getName() throws Exception {
    OnnxruntimeModule ortModule = new OnnxruntimeModule(reactContext);
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

      // test loadModel()
      {
        InputStream modelStream = reactContext.getResources().openRawResource(ai.onnxruntime.reactnative.test.R.raw.test_types_float);
        JavaOnlyMap options = new JavaOnlyMap();
        try {
          ReadableMap resultMap = ortModule.loadModel("test", modelStream, options);
          ReadableArray inputNames = resultMap.getArray("inputNames");
          ReadableArray outputNames = resultMap.getArray("outputNames");

          Assert.assertEquals(resultMap.getString("key"), "test");
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

          inputTensorMap.putString("type", TensorHelper.TensorTypeFloat);

          ByteBuffer buffer = ByteBuffer.allocate(5 * Float.BYTES).order(ByteOrder.nativeOrder());
          FloatBuffer floatBuffer = buffer.asFloatBuffer();
          for (float value : inputData) {
            floatBuffer.put(value);
          }
          floatBuffer.rewind();
          String dataEncoded = Base64.encodeToString(buffer.array(), Base64.DEFAULT);
          inputTensorMap.putString("data", dataEncoded);

          inputDataMap.putMap("input", inputTensorMap);
        }

        JavaOnlyArray outputNames = new JavaOnlyArray();
        outputNames.pushString("output");

        JavaOnlyMap options = new JavaOnlyMap();
        options.putBoolean("encodeTensorData", true);

        try {
          ReadableMap resultMap = ortModule.run("test", inputDataMap, outputNames, options);

          ReadableMap outputMap = resultMap.getMap("output");
          for (int i = 0; i < 2; ++i) {
            Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
          }
          Assert.assertEquals(outputMap.getString("type"), TensorHelper.TensorTypeFloat);
          String dataEncoded = outputMap.getString("data");
          FloatBuffer buffer = ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT)).order(ByteOrder.nativeOrder()).asFloatBuffer();
          for (int i = 0; i < 5; ++i) {
            Assert.assertEquals(buffer.get(i), inputData[i], 1e-6f);
          }
        } catch (Exception e) {
          Assert.fail(e.getMessage());
        }
      }
    } finally {
      mockSession.finishMocking();
    }
  }
}
