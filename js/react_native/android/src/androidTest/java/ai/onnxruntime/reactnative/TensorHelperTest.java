// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import static com.android.dx.mockito.inline.extended.ExtendedMockito.mockitoSession;
import static org.mockito.Mockito.when;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;
import ai.onnxruntime.TensorInfo;
import android.content.Context;
import android.util.Base64;
import androidx.test.filters.SmallTest;
import androidx.test.platform.app.InstrumentationRegistry;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.JavaOnlyArray;
import com.facebook.react.bridge.JavaOnlyMap;
import com.facebook.react.bridge.ReadableMap;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.HashMap;
import java.util.Map;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoSession;

@SmallTest
public class TensorHelperTest {
  private OrtEnvironment ortEnvironment;

  @Before
  public void setUp() {
    ortEnvironment = OrtEnvironment.getEnvironment("TensorHelperTest");
  }

  @Test
  public void createInputTensor_float32() throws Exception {
    OnnxTensor outputTensor =
        OnnxTensor.createTensor(ortEnvironment, new float[] {Float.MIN_VALUE, 2.0f, Float.MAX_VALUE});

    JavaOnlyMap inputTensorMap = new JavaOnlyMap();

    JavaOnlyArray dims = new JavaOnlyArray();
    dims.pushInt(3);
    inputTensorMap.putArray("dims", dims);

    inputTensorMap.putString("type", TensorHelper.JsTensorTypeFloat);

    ByteBuffer dataByteBuffer = ByteBuffer.allocate(3 * 4).order(ByteOrder.nativeOrder());
    FloatBuffer dataFloatBuffer = dataByteBuffer.asFloatBuffer();
    dataFloatBuffer.put(Float.MIN_VALUE);
    dataFloatBuffer.put(2.0f);
    dataFloatBuffer.put(Float.MAX_VALUE);
    String dataEncoded = Base64.encodeToString(dataByteBuffer.array(), Base64.DEFAULT);
    inputTensorMap.putString("data", dataEncoded);

    OnnxTensor inputTensor = TensorHelper.createInputTensor(inputTensorMap, ortEnvironment);

    Assert.assertEquals(inputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Assert.assertEquals(outputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Assert.assertEquals(inputTensor.toString(), outputTensor.toString());
    Assert.assertArrayEquals(inputTensor.getFloatBuffer().array(), outputTensor.getFloatBuffer().array(), 1e-6f);

    inputTensor.close();
    outputTensor.close();
  }

  @Test
  public void createInputTensor_int8() throws Exception {
    OnnxTensor outputTensor = OnnxTensor.createTensor(ortEnvironment, new byte[] {Byte.MIN_VALUE, 2, Byte.MAX_VALUE});

    JavaOnlyMap inputTensorMap = new JavaOnlyMap();

    JavaOnlyArray dims = new JavaOnlyArray();
    dims.pushInt(3);
    inputTensorMap.putArray("dims", dims);

    inputTensorMap.putString("type", TensorHelper.JsTensorTypeByte);

    ByteBuffer dataByteBuffer = ByteBuffer.allocate(3);
    dataByteBuffer.put(Byte.MIN_VALUE);
    dataByteBuffer.put((byte)2);
    dataByteBuffer.put(Byte.MAX_VALUE);
    String dataEncoded = Base64.encodeToString(dataByteBuffer.array(), Base64.DEFAULT);
    inputTensorMap.putString("data", dataEncoded);

    OnnxTensor inputTensor = TensorHelper.createInputTensor(inputTensorMap, ortEnvironment);

    Assert.assertEquals(inputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
    Assert.assertEquals(outputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
    Assert.assertEquals(inputTensor.toString(), outputTensor.toString());
    Assert.assertArrayEquals(inputTensor.getByteBuffer().array(), outputTensor.getByteBuffer().array());

    inputTensor.close();
    outputTensor.close();
  }

  @Test
  public void createInputTensor_int16() throws Exception {
    OnnxTensor outputTensor =
        OnnxTensor.createTensor(ortEnvironment, new short[] {Short.MIN_VALUE, 2, Short.MAX_VALUE});

    JavaOnlyMap inputTensorMap = new JavaOnlyMap();

    JavaOnlyArray dims = new JavaOnlyArray();
    dims.pushInt(3);
    inputTensorMap.putArray("dims", dims);

    inputTensorMap.putString("type", TensorHelper.JsTensorTypeShort);

    ByteBuffer dataByteBuffer = ByteBuffer.allocate(3 * 2).order(ByteOrder.nativeOrder());
    ShortBuffer dataShortBuffer = dataByteBuffer.asShortBuffer();
    dataShortBuffer.put(Short.MIN_VALUE);
    dataShortBuffer.put((short)2);
    dataShortBuffer.put(Short.MAX_VALUE);
    String dataEncoded = Base64.encodeToString(dataByteBuffer.array(), Base64.DEFAULT);
    inputTensorMap.putString("data", dataEncoded);

    OnnxTensor inputTensor = TensorHelper.createInputTensor(inputTensorMap, ortEnvironment);

    Assert.assertEquals(inputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
    Assert.assertEquals(outputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
    Assert.assertEquals(inputTensor.toString(), outputTensor.toString());
    Assert.assertArrayEquals(inputTensor.getShortBuffer().array(), outputTensor.getShortBuffer().array());

    inputTensor.close();
    outputTensor.close();
  }

  @Test
  public void createInputTensor_int32() throws Exception {
    OnnxTensor outputTensor =
        OnnxTensor.createTensor(ortEnvironment, new int[] {Integer.MIN_VALUE, 2, Integer.MAX_VALUE});

    JavaOnlyMap inputTensorMap = new JavaOnlyMap();

    JavaOnlyArray dims = new JavaOnlyArray();
    dims.pushInt(3);
    inputTensorMap.putArray("dims", dims);

    inputTensorMap.putString("type", TensorHelper.JsTensorTypeInt);

    ByteBuffer dataByteBuffer = ByteBuffer.allocate(3 * 4).order(ByteOrder.nativeOrder());
    IntBuffer dataIntBuffer = dataByteBuffer.asIntBuffer();
    dataIntBuffer.put(Integer.MIN_VALUE);
    dataIntBuffer.put(2);
    dataIntBuffer.put(Integer.MAX_VALUE);
    String dataEncoded = Base64.encodeToString(dataByteBuffer.array(), Base64.DEFAULT);
    inputTensorMap.putString("data", dataEncoded);

    OnnxTensor inputTensor = TensorHelper.createInputTensor(inputTensorMap, ortEnvironment);

    Assert.assertEquals(inputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    Assert.assertEquals(outputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    Assert.assertEquals(inputTensor.toString(), outputTensor.toString());
    Assert.assertArrayEquals(inputTensor.getIntBuffer().array(), outputTensor.getIntBuffer().array());

    inputTensor.close();
    outputTensor.close();
  }

  @Test
  public void createInputTensor_int64() throws Exception {
    OnnxTensor outputTensor =
        OnnxTensor.createTensor(ortEnvironment, new long[] {Long.MIN_VALUE, 15000000001L, Long.MAX_VALUE});

    JavaOnlyMap inputTensorMap = new JavaOnlyMap();

    JavaOnlyArray dims = new JavaOnlyArray();
    dims.pushInt(3);
    inputTensorMap.putArray("dims", dims);

    inputTensorMap.putString("type", TensorHelper.JsTensorTypeLong);

    ByteBuffer dataByteBuffer = ByteBuffer.allocate(3 * 8).order(ByteOrder.nativeOrder());
    LongBuffer dataLongBuffer = dataByteBuffer.asLongBuffer();
    dataLongBuffer.put(Long.MIN_VALUE);
    dataLongBuffer.put(15000000001L);
    dataLongBuffer.put(Long.MAX_VALUE);
    String dataEncoded = Base64.encodeToString(dataByteBuffer.array(), Base64.DEFAULT);
    inputTensorMap.putString("data", dataEncoded);

    OnnxTensor inputTensor = TensorHelper.createInputTensor(inputTensorMap, ortEnvironment);

    Assert.assertEquals(inputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    Assert.assertEquals(outputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    Assert.assertEquals(inputTensor.toString(), outputTensor.toString());
    Assert.assertArrayEquals(inputTensor.getLongBuffer().array(), outputTensor.getLongBuffer().array());

    inputTensor.close();
    outputTensor.close();
  }

  @Test
  public void createInputTensor_string() throws Exception {
    OnnxTensor outputTensor = OnnxTensor.createTensor(ortEnvironment, new String[] {"a", "b", "c"}, new long[] {3});

    JavaOnlyMap inputTensorMap = new JavaOnlyMap();

    JavaOnlyArray dims = new JavaOnlyArray();
    dims.pushInt(3);
    inputTensorMap.putArray("dims", dims);

    inputTensorMap.putString("type", TensorHelper.JsTensorTypeString);

    JavaOnlyArray data = new JavaOnlyArray();
    data.pushString("a");
    data.pushString("b");
    data.pushString("c");
    inputTensorMap.putArray("data", data);

    OnnxTensor inputTensor = TensorHelper.createInputTensor(inputTensorMap, ortEnvironment);

    Assert.assertEquals(inputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    Assert.assertEquals(outputTensor.getInfo().onnxType,
                        TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    Assert.assertEquals(inputTensor.toString(), outputTensor.toString());
    String[] inputData = (String[])inputTensor.getValue();
    String[] outputData = (String[])outputTensor.getValue();
    Assert.assertArrayEquals(inputData, outputData);

    inputTensor.close();
    outputTensor.close();
  }

  @Test
  public void createInputTensor_double() throws Exception {
    OnnxTensor outputTensor =
        OnnxTensor.createTensor(ortEnvironment, new double[] {Double.MIN_VALUE, 1.8e+30, Double.MAX_VALUE});

    JavaOnlyMap inputTensorMap = new JavaOnlyMap();

    JavaOnlyArray dims = new JavaOnlyArray();
    dims.pushInt(3);
    inputTensorMap.putArray("dims", dims);

    inputTensorMap.putString("type", TensorHelper.JsTensorTypeDouble);

    ByteBuffer dataByteBuffer = ByteBuffer.allocate(3 * 8).order(ByteOrder.nativeOrder());
    DoubleBuffer dataDoubleBuffer = dataByteBuffer.asDoubleBuffer();
    dataDoubleBuffer.put(Double.MIN_VALUE);
    dataDoubleBuffer.put(1.8e+30);
    dataDoubleBuffer.put(Double.MAX_VALUE);
    String dataEncoded = Base64.encodeToString(dataByteBuffer.array(), Base64.DEFAULT);
    inputTensorMap.putString("data", dataEncoded);

    OnnxTensor inputTensor = TensorHelper.createInputTensor(inputTensorMap, ortEnvironment);

    Assert.assertEquals(inputTensor.getInfo().onnxType, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
    Assert.assertEquals(outputTensor.getInfo().onnxType,
                        TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
    Assert.assertEquals(inputTensor.toString(), outputTensor.toString());
    Assert.assertArrayEquals(inputTensor.getDoubleBuffer().array(), outputTensor.getDoubleBuffer().array(), 1e-6f);

    inputTensor.close();
    outputTensor.close();
  }

  @Test
  public void createOutputTensor_bool() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_bool);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      boolean[] inputData = new boolean[] {true, false, false, true, false};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      Object tensorInput = OrtUtil.reshape(inputData, dims);
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, tensorInput);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeBool);
      String dataEncoded = outputMap.getString("data");
      ByteBuffer buffer = ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT));
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(buffer.get(i) == 1, inputData[i]);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  @Test
  public void createOutputTensor_double() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_double);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      double[] inputData = new double[] {1.0f, 2.0f, -3.0f, Double.MIN_VALUE, Double.MAX_VALUE};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      Object tensorInput = OrtUtil.reshape(inputData, dims);
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, tensorInput);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeDouble);
      String dataEncoded = outputMap.getString("data");
      DoubleBuffer buffer =
          ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT)).order(ByteOrder.nativeOrder()).asDoubleBuffer();
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(buffer.get(i), inputData[i], 1e-6f);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  @Test
  public void createOutputTensor_float() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_float);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      float[] inputData = new float[] {1.0f, 2.0f, -3.0f, Float.MIN_VALUE, Float.MAX_VALUE};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      Object tensorInput = OrtUtil.reshape(inputData, dims);
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, tensorInput);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeFloat);
      String dataEncoded = outputMap.getString("data");
      FloatBuffer buffer =
          ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT)).order(ByteOrder.nativeOrder()).asFloatBuffer();
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(buffer.get(i), inputData[i], 1e-6f);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  @Test
  public void createOutputTensor_int8() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_int8);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      byte[] inputData = new byte[] {1, 2, -3, Byte.MAX_VALUE, Byte.MAX_VALUE};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      Object tensorInput = OrtUtil.reshape(inputData, dims);
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, tensorInput);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeByte);
      String dataEncoded = outputMap.getString("data");
      ByteBuffer buffer = ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT));
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(buffer.get(i), inputData[i]);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  @Test
  public void createOutputTensor_int16() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_int16);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      short[] inputData = new short[] {1, 2, -3, Short.MIN_VALUE, Short.MAX_VALUE};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      Object tensorInput = OrtUtil.reshape(inputData, dims);
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, tensorInput);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeShort);
      String dataEncoded = outputMap.getString("data");
      ShortBuffer buffer =
          ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT)).order(ByteOrder.nativeOrder()).asShortBuffer();
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(buffer.get(i), inputData[i]);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  @Test
  public void createOutputTensor_int32() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_int32);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      int[] inputData = new int[] {1, 2, -3, Integer.MIN_VALUE, Integer.MAX_VALUE};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      Object tensorInput = OrtUtil.reshape(inputData, dims);
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, tensorInput);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeInt);
      String dataEncoded = outputMap.getString("data");
      IntBuffer buffer =
          ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT)).order(ByteOrder.nativeOrder()).asIntBuffer();
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(buffer.get(i), inputData[i]);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  @Test
  public void createOutputTensor_int64() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_int64);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      long[] inputData = new long[] {1, 2, -3, Long.MIN_VALUE, Long.MAX_VALUE};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      Object tensorInput = OrtUtil.reshape(inputData, dims);
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, tensorInput);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeLong);
      String dataEncoded = outputMap.getString("data");
      LongBuffer buffer =
          ByteBuffer.wrap(Base64.decode(dataEncoded, Base64.DEFAULT)).order(ByteOrder.nativeOrder()).asLongBuffer();
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(buffer.get(i), inputData[i]);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  @Test
  public void createOutputTensor_string() throws Exception {
    MockitoSession mockSession = mockitoSession().mockStatic(Arguments.class).startMocking();
    try {
      when(Arguments.createMap()).thenAnswer(i -> new JavaOnlyMap());
      when(Arguments.createArray()).thenAnswer(i -> new JavaOnlyArray());

      OrtSession.SessionOptions options = new OrtSession.SessionOptions();
      byte[] modelData = readBytesFromResourceFile(ai.onnxruntime.reactnative.test.R.raw.test_types_string);
      OrtSession session = ortEnvironment.createSession(modelData, options);

      long[] dims = new long[] {1, 5};
      String[] inputData = new String[] {"a", "b", "c", "d", "e"};

      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      OnnxTensor onnxTensor = OnnxTensor.createTensor(ortEnvironment, inputData, dims);
      container.put(inputName, onnxTensor);

      OrtSession.Result result = session.run(container);
      String[] outputData = (String[])((OnnxTensor)result.get(0)).getValue();

      ReadableMap resultMap = TensorHelper.createOutputTensor(result);
      ReadableMap outputMap = resultMap.getMap("output");
      for (int i = 0; i < 2; ++i) {
        Assert.assertEquals(outputMap.getArray("dims").getInt(i), dims[i]);
      }
      Assert.assertEquals(outputMap.getString("type"), TensorHelper.JsTensorTypeString);
      for (int i = 0; i < 5; ++i) {
        Assert.assertEquals(outputMap.getArray("data").getString(i), inputData[i]);
      }

      OnnxValue.close(container);
    } finally {
      mockSession.finishMocking();
    }
  }

  private byte[] readBytesFromResourceFile(int resourceId) throws Exception {
    Context context = InstrumentationRegistry.getInstrumentation().getContext();
    InputStream inputStream = context.getResources().openRawResource(resourceId);
    ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();

    int bufferSize = 1024;
    byte[] buffer = new byte[bufferSize];

    int len;
    while ((len = inputStream.read(buffer)) != -1) {
      byteBuffer.write(buffer, 0, len);
    }

    return byteBuffer.toByteArray();
  }
}
