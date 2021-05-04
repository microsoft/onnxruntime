// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import android.util.Base64;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;
import ai.onnxruntime.TensorInfo;

public class TensorHelper {
  /**
   * Supported tensor data type
   */
  public static final String JsTensorTypeBool = "bool";
  public static final String JsTensorTypeByte = "int8";
  public static final String JsTensorTypeShort = "int16";
  public static final String JsTensorTypeInt = "int32";
  public static final String JsTensorTypeLong = "int64";
  public static final String JsTensorTypeFloat = "float32";
  public static final String JsTensorTypeDouble = "float64";
  public static final String JsTensorTypeString = "string";

  /**
   * It creates an input tensor from a map passed by react native js.
   * 'data' must be a string type as data is encoded as base64. It first decodes it and creates a tensor.
   */
  public static OnnxTensor createInputTensor(ReadableMap inputTensor, OrtEnvironment ortEnvironment) throws Exception {
    // shape
    ReadableArray dimsArray = inputTensor.getArray("dims");
    long[] dims = new long[dimsArray.size()];
    for (int i = 0; i < dimsArray.size(); ++i) {
      dims[i] = dimsArray.getInt(i);
    }

    // type
    TensorInfo.OnnxTensorType tensorType = getOnnxTensorType(inputTensor.getString("type"));

    // data
    OnnxTensor onnxTensor = null;
    if (tensorType == TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      ReadableArray values = inputTensor.getArray("data");
      String[] buffer = new String[values.size()];
      for (int i = 0; i < values.size(); ++i) {
        buffer[i] = values.getString(i);
      }
      onnxTensor = OnnxTensor.createTensor(ortEnvironment, buffer, dims);
    } else {
      String data = inputTensor.getString("data");
      ByteBuffer values = ByteBuffer.wrap(Base64.decode(data, Base64.DEFAULT)).order(ByteOrder.nativeOrder());
      onnxTensor = createInputTensor(tensorType, dims, values, ortEnvironment);
    }

    return onnxTensor;
  }

  /**
   * It creates an output map from an output tensor.
   * a data array is encoded as base64 string.
   */
  public static WritableMap createOutputTensor(OrtSession.Result result) throws Exception {
    WritableMap outputTensorMap = Arguments.createMap();

    Iterator<Map.Entry<String, OnnxValue>> iterator = result.iterator();
    while (iterator.hasNext()) {
      Map.Entry<String, OnnxValue> entry = iterator.next();
      String outputName = entry.getKey();
      OnnxValue onnxValue = (OnnxValue)entry.getValue();
      if (onnxValue.getType() != OnnxValue.OnnxValueType.ONNX_TYPE_TENSOR) {
        throw new Exception("Not supported type: " + onnxValue.getType().toString());
      }

      OnnxTensor onnxTensor = (OnnxTensor)onnxValue;
      WritableMap outputTensor = Arguments.createMap();

      // dims
      WritableArray outputDims = Arguments.createArray();
      long[] dims = onnxTensor.getInfo().getShape();
      for (long dim : dims) {
        outputDims.pushInt((int)dim);
      }
      outputTensor.putArray("dims", outputDims);

      // type
      outputTensor.putString("type", getJsTensorType(onnxTensor.getInfo().onnxType));

      // data
      if (onnxTensor.getInfo().onnxType == TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        String[] buffer = (String[])onnxTensor.getValue();
        WritableArray dataArray = Arguments.createArray();
        for (String value: buffer) {
          dataArray.pushString(value);
        }
        outputTensor.putArray("data", dataArray);
      } else {
        String data = createOutputTensor(onnxTensor);
        outputTensor.putString("data", data);
      }

      outputTensorMap.putMap(outputName, outputTensor);
    }

    return outputTensorMap;
  }

  private static OnnxTensor createInputTensor(TensorInfo.OnnxTensorType tensorType,
                                             long[] dims,
                                             ByteBuffer values,
                                             OrtEnvironment ortEnvironment) throws Exception {
    OnnxTensor tensor = null;
    switch (tensorType) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
        FloatBuffer buffer = values.asFloatBuffer();
        tensor = OnnxTensor.createTensor(ortEnvironment, buffer, dims);
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
        ByteBuffer buffer = values;
        tensor = OnnxTensor.createTensor(ortEnvironment, buffer, dims);
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
        ShortBuffer buffer = values.asShortBuffer();
        tensor = OnnxTensor.createTensor(ortEnvironment, buffer, dims);
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        IntBuffer buffer = values.asIntBuffer();
        tensor = OnnxTensor.createTensor(ortEnvironment, buffer, dims);
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        LongBuffer buffer = values.asLongBuffer();
        tensor = OnnxTensor.createTensor(ortEnvironment, buffer, dims);
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
        DoubleBuffer buffer = values.asDoubleBuffer();
        tensor = OnnxTensor.createTensor(ortEnvironment, buffer, dims);
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      default:
        throw new IllegalStateException("Unexpected value: " + tensorType.toString());
    }

    return tensor;
  }

  private static String createOutputTensor(OnnxTensor onnxTensor) throws Exception {
    TensorInfo tensorInfo = onnxTensor.getInfo();
    ByteBuffer buffer = null;

    int capacity = (int) OrtUtil.elementCount(onnxTensor.getInfo().getShape());

    switch (tensorInfo.onnxType) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        buffer = ByteBuffer.allocate(capacity * 4).order(ByteOrder.nativeOrder());
        buffer.asFloatBuffer().put(onnxTensor.getFloatBuffer());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        buffer = ByteBuffer.allocate(capacity).order(ByteOrder.nativeOrder());
        buffer.put(onnxTensor.getByteBuffer());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        buffer = ByteBuffer.allocate(capacity * 2).order(ByteOrder.nativeOrder());
        buffer.asShortBuffer().put(onnxTensor.getShortBuffer());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        buffer = ByteBuffer.allocate(capacity * 4).order(ByteOrder.nativeOrder());
        buffer.asIntBuffer().put(onnxTensor.getIntBuffer());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        buffer = ByteBuffer.allocate(capacity * 8).order(ByteOrder.nativeOrder());
        buffer.asLongBuffer().put(onnxTensor.getLongBuffer());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
        buffer = ByteBuffer.allocate(capacity * 8).order(ByteOrder.nativeOrder());
        buffer.asDoubleBuffer().put(onnxTensor.getDoubleBuffer());
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      default:
        throw new IllegalStateException("Unexpected type: " + tensorInfo.onnxType.toString());
    }

    String data = Base64.encodeToString(buffer.array(), Base64.DEFAULT);
    return data;
  }

  private static final Map<String, TensorInfo.OnnxTensorType> JsTensorTypeToOnnxTensorTypeMap = Stream.of(new Object[][]{
    {JsTensorTypeFloat, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
    {JsTensorTypeByte, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8},
    {JsTensorTypeShort, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16},
    {JsTensorTypeInt, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
    {JsTensorTypeLong, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
    {JsTensorTypeString, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
    {JsTensorTypeBool, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL},
    {JsTensorTypeDouble, TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE},
  }).collect(Collectors.toMap(p -> (String)p[0], p -> (TensorInfo.OnnxTensorType)p[1]));

  private static TensorInfo.OnnxTensorType getOnnxTensorType(String type) {
    if (JsTensorTypeToOnnxTensorTypeMap.containsKey(type)) {
      return JsTensorTypeToOnnxTensorTypeMap.get(type);
    } else {
      return TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }

  private static final Map<TensorInfo.OnnxTensorType, String> OnnxTensorTypeToJsTensorTypeMap = Stream.of(new Object[][]{
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, JsTensorTypeFloat},
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, JsTensorTypeByte},
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, JsTensorTypeShort},
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, JsTensorTypeInt},
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, JsTensorTypeLong},
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, JsTensorTypeString},
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, JsTensorTypeBool},
    {TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, JsTensorTypeDouble},
  }).collect(Collectors.toMap(p -> (TensorInfo.OnnxTensorType)p[0], p -> (String)p[1]));

  private static String getJsTensorType(TensorInfo.OnnxTensorType type) {
    if (OnnxTensorTypeToJsTensorTypeMap.containsKey(type)) {
      return OnnxTensorTypeToJsTensorTypeMap.get(type);
    } else {
      return "undefined";
    }
  }
}
