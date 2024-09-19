/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Assertions;

/** Test helpers for manipulating primitive arrays. */
public class TestHelpers {

  private static final Pattern LOAD_PATTERN = Pattern.compile("[,\\[\\] ]");

  static void deleteDirectoryTree(Path input) throws IOException {
    Files.walk(input).sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
  }

  static boolean[] toPrimitiveBoolean(List<Boolean> input) {
    boolean[] output = new boolean[input.size()];

    for (int i = 0; i < output.length; i++) {
      output[i] = input.get(i);
    }

    return output;
  }

  static byte[] toPrimitiveByte(List<Byte> input) {
    byte[] output = new byte[input.size()];

    for (int i = 0; i < output.length; i++) {
      output[i] = input.get(i);
    }

    return output;
  }

  static short[] toPrimitiveShort(List<Short> input) {
    short[] output = new short[input.size()];

    for (int i = 0; i < output.length; i++) {
      output[i] = input.get(i);
    }

    return output;
  }

  static int[] toPrimitiveInteger(List<Integer> input) {
    int[] output = new int[input.size()];

    for (int i = 0; i < output.length; i++) {
      output[i] = input.get(i);
    }

    return output;
  }

  static long[] toPrimitiveLong(List<Long> input) {
    long[] output = new long[input.size()];

    for (int i = 0; i < output.length; i++) {
      output[i] = input.get(i);
    }

    return output;
  }

  static float[] toPrimitiveFloat(List<Float> input) {
    float[] output = new float[input.size()];

    for (int i = 0; i < output.length; i++) {
      output[i] = input.get(i);
    }

    return output;
  }

  static double[] toPrimitiveDouble(List<Double> input) {
    double[] output = new double[input.size()];

    for (int i = 0; i < output.length; i++) {
      output[i] = input.get(i);
    }

    return output;
  }

  static boolean[] flattenBoolean(Object o) {
    List<Boolean> output = new ArrayList<>();

    flatten((Object[]) o, output, boolean.class);

    return toPrimitiveBoolean(output);
  }

  static byte[] flattenByte(Object o) {
    List<Byte> output = new ArrayList<>();

    flatten((Object[]) o, output, byte.class);

    return toPrimitiveByte(output);
  }

  static short[] flattenShort(Object o) {
    List<Short> output = new ArrayList<>();

    flatten((Object[]) o, output, short.class);

    return toPrimitiveShort(output);
  }

  static int[] flattenInteger(Object o) {
    List<Integer> output = new ArrayList<>();

    flatten((Object[]) o, output, int.class);

    return toPrimitiveInteger(output);
  }

  static long[] flattenLong(Object o) {
    List<Long> output = new ArrayList<>();

    flatten((Object[]) o, output, long.class);

    return toPrimitiveLong(output);
  }

  public static float[] flattenFloat(Object o) {
    List<Float> output = new ArrayList<>();

    flatten((Object[]) o, output, float.class);

    return toPrimitiveFloat(output);
  }

  static double[] flattenDouble(Object o) {
    List<Double> output = new ArrayList<>();

    flatten((Object[]) o, output, double.class);

    return toPrimitiveDouble(output);
  }

  static String[] flattenString(Object o) {
    List<String> output = new ArrayList<>();

    flatten((Object[]) o, output, String.class);

    return output.toArray(new String[0]);
  }

  static void flatten(Object[] input, List output, Class<?> primitiveClazz) {
    for (Object i : input) {
      Class<?> iClazz = i.getClass();
      if (iClazz.isArray()) {
        if (iClazz.getComponentType().isArray()) {
          flatten((Object[]) i, output, primitiveClazz);
        } else if ((iClazz.getComponentType().isPrimitive()
                || iClazz.getComponentType().equals(String.class))
            && iClazz.getComponentType().equals(primitiveClazz)) {
          flattenBase(i, output, primitiveClazz);
        } else {
          throw new IllegalStateException(
              "Found a non-primitive, non-array element type, " + iClazz);
        }
      } else {
        throw new IllegalStateException(
            "Found an element type where there should have been an array. Class = " + iClazz);
      }
    }
  }

  @SuppressWarnings("unchecked")
  static void flattenBase(Object input, List output, Class<?> primitiveClass) {
    if (primitiveClass.equals(boolean.class)) {
      flattenBooleanBase((boolean[]) input, output);
    } else if (primitiveClass.equals(byte.class)) {
      flattenByteBase((byte[]) input, output);
    } else if (primitiveClass.equals(short.class)) {
      flattenShortBase((short[]) input, output);
    } else if (primitiveClass.equals(int.class)) {
      flattenIntBase((int[]) input, output);
    } else if (primitiveClass.equals(long.class)) {
      flattenLongBase((long[]) input, output);
    } else if (primitiveClass.equals(float.class)) {
      flattenFloatBase((float[]) input, output);
    } else if (primitiveClass.equals(double.class)) {
      flattenDoubleBase((double[]) input, output);
    } else if (primitiveClass.equals(String.class)) {
      flattenStringBase((String[]) input, output);
    } else {
      throw new IllegalStateException("Flattening a non-primitive class");
    }
  }

  static void flattenBooleanBase(boolean[] input, List<Boolean> output) {
    for (int i = 0; i < input.length; i++) {
      output.add(input[i]);
    }
  }

  static void flattenByteBase(byte[] input, List<Byte> output) {
    for (int i = 0; i < input.length; i++) {
      output.add(input[i]);
    }
  }

  static void flattenShortBase(short[] input, List<Short> output) {
    for (int i = 0; i < input.length; i++) {
      output.add(input[i]);
    }
  }

  static void flattenIntBase(int[] input, List<Integer> output) {
    for (int i = 0; i < input.length; i++) {
      output.add(input[i]);
    }
  }

  static void flattenLongBase(long[] input, List<Long> output) {
    for (int i = 0; i < input.length; i++) {
      output.add(input[i]);
    }
  }

  static void flattenFloatBase(float[] input, List<Float> output) {
    for (int i = 0; i < input.length; i++) {
      output.add(input[i]);
    }
  }

  static void flattenDoubleBase(double[] input, List<Double> output) {
    for (int i = 0; i < input.length; i++) {
      output.add(input[i]);
    }
  }

  static void flattenStringBase(String[] input, List<String> output) {
    output.addAll(Arrays.asList(input));
  }

  static void loudLogger(Class<?> loggerClass) {
    Logger l = Logger.getLogger(loggerClass.getName());
    l.setLevel(Level.INFO);
  }

  static void quietLogger(Class<?> loggerClass) {
    Logger l = Logger.getLogger(loggerClass.getName());
    l.setLevel(Level.OFF);
  }

  public static Path getResourcePath(String path) {
    return new File(TestHelpers.class.getResource(path).getFile()).toPath();
  }

  public static void zeroBuffer(FloatBuffer buf) {
    for (int i = 0; i < buf.capacity(); i++) {
      buf.put(i, 0.0f);
    }
  }

  public static float[] loadTensorFromFile(Path filename) {
    return loadTensorFromFile(filename, true);
  }

  public static float[] loadTensorFromFile(Path filename, boolean skipHeader) {
    // read data from file
    try (BufferedReader reader = new BufferedReader(new FileReader(filename.toFile()))) {
      if (skipHeader) {
        reader.readLine(); // skip the input name
      }
      String[] dataStr = LOAD_PATTERN.split(reader.readLine());
      List<Float> tensorData = new ArrayList<>();
      for (int i = 0; i < dataStr.length; i++) {
        if (!dataStr[i].isEmpty()) {
          tensorData.add(Float.parseFloat(dataStr[i]));
        }
      }
      return toPrimitiveFloat(tensorData);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static TypeWidth getTypeAndWidth(OnnxMl.TensorProto.DataType elemType) {
    OnnxJavaType type;
    int width;
    switch (elemType) {
      case FLOAT:
        type = OnnxJavaType.FLOAT;
        width = 4;
        break;
      case UINT8:
      case INT8:
        type = OnnxJavaType.INT8;
        width = 1;
        break;
      case UINT16:
      case INT16:
        type = OnnxJavaType.INT16;
        width = 2;
        break;
      case INT32:
      case UINT32:
        type = OnnxJavaType.INT32;
        width = 4;
        break;
      case INT64:
      case UINT64:
        type = OnnxJavaType.INT64;
        width = 8;
        break;
      case STRING:
        type = OnnxJavaType.STRING;
        width = 1;
        break;
      case BOOL:
        type = OnnxJavaType.BOOL;
        width = 1;
        break;
      case FLOAT16:
        type = OnnxJavaType.FLOAT;
        width = 2;
        break;
      case DOUBLE:
        type = OnnxJavaType.DOUBLE;
        width = 8;
        break;
      default:
        type = null;
        width = 0;
        break;
    }
    return new TypeWidth(type, width);
  }

  static StringTensorPair loadTensorFromFilePb(
      OrtEnvironment env, File filename, Map<String, NodeInfo> nodeMetaDict)
      throws IOException, OrtException {
    InputStream is = new BufferedInputStream(new FileInputStream(filename), 1024 * 1024 * 4);
    OnnxMl.TensorProto tensor = OnnxMl.TensorProto.parseFrom(is);
    is.close();

    TypeWidth tw = getTypeAndWidth(OnnxMl.TensorProto.DataType.forNumber(tensor.getDataType()));
    int width = tw.width;
    OnnxJavaType tensorElemType = tw.type;
    long[] intDims = new long[tensor.getDimsCount()];
    for (int i = 0; i < tensor.getDimsCount(); i++) {
      intDims[i] = tensor.getDims(i);
    }

    TensorInfo nodeMeta = null;
    String nodeName = "";
    if (nodeMetaDict.size() == 1) {
      for (Map.Entry<String, NodeInfo> e : nodeMetaDict.entrySet()) {
        nodeMeta = (TensorInfo) e.getValue().getInfo();
        nodeName = e.getKey(); // valid for single node input
      }
    } else if (nodeMetaDict.size() > 1) {
      if (!tensor.getName().isEmpty()) {
        nodeMeta = (TensorInfo) nodeMetaDict.get(tensor.getName()).getInfo();
        nodeName = tensor.getName();
      } else {
        boolean matchfound = false;
        // try to find from matching type and shape
        for (Map.Entry<String, NodeInfo> e : nodeMetaDict.entrySet()) {
          if (e.getValue().getInfo() instanceof TensorInfo) {
            TensorInfo meta = (TensorInfo) e.getValue().getInfo();
            if (tensorElemType == meta.type && tensor.getDimsCount() == meta.shape.length) {
              int i = 0;
              for (; i < meta.shape.length; i++) {
                if (meta.shape[i] != -1 && meta.shape[i] != intDims[i]) {
                  break;
                }
              }
              if (i >= meta.shape.length) {
                matchfound = true;
                nodeMeta = meta;
                nodeName = e.getKey();
                break;
              }
            }
          }
        }
        if (!matchfound) {
          // throw error
          throw new IllegalStateException(
              "No matching Tensor found in InputOutputMetadata corresponding to the serialized tensor loaded from "
                  + filename);
        }
      }
    } else {
      // throw error
      throw new IllegalStateException(
          "While reading the serialized tensor loaded from "
              + filename
              + ", metaDataDict has 0 elements");
    }

    Assertions.assertEquals(tensorElemType, nodeMeta.type);
    Assertions.assertEquals(nodeMeta.shape.length, tensor.getDimsCount());
    for (int i = 0; i < nodeMeta.shape.length; i++) {
      Assertions.assertTrue((nodeMeta.shape[i] == -1) || (nodeMeta.shape[i] == intDims[i]));
    }

    ByteBuffer buffer = ByteBuffer.wrap(tensor.getRawData().toByteArray());

    OnnxTensor onnxTensor = OnnxTensor.createTensor(env, buffer, intDims, tensorElemType);

    return new StringTensorPair(nodeName, onnxTensor);
  }

  public static OnnxTensor makeIdentityMatrix(OrtEnvironment env, int size) throws OrtException {
    float[][] values = new float[size][size];
    for (int i = 0; i < values.length; i++) {
      values[i][i] = 1.0f;
    }

    return OnnxTensor.createTensor(env, values);
  }

  public static OnnxTensor makeIdentityMatrixBuf(OrtEnvironment env, int size) throws OrtException {
    FloatBuffer buf =
        ByteBuffer.allocateDirect(size * size * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    for (int i = 0; i < size; i++) {
      buf.put(i * size + i, 1.0f);
    }

    return OnnxTensor.createTensor(env, buf, new long[] {size, size});
  }

  private static class TypeWidth {
    public final OnnxJavaType type;
    public final int width;

    public TypeWidth(OnnxJavaType type, int width) {
      this.type = type;
      this.width = width;
    }
  }

  static class StringTensorPair {
    public final String string;
    public final OnnxTensor tensor;

    public StringTensorPair(String string, OnnxTensor tensor) {
      this.string = string;
      this.tensor = tensor;
    }
  }
}
