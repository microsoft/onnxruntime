/*
 * Copyright (c) 2021, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.Collections;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class OnnxTensorTest {

  @Test
  public void testScalarCreation() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String[] stringValues = new String[] {"true", "false"};
    for (String s : stringValues) {
      try (OnnxTensor t = OnnxTensor.createTensor(env, s)) {
        Assertions.assertEquals(s, t.getValue());
      }
    }

    boolean[] boolValues = new boolean[] {true, false};
    for (boolean b : boolValues) {
      try (OnnxTensor t = OnnxTensor.createTensor(env, b)) {
        Assertions.assertEquals(b, t.getValue());
      }
    }

    int[] intValues =
        new int[] {-1, 0, 1, 12345678, -12345678, Integer.MAX_VALUE, Integer.MIN_VALUE};
    for (int i : intValues) {
      try (OnnxTensor t = OnnxTensor.createTensor(env, i)) {
        Assertions.assertEquals(i, t.getValue());
      }
    }

    long[] longValues =
        new long[] {-1L, 0L, 1L, 12345678L, -12345678L, Long.MAX_VALUE, Long.MIN_VALUE};
    for (long l : longValues) {
      try (OnnxTensor t = OnnxTensor.createTensor(env, l)) {
        Assertions.assertEquals(l, t.getValue());
      }
    }

    float[] floatValues =
        new float[] {
          -1.0f,
          0.0f,
          -0.0f,
          1.0f,
          1234.5678f,
          -1234.5678f,
          (float) Math.PI,
          (float) Math.E,
          Float.MAX_VALUE,
          Float.MIN_VALUE
        };
    for (float f : floatValues) {
      try (OnnxTensor t = OnnxTensor.createTensor(env, f)) {
        Assertions.assertEquals(f, t.getValue());
      }
    }

    double[] doubleValues =
        new double[] {
          -1.0,
          0.0,
          -0.0,
          1.0,
          1234.5678,
          -1234.5678,
          Math.PI,
          Math.E,
          Double.MAX_VALUE,
          Double.MIN_VALUE
        };
    for (double d : doubleValues) {
      try (OnnxTensor t = OnnxTensor.createTensor(env, d)) {
        Assertions.assertEquals(d, t.getValue());
      }
    }
  }

  @Test
  public void testStringCreation() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String[] arrValues = new String[] {"this", "is", "a", "single", "dimensional", "string"};
    try (OnnxTensor t = OnnxTensor.createTensor(env, arrValues)) {
      Assertions.assertArrayEquals(new long[] {6}, t.getInfo().shape);
      String[] output = (String[]) t.getValue();
      Assertions.assertArrayEquals(arrValues, output);
    }

    String[][] stringValues =
        new String[][] {{"this", "is", "a"}, {"multi", "dimensional", "string"}};
    try (OnnxTensor t = OnnxTensor.createTensor(env, stringValues)) {
      Assertions.assertArrayEquals(new long[] {2, 3}, t.getInfo().shape);
      String[][] output = (String[][]) t.getValue();
      Assertions.assertArrayEquals(stringValues, output);
    }

    String[][][] deepStringValues =
        new String[][][] {
          {{"this", "is", "a"}, {"multi", "dimensional", "string"}},
          {{"with", "lots", "more"}, {"dimensions", "than", "before"}}
        };
    try (OnnxTensor t = OnnxTensor.createTensor(env, deepStringValues)) {
      Assertions.assertArrayEquals(new long[] {2, 2, 3}, t.getInfo().shape);
      String[][][] output = (String[][][]) t.getValue();
      Assertions.assertArrayEquals(deepStringValues, output);
    }
  }

  @Test
  public void testUint8Creation() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    byte[] buf = new byte[] {0, 1};
    ByteBuffer data = ByteBuffer.wrap(buf);
    long[] shape = new long[] {2};
    try (OnnxTensor t = OnnxTensor.createTensor(env, data, shape, OnnxJavaType.UINT8)) {
      Assertions.assertArrayEquals(buf, (byte[]) t.getValue());
    }
  }

  @Test
  public void testEmptyTensor() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    FloatBuffer buf = FloatBuffer.allocate(0);
    long[] shape = new long[] {4, 0};
    try (OnnxTensor t = OnnxTensor.createTensor(env, buf, shape)) {
      Assertions.assertArrayEquals(shape, t.getInfo().getShape());
      float[][] output = (float[][]) t.getValue();
      Assertions.assertEquals(4, output.length);
      Assertions.assertEquals(0, output[0].length);
      FloatBuffer fb = t.getFloatBuffer();
      Assertions.assertEquals(0, fb.remaining());
    }
    shape = new long[] {0, 4};
    try (OnnxTensor t = OnnxTensor.createTensor(env, buf, shape)) {
      Assertions.assertArrayEquals(shape, t.getInfo().getShape());
      float[][] output = (float[][]) t.getValue();
      Assertions.assertEquals(0, output.length);
    }
  }

  @Test
  public void testBf16ToFp32() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String modelPath = TestHelpers.getResourcePath("/java-bf16-to-fp32.onnx").toString();
    SplittableRandom rng = new SplittableRandom(1);

    float[][] input = new float[10][5];
    ByteBuffer buf = ByteBuffer.allocateDirect(2 * 10 * 5).order(ByteOrder.nativeOrder());
    ShortBuffer shortBuf = buf.asShortBuffer();

    // Generate data
    for (int i = 0; i < input.length; i++) {
      for (int j = 0; j < input[0].length; j++) {
        short bits = (short) rng.nextInt();
        input[i][j] = OrtUtil.bf16ToFloat(bits);
        shortBuf.put(bits);
      }
    }
    shortBuf.rewind();

    try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, opts);
        OnnxTensor tensor =
            OnnxTensor.createTensor(env, buf, new long[] {10, 5}, OnnxJavaType.BFLOAT16);
        OrtSession.Result result = session.run(Collections.singletonMap("input", tensor))) {
      OnnxTensor output = (OnnxTensor) result.get(0);
      float[][] outputArr = (float[][]) output.getValue();
      for (int i = 0; i < input.length; i++) {
        Assertions.assertArrayEquals(input[i], outputArr[i]);
      }
    }
  }

  @Test
  public void testFp16ToFp32() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String modelPath = TestHelpers.getResourcePath("/java-fp16-to-fp32.onnx").toString();
    SplittableRandom rng = new SplittableRandom(1);

    float[][] input = new float[10][5];
    ByteBuffer buf = ByteBuffer.allocateDirect(2 * 10 * 5).order(ByteOrder.nativeOrder());
    ShortBuffer shortBuf = buf.asShortBuffer();

    // Generate data
    for (int i = 0; i < input.length; i++) {
      for (int j = 0; j < input[0].length; j++) {
        short bits = (short) rng.nextInt();
        input[i][j] = OrtUtil.fp16ToFloat(bits);
        shortBuf.put(bits);
      }
    }
    shortBuf.rewind();

    try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, opts);
        OnnxTensor tensor =
            OnnxTensor.createTensor(env, buf, new long[] {10, 5}, OnnxJavaType.FLOAT16);
        OrtSession.Result result = session.run(Collections.singletonMap("input", tensor))) {
      OnnxTensor output = (OnnxTensor) result.get(0);
      float[][] outputArr = (float[][]) output.getValue();
      for (int i = 0; i < input.length; i++) {
        Assertions.assertArrayEquals(input[i], outputArr[i]);
      }
    }
  }

  @Test
  public void testFp32ToFp16() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String modelPath = TestHelpers.getResourcePath("/java-fp32-to-fp16.onnx").toString();
    SplittableRandom rng = new SplittableRandom(1);

    float[][] input = new float[10][5];
    FloatBuffer floatBuf =
        ByteBuffer.allocateDirect(4 * 10 * 5).order(ByteOrder.nativeOrder()).asFloatBuffer();
    ShortBuffer shortBuf = ShortBuffer.allocate(10 * 5);

    // Generate data
    for (int i = 0; i < input.length; i++) {
      for (int j = 0; j < input[0].length; j++) {
        int bits = rng.nextInt();
        input[i][j] = Float.intBitsToFloat(bits);
        floatBuf.put(input[i][j]);
        shortBuf.put(OrtUtil.floatToFp16(input[i][j]));
      }
    }
    floatBuf.rewind();
    shortBuf.rewind();

    try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, opts);
        OnnxTensor tensor = OnnxTensor.createTensor(env, floatBuf, new long[] {10, 5});
        OrtSession.Result result = session.run(Collections.singletonMap("input", tensor))) {
      OnnxTensor output = (OnnxTensor) result.get(0);

      // Check outbound Java side cast to fp32 works
      FloatBuffer castOutput = output.getFloatBuffer();
      float[] expectedFloatArr = new float[10 * 5];
      OrtUtil.convertFp16BufferToFloatBuffer(shortBuf).get(expectedFloatArr);
      float[] actualFloatArr = new float[10 * 5];
      castOutput.get(actualFloatArr);
      Assertions.assertArrayEquals(expectedFloatArr, actualFloatArr);

      // Check bits are correct
      ShortBuffer outputBuf = output.getShortBuffer();
      short[] expectedShortArr = new short[10 * 5];
      shortBuf.get(expectedShortArr);
      short[] actualShortArr = new short[10 * 5];
      outputBuf.get(actualShortArr);
      Assertions.assertArrayEquals(expectedShortArr, actualShortArr);
    }
  }

  @Test
  public void testFp32ToBf16() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String modelPath = TestHelpers.getResourcePath("/java-fp32-to-bf16.onnx").toString();
    SplittableRandom rng = new SplittableRandom(1);

    float[][] input = new float[10][5];
    FloatBuffer floatBuf =
        ByteBuffer.allocateDirect(4 * 10 * 5).order(ByteOrder.nativeOrder()).asFloatBuffer();
    ShortBuffer shortBuf = ShortBuffer.allocate(10 * 5);

    // Generate data
    for (int i = 0; i < input.length; i++) {
      for (int j = 0; j < input[0].length; j++) {
        int bits = rng.nextInt();
        input[i][j] = Float.intBitsToFloat(bits);
        floatBuf.put(input[i][j]);
        shortBuf.put(OrtUtil.floatToBf16(input[i][j]));
      }
    }
    floatBuf.rewind();
    shortBuf.rewind();

    try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, opts);
        OnnxTensor tensor = OnnxTensor.createTensor(env, floatBuf, new long[] {10, 5});
        OrtSession.Result result = session.run(Collections.singletonMap("input", tensor))) {
      OnnxTensor output = (OnnxTensor) result.get(0);

      // Check outbound Java side cast to fp32 works
      FloatBuffer castOutput = output.getFloatBuffer();
      float[] expectedFloatArr = new float[10 * 5];
      OrtUtil.convertBf16BufferToFloatBuffer(shortBuf).get(expectedFloatArr);
      float[] actualFloatArr = new float[10 * 5];
      castOutput.get(actualFloatArr);
      Assertions.assertArrayEquals(expectedFloatArr, actualFloatArr);

      // Check bits are correct
      ShortBuffer outputBuf = output.getShortBuffer();
      short[] expectedShortArr = new short[10 * 5];
      shortBuf.get(expectedShortArr);
      short[] actualShortArr = new short[10 * 5];
      outputBuf.get(actualShortArr);
      Assertions.assertArrayEquals(expectedShortArr, actualShortArr);
    }
  }

  @Test
  public void testFp16RoundTrip() {
    for (int i = 0; i < 0xffff; i++) {
      // Round trip every value
      short curVal = (short) (0xffff & i);
      float upcast = OrtUtil.mlasFp16ToFloat(curVal);
      short output = OrtUtil.mlasFloatToFp16(upcast);
      if (!Float.isNaN(upcast)) {
        // We coerce NaNs to the same value.
        Assertions.assertEquals(
            curVal,
            output,
            "Expected " + curVal + " received " + output + ", intermediate float was " + upcast);
      }
    }
  }

  @Test
  public void testBf16RoundTrip() {
    for (int i = 0; i < 0xffff; i++) {
      // Round trip every value
      short curVal = (short) (0xffff & i);
      float upcast = OrtUtil.bf16ToFloat(curVal);
      short output = OrtUtil.floatToBf16(upcast);
      if (!Float.isNaN(upcast)) {
        // We coerce NaNs to the same value.
        Assertions.assertEquals(
            curVal,
            output,
            "Expected " + curVal + " received " + output + ", intermediate float was " + upcast);
      }
    }
  }
}
