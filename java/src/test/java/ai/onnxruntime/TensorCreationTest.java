/*
 * Copyright (c) 2021, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TensorCreationTest {

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
}
