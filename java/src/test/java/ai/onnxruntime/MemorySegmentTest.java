/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

package ai.onnxruntime;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.Map;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/** Tests for interop with Java 22's MemorySegments. */
public class MemorySegmentTest {

  @Test
  public void testSegments() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    // Construct a big segment.
    try (Arena arena = Arena.ofConfined()) {
      // e.g., a 256k vocab with 4096 embedding dimensions
      long[] shape = new long[] {256 * 1024, 4 * 1024};
      MemorySegment segment = arena.allocate(4L * OrtUtil.elementCount(shape));
      // Fill segment with appropriate values
      for (int i = 0; i < 256 * 1024; i++) {
        float floati = (float) i;
        for (int j = 0; j < 4096; j++) {
          segment.set(ValueLayout.JAVA_FLOAT, 4L * i * 4096L + 4L * j, floati);
        }
      }
      OnnxTensor bigTensor = OnnxTensor.createTensor(env, segment, shape, OnnxJavaType.FLOAT);

      try {
        FloatBuffer fb = bigTensor.getFloatBuffer();
        Assertions.fail("Should have thrown an exception");
      } catch (OrtException e) {
        Assertions.assertTrue(
            e.getMessage().contains("Cannot construct a java.nio.Buffer of this size."));
      }

      try {
        float[][] arr = (float[][]) bigTensor.getValue();
        Assertions.fail("Should have thrown an exception");
      } catch (OrtException e) {
        Assertions.assertTrue(
            e.getMessage().contains("This tensor is not representable in Java, it's too big"));
      }

      MemorySegment ref = (MemorySegment) bigTensor.getSegmentRef().get();
      Assertions.assertSame(segment, ref);

      MemorySegment other = (MemorySegment) bigTensor.getSegment();
      Assertions.assertEquals(segment, other);
    }
  }

  @Test
  public void testSmallSegment() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    // Construct a small segment.
    try (Arena arena = Arena.ofConfined()) {
      long[] shape = new long[] {5, 4};
      MemorySegment segment = arena.allocate(4L * OrtUtil.elementCount(shape));
      // Fill segment with appropriate values
      for (int i = 0; i < 5; i++) {
        float floati = (float) i;
        for (int j = 0; j < 4; j++) {
          segment.set(ValueLayout.JAVA_FLOAT, 4L * i * 4L + 4L * j, floati);
        }
      }
      OnnxTensor smallTensor = OnnxTensor.createTensor(env, segment, shape, OnnxJavaType.FLOAT);

      FloatBuffer fb = smallTensor.getFloatBuffer();

      float[][] arr = (float[][]) smallTensor.getValue();

      float[] fbArr = new float[fb.remaining()];
      fb.get(fbArr);
      float[][] reshaped = (float[][]) OrtUtil.reshape(fbArr, shape);
      Assertions.assertArrayEquals(arr, reshaped);

      MemorySegment ref = (MemorySegment) smallTensor.getSegmentRef().get();
      Assertions.assertSame(segment, ref);

      MemorySegment other = (MemorySegment) smallTensor.getSegment();
      Assertions.assertEquals(segment, other);
      Assertions.assertNotSame(segment, other);
    }
  }

  @Test
  public void testModel() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String modelPath = TestHelpers.getResourcePath("/java-external-embedding.onnx").toString();

    // Construct segment for use as embedding parameters.
    try (Arena arena = Arena.ofConfined()) {
      // i.e. a 256k vocab with 4096 embedding dimensions
      long[] shape = new long[] {256 * 1024, 4 * 1024};
      MemorySegment segment = arena.allocate(4L * OrtUtil.elementCount(shape));
      // Fill segment with appropriate values
      for (int i = 0; i < 256 * 1024; i++) {
        float floati = (float) i;
        for (int j = 0; j < 4096; j++) {
          segment.set(ValueLayout.JAVA_FLOAT, 4L * i * 4096L + 4L * j, floati);
        }
      }
      OnnxTensor embedding = OnnxTensor.createTensor(env, segment, shape, OnnxJavaType.FLOAT);

      // Construct input tensor
      long[][] inputArr =
          new long[][] {{64, 128, 256, 512, 0, 0, 0}, {1, 2, 3, 4, 5, 6, 128 * 1024}};
      OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArr);

      // Construct model using external initializer.
      try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
        opts.addExternalInitializers(Map.of("embedding", embedding));

        // Run model
        try (OrtSession session = env.createSession(modelPath, opts);
            OrtSession.Result output = session.run(Map.of("input", inputTensor))) {
          // Validate output, which is [batch_size, seq_length, embedding_dimension]
          // The embedding values should be filled with the index of that embedding
          float[][][] outputArr = (float[][][]) output.get("output").get().getValue();
          Assertions.assertEquals(2, outputArr.length);
          Assertions.assertEquals(7, outputArr[0].length);
          Assertions.assertEquals(4096, outputArr[0][0].length);
          for (int i = 0; i < inputArr.length; i++) {
            for (int j = 0; j < inputArr[0].length; j++) {
              float testVal = inputArr[i][j];
              for (int k = 0; k < 4096; k++) {
                Assertions.assertEquals(
                    testVal,
                    outputArr[i][j][k],
                    "At position [" + i + "," + j + "," + k + "] values differ.");
              }
            }
          }
        }
      }
    }
  }
}
