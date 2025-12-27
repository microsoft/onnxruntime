/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

package ai.onnxruntime;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.logging.Logger;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledForJreRange;
import org.junit.jupiter.api.condition.JRE;

/** Tests for interop with Java 22's MemorySegments. */
@EnabledForJreRange(min = JRE.JAVA_22)
public class MemorySegmentTest {
  /** Shim so the tests can create memory segments from an arena. */
  static final class ArenaShim implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(ArenaShim.class.getName());
    private static final Class<?> arenaClass;

    /*
     * Method handles that bind to methods on java.lang.foreign.MemorySegment.
     */
    private static final MethodHandle ofConfined;
    private static final MethodHandle allocate;
    private static final MethodHandle close;

    private final Object arena;

    static {
      Class<?> tmpArenaClass;
      MethodHandle tmpOfConfined;
      MethodHandle tmpAllocate;
      MethodHandle tmpClose;
      MethodHandles.Lookup lookup = MethodHandles.lookup();
      try {
        tmpArenaClass = Class.forName("java.lang.foreign.Arena");
        Class<?> segmentClass = Class.forName("java.lang.foreign.MemorySegment");
        tmpOfConfined =
            lookup.findStatic(tmpArenaClass, "ofConfined", MethodType.methodType(tmpArenaClass));
        tmpAllocate =
            lookup.findVirtual(
                tmpArenaClass, "allocate", MethodType.methodType(segmentClass, long.class));
        tmpClose = lookup.findVirtual(tmpArenaClass, "close", MethodType.methodType(void.class));
      } catch (IllegalAccessException | NoSuchMethodException | ClassNotFoundException e) {
        logger.info("Running on Java 21 or earlier, Arena not available");
        tmpArenaClass = null;
        tmpOfConfined = null;
        tmpAllocate = null;
        tmpClose = null;
      }
      arenaClass = tmpArenaClass;
      ofConfined = tmpOfConfined;
      allocate = tmpAllocate;
      close = tmpClose;
    }

    private ArenaShim(Object arena) {
      this.arena = arena;
    }

    static ArenaShim ofConfined() {
      if (arenaClass != null) {
        try {
          return new ArenaShim(ofConfined.invoke());
        } catch (Throwable e) {
          throw new AssertionError("Should not reach here", e);
        }
      } else {
        throw new UnsupportedOperationException("java.lang.foreign.Arena is not available.");
      }
    }

    Object allocate(long size) {
      if (arenaClass != null) {
        try {
          return allocate.invoke(arena, size);
        } catch (Throwable e) {
          throw new AssertionError("Should not reach here", e);
        }
      } else {
        throw new UnsupportedOperationException("java.lang.foreign.Arena is not available.");
      }
    }

    public void close() {
      if (arenaClass != null) {
        try {
          close.invoke(arena);
        } catch (Throwable e) {
          throw new AssertionError("Should not reach here", e);
        }
      } else {
        throw new UnsupportedOperationException("java.lang.foreign.Arena is not available.");
      }
    }
  }

  @Test
  public void testSegments() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    // Construct a big segment.
    try (ArenaShim arena = ArenaShim.ofConfined()) {
      // e.g., a 256k vocab with 4096 embedding dimensions
      long[] shape = new long[] {256 * 1024, 4 * 1024};
      MemorySegmentShim segment =
          new MemorySegmentShim(arena.allocate(4L * OrtUtil.elementCount(shape)));
      // Fill segment with appropriate values
      for (int i = 0; i < 256 * 1024; i++) {
        float floati = (float) i;
        for (int j = 0; j < 4096; j++) {
          segment.set(4L * i * 4096L + 4L * j, floati);
        }
      }
      OnnxTensor bigTensor =
          OnnxTensor.createTensorFromMemorySegment(env, segment.get(), shape, OnnxJavaType.FLOAT);

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

      Object refMemorySegment = bigTensor.getSegmentRef().get();
      Assertions.assertSame(segment.get(), refMemorySegment);

      Object otherMemorySegment = bigTensor.getSegment();
      Assertions.assertEquals(segment.get(), otherMemorySegment);
    }
  }

  @Test
  public void testSmallSegment() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    // Construct a small segment.
    try (ArenaShim arena = ArenaShim.ofConfined()) {
      long[] shape = new long[] {5, 4};
      MemorySegmentShim segment =
          new MemorySegmentShim(arena.allocate(4L * OrtUtil.elementCount(shape)));
      // Fill segment with appropriate values
      for (int i = 0; i < 5; i++) {
        float floati = (float) i;
        for (int j = 0; j < 4; j++) {
          segment.set(4L * i * 4L + 4L * j, floati);
        }
      }
      OnnxTensor smallTensor =
          OnnxTensor.createTensorFromMemorySegment(env, segment.get(), shape, OnnxJavaType.FLOAT);

      FloatBuffer fb = smallTensor.getFloatBuffer();

      float[][] arr = (float[][]) smallTensor.getValue();

      float[] fbArr = new float[fb.remaining()];
      fb.get(fbArr);
      float[][] reshaped = (float[][]) OrtUtil.reshape(fbArr, shape);
      Assertions.assertArrayEquals(arr, reshaped);

      Object refMemorySegment = smallTensor.getSegmentRef().get();
      Assertions.assertSame(segment.get(), refMemorySegment);

      Object otherMemorySegment = smallTensor.getSegment();
      Assertions.assertEquals(segment.get(), otherMemorySegment);
      Assertions.assertNotSame(segment.get(), otherMemorySegment);
    }
  }

  @Test
  public void testModel() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String modelPath = TestHelpers.getResourcePath("/java-external-embedding.onnx").toString();

    // Construct segment for use as embedding parameters.
    try (ArenaShim arena = ArenaShim.ofConfined()) {
      // i.e. a 256k vocab with 4096 embedding dimensions
      long[] shape = new long[] {256 * 1024, 4 * 1024};
      MemorySegmentShim segment =
          new MemorySegmentShim(arena.allocate(4L * OrtUtil.elementCount(shape)));
      // Fill segment with appropriate values
      for (int i = 0; i < 256 * 1024; i++) {
        float floati = (float) i;
        for (int j = 0; j < 4096; j++) {
          segment.set(4L * i * 4096L + 4L * j, floati);
        }
      }
      OnnxTensor embedding =
          OnnxTensor.createTensorFromMemorySegment(env, segment.get(), shape, OnnxJavaType.FLOAT);

      // Construct input tensor
      long[][] inputArr =
          new long[][] {{64, 128, 256, 512, 0, 0, 0}, {1, 2, 3, 4, 5, 6, 128 * 1024}};
      OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArr);

      // Construct model using external initializer.
      try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
        opts.addExternalInitializers(Collections.singletonMap("embedding", embedding));

        // Run model
        try (OrtSession session = env.createSession(modelPath, opts);
            OrtSession.Result output =
                session.run(Collections.singletonMap("input", inputTensor))) {
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
