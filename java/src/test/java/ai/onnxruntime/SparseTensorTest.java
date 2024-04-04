/*
 * Copyright (c) 2021, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

public class SparseTensorTest {

  @Test
  public void testCSRC() throws OrtException {
    String modelPath =
        TestHelpers.getResourcePath("/generic_sparse_to_dense_matmul.onnx").toString();
    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
      try (OrtSession session = env.createSession(modelPath, options)) {
        Map<String, OnnxTensorLike> inputMap = new HashMap<>();

        OnnxTensor denseIdMatrix = TestHelpers.makeIdentityMatrix(env, 3);
        long[] shape = new long[] {3, 3};
        /*
         * Sparse matrix:
         * [
         *  0 1 0
         *  1 0 1
         *  4 0 6
         * ]
         */
        LongBuffer outerIndices =
            ByteBuffer.allocateDirect(4 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        outerIndices.put(0);
        outerIndices.put(1);
        outerIndices.put(3);
        outerIndices.put(5);
        outerIndices.rewind();
        LongBuffer innerIndices =
            ByteBuffer.allocateDirect(5 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        innerIndices.put(1);
        innerIndices.put(0);
        innerIndices.put(2);
        innerIndices.put(0);
        innerIndices.put(2);
        innerIndices.rewind();

        FloatBuffer data =
            ByteBuffer.allocateDirect(5 * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        data.put(1);
        data.put(1);
        data.put(1);
        data.put(4);
        data.put(6);
        data.rewind();

        OnnxSparseTensor.CSRCTensor csrcTensor =
            new OnnxSparseTensor.CSRCTensor(
                outerIndices, innerIndices, data, shape, OnnxJavaType.FLOAT, 5);
        OnnxSparseTensor tensor = OnnxSparseTensor.createSparseTensor(env, csrcTensor);

        inputMap.put("sparse_A", tensor);
        inputMap.put("dense_B", denseIdMatrix);

        OrtSession.Result result = session.run(inputMap);

        OnnxTensor outputTensor = (OnnxTensor) result.get(0);
        assertArrayEquals(shape, outputTensor.getInfo().getShape());
        float[] output = outputTensor.getFloatBuffer().array();
        float[] expected = new float[] {0, 1, 0, 1, 0, 1, 4, 0, 6};
        assertArrayEquals(expected, output, 1e-6f);
        result.close();
        inputMap.clear();

        // check that the get methods return new buffers which exist past the tensor lifetime.
        Buffer valuesOne = tensor.getValuesBuffer();
        Buffer valuesTwo = tensor.getValuesBuffer();
        Buffer indicesOne = tensor.getIndicesBuffer();
        Buffer indicesTwo = tensor.getIndicesBuffer();
        Buffer innerIndicesOne = tensor.getInnerIndicesBuffer();
        Buffer innerIndicesTwo = tensor.getInnerIndicesBuffer();
        tensor.close();
        assertEquals(valuesOne, valuesTwo);
        assertFalse(valuesOne == valuesTwo);
        assertEquals(indicesOne, indicesTwo);
        assertFalse(indicesOne == indicesTwo);
        assertEquals(innerIndicesOne, innerIndicesTwo);
        assertFalse(innerIndicesOne == innerIndicesTwo);

        long[] rectangularShape = new long[] {2, 3};
        /*
         * Sparse matrix:
         * [
         *   1 0 3
         *   0 5 6
         * ]
         */
        outerIndices =
            ByteBuffer.allocateDirect(3 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        outerIndices.put(0);
        outerIndices.put(2);
        outerIndices.put(4);
        outerIndices.rewind();
        innerIndices =
            ByteBuffer.allocateDirect(4 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        innerIndices.put(0);
        innerIndices.put(2);
        innerIndices.put(1);
        innerIndices.put(2);
        innerIndices.rewind();

        data = ByteBuffer.allocateDirect(4 * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        data.put(1);
        data.put(3);
        data.put(5);
        data.put(6);
        data.rewind();

        csrcTensor =
            new OnnxSparseTensor.CSRCTensor(
                outerIndices, innerIndices, data, rectangularShape, OnnxJavaType.FLOAT, 4);
        tensor = OnnxSparseTensor.createSparseTensor(env, csrcTensor);

        assertArrayEquals(new long[] {3}, tensor.getIndicesShape());
        assertArrayEquals(new long[] {4}, tensor.getInnerIndicesShape());
        assertArrayEquals(new long[] {4}, tensor.getValuesShape());

        inputMap.put("sparse_A", tensor);
        inputMap.put("dense_B", denseIdMatrix);

        result = session.run(inputMap);

        outputTensor = (OnnxTensor) result.get(0);
        assertArrayEquals(rectangularShape, outputTensor.getInfo().getShape());
        output = outputTensor.getFloatBuffer().array();
        expected = new float[] {1, 0, 3, 0, 5, 6};
        assertArrayEquals(expected, output, 1e-6f);
        result.close();
        tensor.close();
        inputMap.clear();
        denseIdMatrix.close();

        denseIdMatrix = TestHelpers.makeIdentityMatrix(env, 4);
        long[] vectorShape = new long[] {1, 4};
        /*
         * Sparse matrix:
         * [
         *   1 0 0 4
         * ]
         */
        outerIndices =
            ByteBuffer.allocateDirect(2 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        outerIndices.put(0);
        outerIndices.put(2);
        outerIndices.rewind();
        innerIndices =
            ByteBuffer.allocateDirect(2 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        innerIndices.put(0);
        innerIndices.put(3);
        innerIndices.rewind();

        data = ByteBuffer.allocateDirect(2 * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        data.put(1);
        data.put(4);
        data.rewind();

        csrcTensor =
            new OnnxSparseTensor.CSRCTensor(
                outerIndices, innerIndices, data, vectorShape, OnnxJavaType.FLOAT, 2);
        tensor = OnnxSparseTensor.createSparseTensor(env, csrcTensor);

        assertArrayEquals(new long[] {2}, tensor.getIndicesShape());
        assertArrayEquals(new long[] {2}, tensor.getInnerIndicesShape());
        assertArrayEquals(new long[] {2}, tensor.getValuesShape());

        inputMap.put("sparse_A", tensor);
        inputMap.put("dense_B", denseIdMatrix);

        result = session.run(inputMap);

        outputTensor = (OnnxTensor) result.get(0);
        assertArrayEquals(vectorShape, outputTensor.getInfo().getShape());
        output = outputTensor.getFloatBuffer().array();
        expected = new float[] {1, 0, 0, 4};
        assertArrayEquals(expected, output, 1e-6f);
        result.close();
        tensor.close();
        inputMap.clear();
        denseIdMatrix.close();
      }
    }
  }

  @Test
  public void testCOO() throws OrtException {
    String modelPath =
        TestHelpers.getResourcePath("/generic_sparse_to_dense_matmul.onnx").toString();
    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
      try (OrtSession session = env.createSession(modelPath, options)) {
        Map<String, OnnxTensorLike> inputMap = new HashMap<>();

        OnnxTensor denseIdMatrix = TestHelpers.makeIdentityMatrix(env, 3);
        long[] shape = new long[] {3, 3};
        /*
         * Sparse matrix:
         * [
         *  0 1 0
         *  1 0 1
         *  4 0 6
         * ]
         */
        LongBuffer indices =
            ByteBuffer.allocateDirect(2 * 5 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        indices.put(0);
        indices.put(1);
        indices.put(1);
        indices.put(0);
        indices.put(1);
        indices.put(2);
        indices.put(2);
        indices.put(0);
        indices.put(2);
        indices.put(2);
        indices.rewind();

        FloatBuffer data =
            ByteBuffer.allocateDirect(5 * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        data.put(1);
        data.put(1);
        data.put(1);
        data.put(4);
        data.put(6);
        data.rewind();

        OnnxSparseTensor.COOTensor cooTensor =
            new OnnxSparseTensor.COOTensor(
                indices, new long[] {5, 2}, data, shape, OnnxJavaType.FLOAT, 5);
        OnnxSparseTensor tensor = OnnxSparseTensor.createSparseTensor(env, cooTensor);

        inputMap.put("sparse_A", tensor);
        inputMap.put("dense_B", denseIdMatrix);

        OrtSession.Result result = session.run(inputMap);

        OnnxTensor outputTensor = (OnnxTensor) result.get(0);
        assertArrayEquals(shape, outputTensor.getInfo().getShape());
        float[] output = outputTensor.getFloatBuffer().array();
        float[] expected = new float[] {0, 1, 0, 1, 0, 1, 4, 0, 6};
        assertArrayEquals(expected, output, 1e-6f);
        result.close();
        tensor.close();
        inputMap.clear();

        /* disabled as sparse_dense_matmul doesn't support COO tensors with 1d indices
        // Run the same tensor through, but using 1d indexing rather than 2d indexing
        indices = ByteBuffer.allocateDirect(5 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        indices.put(1);
        indices.put(3);
        indices.put(5);
        indices.put(6);
        indices.put(8);
        indices.rewind();

        cooTensor = new OnnxSparseTensor.COOTensor(indices, new long[]{5}, data, shape, OnnxJavaType.FLOAT, 5);
        tensor = OnnxSparseTensor.createSparseTensor(env, cooTensor);

        inputMap.put("sparse_A", tensor);
        inputMap.put("dense_B", denseIdMatrix);

        result = session.run(inputMap);

        outputTensor = (OnnxTensor) result.get(0);
        assertArrayEquals(shape, outputTensor.getInfo().getShape());
        output = outputTensor.getFloatBuffer().array();
        assertArrayEquals(expected, output, 1e-6f);
        result.close();
        tensor.close();
        inputMap.clear();
         */

        long[] rectangularShape = new long[] {2, 3};
        /*
         * Sparse matrix:
         * [
         *   1 0 3
         *   0 5 6
         * ]
         */
        indices =
            ByteBuffer.allocateDirect(2 * 4 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        indices.put(0);
        indices.put(0);
        indices.put(0);
        indices.put(2);
        indices.put(1);
        indices.put(1);
        indices.put(1);
        indices.put(2);
        indices.rewind();

        data = ByteBuffer.allocateDirect(4 * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        data.put(1);
        data.put(3);
        data.put(5);
        data.put(6);
        data.rewind();

        cooTensor =
            new OnnxSparseTensor.COOTensor(
                indices, new long[] {4, 2}, data, rectangularShape, OnnxJavaType.FLOAT, 4);
        tensor = OnnxSparseTensor.createSparseTensor(env, cooTensor);

        assertArrayEquals(new long[] {4, 2}, tensor.getIndicesShape());
        assertArrayEquals(new long[] {4}, tensor.getValuesShape());

        inputMap.put("sparse_A", tensor);
        inputMap.put("dense_B", denseIdMatrix);

        result = session.run(inputMap);

        outputTensor = (OnnxTensor) result.get(0);
        assertArrayEquals(rectangularShape, outputTensor.getInfo().getShape());
        output = outputTensor.getFloatBuffer().array();
        expected = new float[] {1, 0, 3, 0, 5, 6};
        assertArrayEquals(expected, output, 1e-6f);
        result.close();
        tensor.close();
        inputMap.clear();
        denseIdMatrix.close();

        denseIdMatrix = TestHelpers.makeIdentityMatrix(env, 4);
        long[] vectorShape = new long[] {1, 4};
        /*
         * Sparse matrix:
         * [
         *   1
         *   0
         *   0
         *   4
         * ]
         */
        indices = ByteBuffer.allocateDirect(4 * 8).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
        indices.put(0);
        indices.put(0);
        indices.put(0);
        indices.put(3);
        indices.rewind();

        data = ByteBuffer.allocateDirect(2 * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        data.put(1);
        data.put(4);
        data.rewind();

        cooTensor =
            new OnnxSparseTensor.COOTensor(
                indices, new long[] {2, 2}, data, vectorShape, OnnxJavaType.FLOAT, 2);
        tensor = OnnxSparseTensor.createSparseTensor(env, cooTensor);

        assertArrayEquals(new long[] {2, 2}, tensor.getIndicesShape());
        assertArrayEquals(new long[] {2}, tensor.getValuesShape());

        inputMap.put("sparse_A", tensor);
        inputMap.put("dense_B", denseIdMatrix);

        result = session.run(inputMap);

        outputTensor = (OnnxTensor) result.get(0);
        assertArrayEquals(vectorShape, outputTensor.getInfo().getShape());
        output = outputTensor.getFloatBuffer().array();
        expected = new float[] {1, 0, 0, 4};
        assertArrayEquals(expected, output, 1e-6f);
        result.close();
        tensor.close();
        inputMap.clear();
        denseIdMatrix.close();
      }
    }
  }

  @Test
  public void testCOOOutput() throws OrtException {
    String modelPath = TestHelpers.getResourcePath("/sparse_initializer_as_output.onnx").toString();
    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
      try (OrtSession session = env.createSession(modelPath, options)) {
        Map<String, NodeInfo> outputs = session.getOutputInfo();
        assertEquals(1, outputs.size());

        NodeInfo info = outputs.get("values");
        assertNotNull(info);
        assertTrue(info.getInfo() instanceof TensorInfo);

        TensorInfo outputInfo = (TensorInfo) info.getInfo();
        assertArrayEquals(new long[] {3, 3}, outputInfo.getShape());
        assertEquals(
            TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, outputInfo.onnxType);
        assertEquals(OnnxJavaType.FLOAT, outputInfo.type);

        OrtSession.Result result = session.run(Collections.emptyMap());
        OnnxValue output = result.get("values").get();

        assertTrue(output instanceof OnnxSparseTensor);

        OnnxSparseTensor sparseTensor = (OnnxSparseTensor) output;

        assertEquals(OnnxSparseTensor.SparseTensorType.COO, sparseTensor.getSparseTensorType());

        assertArrayEquals(new long[] {3}, sparseTensor.getIndicesShape());
        assertArrayEquals(new long[] {3}, sparseTensor.getValuesShape());
        assertArrayEquals(new long[] {3, 3}, sparseTensor.getInfo().getShape());

        OnnxSparseTensor.SparseTensor<? extends Buffer> javaTensor = sparseTensor.getValue();

        assertTrue(javaTensor instanceof OnnxSparseTensor.COOTensor);

        OnnxSparseTensor.COOTensor cooTensor = (OnnxSparseTensor.COOTensor) javaTensor;

        long[] indices = new long[3];
        cooTensor.getIndices().get(indices);
        float[] data = new float[3];
        ((FloatBuffer) cooTensor.getValues()).get(data);

        assertArrayEquals(new long[] {2, 3, 5}, indices);
        assertArrayEquals(
            new float[] {1.764052391052246f, 0.40015721321105957f, 0.978738009929657f}, data);
      }
    }
  }
}
