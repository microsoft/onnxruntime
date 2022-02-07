/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;

/**
 * A Java object wrapping an OnnxSparseTensor.
 * <p> Sparse tensors support a variety of formats, and the {@link #getValue} method
 * returns a different static inner class representing each type.
 */
public final class OnnxSparseTensor implements OnnxValue {
    static {
        try {
            OnnxRuntime.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load onnx-runtime library", e);
        }
    }

    private final long nativeHandle;

    private final long allocatorHandle;

    private final TensorInfo info;

    OnnxSparseTensor(long nativeHandle, long allocatorHandle, TensorInfo info) {
        this.nativeHandle = nativeHandle;
        this.allocatorHandle = allocatorHandle;
        this.info = info;
    }

    public static OnnxSparseTensor createSparseTensor(OrtEnvironment env, SparseTensor tensor) {
        return createSparseTensor(env, env.defaultAllocator, tensor);
    }

    static OnnxSparseTensor createSparseTensor(OrtEnvironment env, OrtAllocator allocator, SparseTensor tensor) {
        if ((!env.isClosed()) && (!allocator.isClosed())) {
            return createSparseTensor(type, allocator, tensor.getSparityType().value, tensor.getShape());
        } else {
            throw new IllegalStateException("Trying to create an OnnxSparseTensor on a closed OrtAllocator.");
        }
    }

    @Override
    public OnnxValueType getType() {
        return OnnxValueType.ONNX_TYPE_SPARSETENSOR;
    }

    /**
     * Returns the native pointer.
     * @return The native pointer.
     */
    long getNativeHandle() {
        return nativeHandle;
    }

    @Override
    public SparseTensor getValue() throws OrtException {
        return null;
    }

    @Override
    public ValueInfo getInfo() {
        return null;
    }

    @Override
    public void close() {
        close(OnnxRuntime.ortApiHandle, nativeHandle);
    }

    /**
     * Wraps the indices in a direct byte buffer.
     *
     * @param apiHandle The OrtApi pointer.
     * @param nativeHandle The OrtSparseTensor pointer.
     * @return A ByteBuffer wrapping the indices.
     */
    private native ByteBuffer getIndexBuffer(long apiHandle, long nativeHandle);

    /**
     * Wraps the data in a direct byte buffer.
     *
     * @param apiHandle The OrtApi pointer.
     * @param nativeHandle The OrtSparseTensor pointer.
     * @return A ByteBuffer wrapping the indices.
     */
    private native ByteBuffer getDataBuffer(long apiHandle, long nativeHandle);

    private native void close(long apiHandle, long nativeHandle);

    private static native long createSparseTensorFromBuffer(
            long apiHandle,
            long allocatorHandle,
            Buffer indexData,
            int indexBufferPos,
            long indexBufferSize,
            Buffer data,
            int bufferPos,
            long bufferSize,
            long[] shape,
            int onnxType,
            int sparsityType)
            throws OrtException;

    /**
     * The type of the sparse tensor.
     *
     * <p> Should be synchronized with OrtSparseFormat in the C API.
     */
    public enum SparseTensorType {
        /**
         * Undefined sparse tensor.
         */
        UNDEFINED(0),
        /**
         * COO sparse tensor.
         */
        COO(1),
        /**
         * CSR or CSC sparse tensor.
         */
        CSRC(2),
        /**
         * Block sparse tensor.
         */
        BLOCK_SPARSE(4);

        /**
         * The int value mirroring OrtSparseFormat.
         */
        public final int value;

        private SparseTensorType(int value) {
            this.value = value;
        }
    }

    /**
     * Top level interfaces for the Java side representation of a sparse tensor.
     * <p> Will be sealed to {@link COOTensor}, {@link CSRCTensor} and {@link BlockSparseTensor} when possible.
     */
    public static interface SparseTensor {
        /**
         * Gets the shape of the sparse tensor.
         * @return The sparse tensor shape.
         */
        public long[] getShape();

        /**
         * The data type of the sparse tensor.
         * @return The sparse tensor data type.
         */
        public OnnxJavaType getType();

        /**
         * The sparsity type of the sparse tensor.
         * @return The sparse tensor sparsity type.
         */
        public SparseTensorType getSparityType();

        /**
         * The number of non-zero elements.
         * @return The number of non-zero elements.
         */
        public long getNumNonZeroElements();
    }

    public static final class COOTensor implements SparseTensor {
        public SparseTensorType getSparityType() {
            return SparseTensorType.COO;
        }
    }

    public static final class CSRCTensor implements SparseTensor {
        public SparseTensorType getSparityType() {
            return SparseTensorType.CSRC;
        }
    }

    public static final class BlockSparseTensor implements SparseTensor {
        public SparseTensorType getSparityType() {
            return SparseTensorType.BLOCK_SPARSE;
        }
    }

}
