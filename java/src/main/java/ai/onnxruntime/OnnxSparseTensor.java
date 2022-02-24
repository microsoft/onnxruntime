/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

/**
 * A Java object wrapping an OnnxSparseTensor.
 *
 * <p>Sparse tensors support a variety of formats, and the {@link #getValue} method returns a
 * different static inner class representing each type.
 */
public final class OnnxSparseTensor extends OnnxTensorLike {
  private final SparseTensorType sparseTensorType;

  // Held to prevent deallocation while used in native code.
  private final Buffer indices;
  private final LongBuffer innerIndices;
  private final Buffer data;

  /**
   * Construct a sparse tensor from JNI.
   *
   * @param nativeHandle The tensor native handle.
   * @param allocatorHandle The allocator handle.
   * @param sparseType The sparsity type.
   * @param info The tensor info.
   */
  OnnxSparseTensor(
      long nativeHandle, long allocatorHandle, SparseTensorType sparseType, TensorInfo info) {
    this(nativeHandle, allocatorHandle, sparseType, info, null, null, null);
  }

  /**
   * Construct a COO or block sparse tensor.
   *
   * @param nativeHandle The tensor native handle.
   * @param allocatorHandle The allocator handle.
   * @param sparseType The sparsity type.
   * @param info The tensor info.
   * @param indices The indices buffer.
   * @param data The data buffer.
   */
  OnnxSparseTensor(
      long nativeHandle,
      long allocatorHandle,
      SparseTensorType sparseType,
      TensorInfo info,
      Buffer indices,
      Buffer data) {
    this(nativeHandle, allocatorHandle, sparseType, info, indices, null, data);
  }

  /**
   * Construct a sparse tensor.
   *
   * <p>If the tensor is COO or block sparse then innerIndices may be null.
   *
   * @param nativeHandle The tensor native handle.
   * @param allocatorHandle The allocator handle.
   * @param sparseType The sparsity type.
   * @param info The tensor info.
   * @param indices The indices buffer.
   * @param innerIndices The inner indices buffer.
   * @param data The data buffer.
   */
  OnnxSparseTensor(
      long nativeHandle,
      long allocatorHandle,
      SparseTensorType sparseType,
      TensorInfo info,
      Buffer indices,
      LongBuffer innerIndices,
      Buffer data) {
    super(nativeHandle, allocatorHandle, info);
    this.sparseTensorType = sparseType;
    this.indices = indices;
    this.innerIndices = innerIndices;
    this.data = data;
  }

  /**
   * Creates a Sparse Tensor in ORT from the Java side representation.
   *
   * @param env The OrtEnvironment.
   * @param tensor The Java side representation.
   * @return The sparse tensor in ORT.
   * @throws OrtException If the tensor could not be created or was invalid.
   */
  public static OnnxSparseTensor createSparseTensor(OrtEnvironment env, SparseTensor tensor)
      throws OrtException {
    return createSparseTensor(env, env.defaultAllocator, tensor);
  }

  static OnnxSparseTensor createSparseTensor(
      OrtEnvironment env, OrtAllocator allocator, SparseTensor tensor) throws OrtException {
    if ((!env.isClosed()) && (!allocator.isClosed())) {
      TensorInfo info = TensorInfo.constructFromSparseTensor(tensor);
      OnnxJavaType indicesType = tensor.getIndicesType();
      OrtUtil.BufferTuple indicesTuple = OrtUtil.prepareBuffer(tensor.getIndices(), indicesType);
      OrtUtil.BufferTuple dataTuple = OrtUtil.prepareBuffer(tensor.getData(), info.type);
      if (!((indicesTuple.data instanceof LongBuffer)
          || (indicesTuple.data instanceof IntBuffer))) {
        throw new IllegalStateException(
            "Unexpected type of indices buffer, found "
                + indicesTuple.data.getClass()
                + ", expected IntBuffer or LongBuffer");
      }
      // Replace with a type switch when using JDK 17+.
      switch (tensor.getSparsityType()) {
        case COO:
        case BLOCK_SPARSE:
          return new OnnxSparseTensor(
              createSparseTensorFromBuffer(
                  OnnxRuntime.ortApiHandle,
                  allocator.handle,
                  indicesTuple.data,
                  indicesTuple.pos,
                  indicesTuple.size,
                  dataTuple.data,
                  dataTuple.pos,
                  info.shape,
                  tensor.getIndicesShape(),
                  tensor.getValuesShape(),
                  info.onnxType.value,
                  tensor.getSparsityType().value),
              allocator.handle,
              tensor.getSparsityType(),
              info,
              indicesTuple.data,
              dataTuple.data);
        case CSRC:
          OrtUtil.BufferTuple innerIndicesTuple =
              OrtUtil.prepareBuffer(((CSRCTensor) tensor).getInnerIndices(), indicesType);
          return new OnnxSparseTensor(
              createCSRCSparseTensorFromBuffer(
                  OnnxRuntime.ortApiHandle,
                  allocator.handle,
                  indicesTuple.data,
                  indicesTuple.pos,
                  indicesTuple.size,
                  innerIndicesTuple.data,
                  innerIndicesTuple.pos,
                  innerIndicesTuple.size,
                  dataTuple.data,
                  dataTuple.pos,
                  info.shape,
                  tensor.getValuesShape(),
                  info.onnxType.value),
              allocator.handle,
              tensor.getSparsityType(),
              info,
              indicesTuple.data,
              dataTuple.data);
        case UNDEFINED:
        default:
          throw new IllegalArgumentException("Cannot create an UNDEFINED sparse tensor.");
      }
    } else {
      throw new IllegalStateException(
          "Trying to create an OnnxSparseTensor on a closed OrtAllocator.");
    }
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_SPARSETENSOR;
  }

  @Override
  public SparseTensor getValue() throws OrtException {
    Buffer buffer = getDataBuffer();
    long[] indicesShape = getIndicesShape(OnnxRuntime.ortApiHandle, allocatorHandle, nativeHandle);
    switch (sparseTensorType) {
      case COO:
        return new COOTensor(
            (LongBuffer) getIndexBuffer(),
            indicesShape,
            buffer,
            info.shape,
            info.type,
            buffer.remaining());
      case CSRC:
        return new CSRCTensor(
            (LongBuffer) getIndexBuffer(),
            getInnerIndexBuffer(),
            buffer,
            info.shape,
            info.type,
            buffer.remaining());
      case BLOCK_SPARSE:
        long[] valuesShape =
            getValuesShape(OnnxRuntime.ortApiHandle, allocatorHandle, nativeHandle);
        return new BlockSparseTensor(
            (IntBuffer) getIndexBuffer(),
            indicesShape,
            buffer,
            valuesShape,
            info.shape,
            info.type,
            buffer.remaining());
      case UNDEFINED:
      default:
        throw new IllegalStateException("Undefined sparsity type in this sparse tensor.");
    }
  }

  @Override
  public void close() {
    close(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Returns the type of this OnnxSparseTensor.
   *
   * @return The sparsity type.
   */
  public SparseTensorType getSparseTensorType() {
    return sparseTensorType;
  }

  /**
   * Gets a copy of the indices.
   *
   * <p>These are the outer indices if it's a CSRC sparse tensor.
   *
   * <p>It's a {@link LongBuffer} if COO or CSRC, and {@link IntBuffer} if Block Sparse.
   *
   * @return The indices.
   */
  public Buffer getIndexBuffer() {
    switch (sparseTensorType) {
      case COO:
      case CSRC:
        return getIndexBuffer(OnnxRuntime.ortApiHandle, nativeHandle)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer();
      case BLOCK_SPARSE:
        return getIndexBuffer(OnnxRuntime.ortApiHandle, nativeHandle)
            .order(ByteOrder.nativeOrder())
            .asIntBuffer();
      case UNDEFINED:
      default:
        throw new IllegalStateException("UNDEFINED sparse tensor type.");
    }
  }

  /**
   * Gets a copy of the inner indices in a CSRC sparse tensor.
   *
   * <p>Throws {@link IllegalStateException} if called on a different sparse tensor type.
   *
   * @return The inner indices.
   */
  public LongBuffer getInnerIndexBuffer() {
    if (sparseTensorType == SparseTensorType.CSRC) {
      return getInnerIndexBuffer(OnnxRuntime.ortApiHandle, nativeHandle)
          .order(ByteOrder.nativeOrder())
          .asLongBuffer();
    } else {
      throw new IllegalStateException(
          "Inner indices are only available for CSRC sparse tensors, this sparse tensor is "
              + sparseTensorType);
    }
  }

  /**
   * Gets a copy of the data buffer.
   *
   * @return The data buffer.
   */
  public Buffer getDataBuffer() {
    ByteBuffer buffer =
        getDataBuffer(OnnxRuntime.ortApiHandle, nativeHandle).order(ByteOrder.nativeOrder());
    switch (info.type) {
      case FLOAT:
        return buffer.asFloatBuffer();
      case DOUBLE:
        return buffer.asDoubleBuffer();
      case INT16:
        return buffer.asShortBuffer();
      case INT32:
        return buffer.asIntBuffer();
      case INT64:
        return buffer.asLongBuffer();
      case BOOL:
      case INT8:
      case UINT8:
        return buffer;
      case STRING:
        throw new IllegalStateException("Unsupported data type String");
      case UNKNOWN:
      default:
        throw new IllegalStateException("Unsupported data type");
    }
  }

  /**
   * Gets the shape of the (outer) indices.
   *
   * @return The indices shape.
   */
  public long[] getIndicesShape() {
    return getIndicesShape(OnnxRuntime.ortApiHandle, allocatorHandle, nativeHandle);
  }

  /**
   * Gets the shape of the inner indices in a CSRC sparse tensor.
   *
   * @return The indices shape.
   */
  public long[] getInnerIndicesShape() {
    if (sparseTensorType == SparseTensorType.CSRC) {
      return getInnerIndicesShape(OnnxRuntime.ortApiHandle, allocatorHandle, nativeHandle);
    } else {
      throw new IllegalStateException(
          "Inner indices are only available for CSRC sparse tensors, this sparse tensor is "
              + sparseTensorType);
    }
  }

  /**
   * Gets the shape of the values.
   *
   * @return The values shape.
   */
  public long[] getValuesShape() {
    return getValuesShape(OnnxRuntime.ortApiHandle, allocatorHandle, nativeHandle);
  }

  /**
   * Gets the shape of the (outer) indices.
   *
   * @param apiHandle The OrtApi pointer.
   * @param allocatorHandle The memory allocator pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return The indices shape.
   */
  private native long[] getIndicesShape(long apiHandle, long allocatorHandle, long nativeHandle);

  /**
   * Gets the shape of the inner indices.
   *
   * @param apiHandle The OrtApi pointer.
   * @param allocatorHandle The memory allocator pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return The inner indices shape.
   */
  private native long[] getInnerIndicesShape(
      long apiHandle, long allocatorHandle, long nativeHandle);

  /**
   * Gets the shape of the values.
   *
   * @param apiHandle The OrtApi pointer.
   * @param allocatorHandle The memory allocator pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return The values shape.
   */
  private native long[] getValuesShape(long apiHandle, long allocatorHandle, long nativeHandle);

  /**
   * Wraps the indices in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the indices.
   */
  private native ByteBuffer getIndexBuffer(long apiHandle, long nativeHandle);

  /**
   * Wraps the inner indices in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the inner indices.
   */
  private native ByteBuffer getInnerIndexBuffer(long apiHandle, long nativeHandle);

  /**
   * Wraps the data in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the indices.
   */
  private native ByteBuffer getDataBuffer(long apiHandle, long nativeHandle);

  /**
   * Closes the sparse tensor.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   */
  private native void close(long apiHandle, long nativeHandle);

  /**
   * Creates a sparse CSRC sparse tensor.
   *
   * @param apiHandle The ORT API pointer.
   * @param allocatorHandle The allocator pointer.
   * @param indexData The outer indices.
   * @param indexBufferPos The outer indices position in bytes.
   * @param indexBufferSize The outer indices buffer size in longs.
   * @param innerIndexData The inner indices.
   * @param innerIndexBufferPos The inner indices position in bytes.
   * @param innerIndexBufferSize The inner indices buffer size in longs.
   * @param data The data.
   * @param bufferPos The data position in bytes.
   * @param denseShape The dense shape of the tensor.
   * @param valuesShape The shape of the values (should be a vector).
   * @param onnxType The type of the values.
   * @return A pointer to an ORT sparse tensor value.
   * @throws OrtException If the tensor could not be created.
   */
  private static native long createCSRCSparseTensorFromBuffer(
      long apiHandle,
      long allocatorHandle,
      Buffer indexData,
      int indexBufferPos,
      long indexBufferSize,
      Buffer innerIndexData,
      int innerIndexBufferPos,
      long innerIndexBufferSize,
      Buffer data,
      int bufferPos,
      long[] denseShape,
      long[] valuesShape,
      int onnxType)
      throws OrtException;

  /**
   * Creates a sparse COO or block sparse tensor.
   *
   * @param apiHandle The ORT API pointer.
   * @param allocatorHandle The allocator pointer.
   * @param indexData The indices.
   * @param indexBufferPos The indices position in bytes.
   * @param indexBufferSize The indices buffer size in longs.
   * @param data The data.
   * @param bufferPos The data position in bytes.
   * @param denseShape The dense shape of the tensor.
   * @param indicesShape The shape of the indices (a vector or matrix for COO, and a matrix for
   *     block sparse).
   * @param valuesShape The shape of the values (a vector for COO, and a block shape for block
   *     sparse).
   * @param onnxType The type of the values.
   * @param sparsityType The sparsity type.
   * @return A pointer to an ORT sparse tensor value.
   * @throws OrtException If the tensor could not be created.
   */
  private static native long createSparseTensorFromBuffer(
      long apiHandle,
      long allocatorHandle,
      Buffer indexData,
      int indexBufferPos,
      long indexBufferSize,
      Buffer data,
      int bufferPos,
      long[] denseShape,
      long[] indicesShape,
      long[] valuesShape,
      int onnxType,
      int sparsityType)
      throws OrtException;

  /**
   * The type of the sparse tensor.
   *
   * <p>Should be synchronized with OrtSparseFormat in the C API.
   */
  public enum SparseTensorType {
    /** Undefined sparse tensor. */
    UNDEFINED(0),
    /** COO sparse tensor. */
    COO(1),
    /** CSR or CSC sparse tensor. */
    CSRC(2),
    /** Block sparse tensor. */
    BLOCK_SPARSE(4);

    /** The int value mirroring OrtSparseFormat. */
    public final int value;

    private static final SparseTensorType[] values = new SparseTensorType[5];

    static {
      values[0] = UNDEFINED;
      values[1] = COO;
      values[2] = CSRC;
      values[3] = UNDEFINED;
      values[4] = BLOCK_SPARSE;
    }

    SparseTensorType(int value) {
      this.value = value;
    }

    /**
     * Maps from an int in native land into a SparseTensorType instance.
     *
     * @param value The value to lookup.
     * @return The enum instance.
     */
    public static SparseTensorType mapFromInt(int value) {
      if ((value > 0) && (value < values.length)) {
        return values[value];
      } else {
        return UNDEFINED;
      }
    }
  }

  /**
   * Top level interfaces for the Java side representation of a sparse tensor.
   *
   * <p>Will be sealed to {@link COOTensor}, {@link CSRCTensor} and {@link BlockSparseTensor} when
   * possible.
   *
   * <p>Does not support Strings as a data type.
   *
   * @param <T> The indices buffer type.
   */
  public static interface SparseTensor<T extends Buffer> {
    /**
     * Gets the dense shape of the sparse tensor.
     *
     * @return The sparse tensor shape.
     */
    public long[] getShape();

    /**
     * Gets the shape of the values of the sparse tensor.
     *
     * @return The sparse tensor value shape.
     */
    public long[] getValuesShape();

    /**
     * Gets the shape of the indices of the sparse tensor.
     *
     * @return The sparse tensor indices shape.
     */
    public long[] getIndicesShape();

    /**
     * The data type of the sparse tensor.
     *
     * @return The sparse tensor data type.
     */
    public OnnxJavaType getType();

    /**
     * The sparsity type of the sparse tensor.
     *
     * @return The sparse tensor sparsity type.
     */
    public SparseTensorType getSparsityType();

    /**
     * The number of non-zero elements.
     *
     * @return The number of non-zero elements.
     */
    public long getNumNonZeroElements();

    /**
     * The indices type of the sparse tensor.
     *
     * <p>Only {@link OnnxJavaType#INT32} and {@link OnnxJavaType#INT64} are supported.
     *
     * @return The sparse tensor indices type.
     */
    public OnnxJavaType getIndicesType();

    /**
     * Get the indices buffer.
     *
     * @return The indices buffer.
     */
    public T getIndices();

    /**
     * Get the data buffer.
     *
     * @return The data buffer.
     */
    public Buffer getData();
  }

  /** Abstract base class for Java sparse tensors */
  private abstract static class BaseSparseTensor<T extends Buffer> implements SparseTensor<T> {
    private final long[] shape;
    private final OnnxJavaType type;
    private final long numNonZero;

    final T indices;
    final Buffer data;

    BaseSparseTensor(T indices, Buffer data, long[] shape, OnnxJavaType type, long numNonZero) {
      this.indices = indices;
      this.data = data;
      this.shape = shape;
      this.type = type;
      this.numNonZero = numNonZero;
      if (data.remaining() != numNonZero) {
        throw new IllegalArgumentException(
            "Expected numNonZero and data.remaining to be equal, found "
                + numNonZero
                + " and "
                + data.remaining()
                + " respectively");
      }
      if (type == OnnxJavaType.STRING) {
        throw new IllegalArgumentException("String SparseTensors are not supported.");
      }
    }

    @Override
    public long[] getShape() {
      return shape;
    }

    @Override
    public OnnxJavaType getType() {
      return type;
    }

    @Override
    public long getNumNonZeroElements() {
      return numNonZero;
    }

    @Override
    public T getIndices() {
      return indices;
    }

    @Override
    public Buffer getData() {
      return data;
    }
  }

  /** The Java side representation of a COO sparse tensor. */
  public static final class COOTensor extends BaseSparseTensor<LongBuffer> {
    private final long[] indicesShape;

    /**
     * Creates a COO sparse tensor suitable for constructing an ORT Sparse Tensor.
     *
     * @param indices The indices. Should be a 1d vector, or a 2d vector.
     * @param indicesShape The shape of the indices.
     * @param data The data.
     * @param shape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public COOTensor(
        LongBuffer indices,
        long[] indicesShape,
        Buffer data,
        long[] shape,
        OnnxJavaType type,
        long numNonZero) {
      super(indices, data, shape, type, numNonZero);
      if ((indicesShape.length > 2)
          || (indicesShape.length == 0)
          || (indicesShape[0] != numNonZero)) {
        throw new IllegalArgumentException(
            "Invalid indices shape, expected [numNonZero, dimension] or [numNonZero] found "
                + Arrays.toString(indicesShape));
      }
      long elementCount = OrtUtil.elementCount(indicesShape);
      if (elementCount != indices.remaining()) {
        throw new IllegalArgumentException(
            "Unexpected number of indices found in buffer, expected "
                + elementCount
                + " found "
                + indices.remaining());
      }
      this.indicesShape = indicesShape;
      if (data.remaining() != numNonZero) {
        throw new IllegalArgumentException(
            "Expected data.remaining() - "
                + data.remaining()
                + " to equal numNonZero - "
                + numNonZero);
      }
    }

    @Override
    public long[] getValuesShape() {
      return new long[] {getNumNonZeroElements()};
    }

    @Override
    public long[] getIndicesShape() {
      return indicesShape;
    }

    @Override
    public OnnxJavaType getIndicesType() {
      return OnnxJavaType.INT64;
    }

    @Override
    public SparseTensorType getSparsityType() {
      return SparseTensorType.COO;
    }
  }

  /** The Java side representation of a CSRC sparse tensor. */
  public static final class CSRCTensor extends BaseSparseTensor<LongBuffer> {
    private final LongBuffer innerIndices;

    /**
     * Creates a CSRC sparse tensor suitable for constructing an ORT Sparse Tensor.
     *
     * @param outerIndices The outer indices.
     * @param innerIndices The inner indices.
     * @param data The data.
     * @param shape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public CSRCTensor(
        LongBuffer outerIndices,
        LongBuffer innerIndices,
        Buffer data,
        long[] shape,
        OnnxJavaType type,
        long numNonZero) {
      super(outerIndices, data, shape, type, numNonZero);
      this.innerIndices = innerIndices;
      long expectedRows = shape[0] + 1;
      if (outerIndices.remaining() != expectedRows) {
        throw new IllegalArgumentException(
            "Outer indices should be equal to the number of rows + 1 in the dense shape, found "
                + outerIndices.remaining()
                + ", expected "
                + expectedRows);
      }
      if (innerIndices.remaining() != numNonZero) {
        throw new IllegalArgumentException(
            "Inner indices should be equal to the number of non-zero elements, found "
                + innerIndices.remaining()
                + ", expected "
                + numNonZero);
      }
    }

    @Override
    public long[] getValuesShape() {
      return new long[] {getNumNonZeroElements()};
    }

    @Override
    public long[] getIndicesShape() {
      return new long[] {indices.remaining()};
    }

    /**
     * Gets the shape of the inner indices.
     *
     * @return The inner indices shape.
     */
    public long[] getInnerIndicesShape() {
      return new long[] {innerIndices.remaining()};
    }

    /**
     * Gets the inner indices buffer.
     *
     * @return The inner indices buffer.
     */
    public LongBuffer getInnerIndices() {
      return innerIndices;
    }

    @Override
    public OnnxJavaType getIndicesType() {
      return OnnxJavaType.INT64;
    }

    @Override
    public SparseTensorType getSparsityType() {
      return SparseTensorType.CSRC;
    }
  }

  /** The Java side representation of a block sparse tensor. */
  public static final class BlockSparseTensor extends BaseSparseTensor<IntBuffer> {
    private final long[] indicesShape;
    private final long[] dataShape;

    /**
     * Construct a block sparse tensor.
     *
     * @param indices The indices.
     * @param indicesShape The shape of the indices.
     * @param data The data.
     * @param dataShape The shape of the data.
     * @param denseShape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public BlockSparseTensor(
        IntBuffer indices,
        long[] indicesShape,
        Buffer data,
        long[] dataShape,
        long[] denseShape,
        OnnxJavaType type,
        long numNonZero) {
      super(indices, data, denseShape, type, numNonZero);
      if (OrtUtil.elementCount(dataShape) != numNonZero) {
        throw new IllegalArgumentException(
            "Expected "
                + numNonZero
                + " entries in the data shape, found "
                + Arrays.toString(dataShape));
      }
      if (numNonZero != data.remaining()) {
        throw new IllegalArgumentException(
            "Expected " + numNonZero + " elements in the data buffer, found " + data.remaining());
      }
      if (OrtUtil.elementCount(indicesShape) != indices.remaining()) {
        throw new IllegalArgumentException(
            "Expected "
                + OrtUtil.elementCount(indicesShape)
                + " elements in the indices buffer, found "
                + indices.remaining());
      }
      if (dataShape.length < 3) {
        throw new IllegalArgumentException(
            "Expected [numBlocks, blockSize, blockSize] or larger, but data shape was "
                + Arrays.toString(dataShape));
      }
      if (indicesShape.length < 2) {
        throw new IllegalArgumentException(
            "Expected [numBlocks, co-ordinates] or larger, but indices shape was "
                + Arrays.toString(indicesShape));
      }
      this.indicesShape = Arrays.copyOf(indicesShape, indicesShape.length);
      this.dataShape = Arrays.copyOf(dataShape, dataShape.length);
    }

    @Override
    public long[] getValuesShape() {
      return dataShape;
    }

    @Override
    public long[] getIndicesShape() {
      return indicesShape;
    }

    @Override
    public OnnxJavaType getIndicesType() {
      return OnnxJavaType.INT32;
    }

    @Override
    public SparseTensorType getSparsityType() {
      return SparseTensorType.BLOCK_SPARSE;
    }
  }
}
