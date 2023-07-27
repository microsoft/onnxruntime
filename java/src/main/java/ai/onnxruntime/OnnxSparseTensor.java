/*
 * Copyright (c) 2022, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.platform.Fp16Conversions;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
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
  private final Buffer values;

  /**
   * Construct a sparse tensor from JNI.
   *
   * @param nativeHandle The tensor native handle.
   * @param allocatorHandle The allocator handle.
   * @param sparseType The sparsity type.
   * @param info The tensor info.
   */
  OnnxSparseTensor(long nativeHandle, long allocatorHandle, int sparseType, TensorInfo info) {
    this(
        nativeHandle,
        allocatorHandle,
        SparseTensorType.mapFromInt(sparseType),
        info,
        null,
        null,
        null);
  }

  /**
   * Construct a COO or block sparse tensor.
   *
   * @param nativeHandle The tensor native handle.
   * @param allocatorHandle The allocator handle.
   * @param sparseType The sparsity type.
   * @param info The tensor info.
   * @param indices The indices buffer.
   * @param values The data buffer.
   */
  OnnxSparseTensor(
      long nativeHandle,
      long allocatorHandle,
      SparseTensorType sparseType,
      TensorInfo info,
      Buffer indices,
      Buffer values) {
    this(nativeHandle, allocatorHandle, sparseType, info, indices, null, values);
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
   * @param values The data buffer.
   */
  OnnxSparseTensor(
      long nativeHandle,
      long allocatorHandle,
      SparseTensorType sparseType,
      TensorInfo info,
      Buffer indices,
      LongBuffer innerIndices,
      Buffer values) {
    super(nativeHandle, allocatorHandle, info);
    this.sparseTensorType = sparseType;
    this.indices = indices;
    this.innerIndices = innerIndices;
    this.values = values;
  }

  /**
   * Creates a Sparse Tensor in ORT from the Java side representation.
   *
   * @param env The OrtEnvironment.
   * @param tensor The Java side representation.
   * @param <T> The buffer type.
   * @return The sparse tensor in ORT.
   * @throws OrtException If the tensor could not be created or was invalid.
   */
  public static <T extends Buffer> OnnxSparseTensor createSparseTensor(
      OrtEnvironment env, SparseTensor<T> tensor) throws OrtException {
    return createSparseTensor(env, env.defaultAllocator, tensor);
  }

  /**
   * Creates a Sparse Tensor in ORT from the Java side representation.
   *
   * @param env The OrtEnvironment.
   * @param allocator The memory allocator.
   * @param tensor The Java side representation.
   * @param <T> The buffer type.
   * @return The sparse tensor in ORT.
   * @throws OrtException If the tensor could not be created or was invalid.
   */
  static <T extends Buffer> OnnxSparseTensor createSparseTensor(
      OrtEnvironment env, OrtAllocator allocator, SparseTensor<T> tensor) throws OrtException {
    if (!allocator.isClosed()) {
      TensorInfo info = TensorInfo.constructFromSparseTensor(tensor);
      OnnxJavaType indicesType = tensor.getIndicesType();
      OrtUtil.BufferTuple indicesTuple = OrtUtil.prepareBuffer(tensor.getIndices(), indicesType);
      OrtUtil.BufferTuple valuesTuple = OrtUtil.prepareBuffer(tensor.getValues(), info.type);
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
                  valuesTuple.data,
                  valuesTuple.pos,
                  info.shape,
                  tensor.getIndicesShape(),
                  tensor.getValuesShape(),
                  info.onnxType.value,
                  tensor.getSparsityType().value),
              allocator.handle,
              tensor.getSparsityType(),
              info,
              indicesTuple.data,
              valuesTuple.data);
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
                  valuesTuple.data,
                  valuesTuple.pos,
                  info.shape,
                  tensor.getValuesShape(),
                  info.onnxType.value),
              allocator.handle,
              tensor.getSparsityType(),
              info,
              indicesTuple.data,
              (LongBuffer) innerIndicesTuple.data,
              valuesTuple.data);
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
  public SparseTensor<? extends Buffer> getValue() throws OrtException {
    Buffer buffer = getValuesBuffer();
    long[] indicesShape = getIndicesShape(OnnxRuntime.ortApiHandle, nativeHandle);
    switch (sparseTensorType) {
      case COO:
        return new COOTensor(
            (LongBuffer) getIndicesBuffer(),
            indicesShape,
            buffer,
            info.shape,
            info.type,
            buffer.remaining());
      case CSRC:
        return new CSRCTensor(
            (LongBuffer) getIndicesBuffer(),
            getInnerIndicesBuffer(),
            buffer,
            info.shape,
            info.type,
            buffer.remaining());
      case BLOCK_SPARSE:
        long[] valuesShape = getValuesShape(OnnxRuntime.ortApiHandle, nativeHandle);
        return new BlockSparseTensor(
            (IntBuffer) getIndicesBuffer(),
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
  public Buffer getIndicesBuffer() {
    switch (sparseTensorType) {
      case COO:
      case CSRC:
        {
          LongBuffer longBuf =
              getIndicesBuffer(OnnxRuntime.ortApiHandle, nativeHandle)
                  .order(ByteOrder.nativeOrder())
                  .asLongBuffer();
          LongBuffer output = LongBuffer.allocate(longBuf.capacity());
          output.put(longBuf);
          output.rewind();
          return output;
        }
      case BLOCK_SPARSE:
        {
          IntBuffer intBuf =
              getIndicesBuffer(OnnxRuntime.ortApiHandle, nativeHandle)
                  .order(ByteOrder.nativeOrder())
                  .asIntBuffer();
          IntBuffer output = IntBuffer.allocate(intBuf.capacity());
          output.put(intBuf);
          output.rewind();
          return output;
        }
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
  public LongBuffer getInnerIndicesBuffer() {
    if (sparseTensorType == SparseTensorType.CSRC) {
      LongBuffer buf =
          getInnerIndicesBuffer(OnnxRuntime.ortApiHandle, nativeHandle)
              .order(ByteOrder.nativeOrder())
              .asLongBuffer();
      LongBuffer output = LongBuffer.allocate(buf.capacity());
      output.put(buf);
      output.rewind();
      return output;
    } else {
      throw new IllegalStateException(
          "Inner indices are only available for CSRC sparse tensors, this sparse tensor is "
              + sparseTensorType);
    }
  }

  /**
   * Gets a copy of the data buffer.
   *
   * <p>As with {@link OnnxTensor} fp16 values are upcast into fp32 and returned as a {@link
   * FloatBuffer}.
   *
   * @return The data buffer.
   */
  public Buffer getValuesBuffer() {
    ByteBuffer buffer =
        getValuesBuffer(OnnxRuntime.ortApiHandle, nativeHandle).order(ByteOrder.nativeOrder());
    switch (info.type) {
      case FLOAT:
        {
          FloatBuffer floatBuf = buffer.asFloatBuffer();
          FloatBuffer output = FloatBuffer.allocate(floatBuf.capacity());
          output.put(floatBuf);
          output.rewind();
          return output;
        }
      case FLOAT16:
        {
          ShortBuffer shortBuffer = buffer.asShortBuffer();
          return Fp16Conversions.convertFp16BufferToFloatBuffer(shortBuffer);
        }
      case BFLOAT16:
        {
          ShortBuffer shortBuffer = buffer.asShortBuffer();
          return Fp16Conversions.convertBf16BufferToFloatBuffer(shortBuffer);
        }
      case DOUBLE:
        {
          DoubleBuffer doubleBuf = buffer.asDoubleBuffer();
          DoubleBuffer output = DoubleBuffer.allocate(doubleBuf.capacity());
          output.put(doubleBuf);
          output.rewind();
          return output;
        }
      case INT16:
        {
          ShortBuffer shortBuf = buffer.asShortBuffer();
          ShortBuffer output = ShortBuffer.allocate(shortBuf.capacity());
          output.put(shortBuf);
          output.rewind();
          return output;
        }
      case INT32:
        {
          IntBuffer intBuf = buffer.asIntBuffer();
          IntBuffer output = IntBuffer.allocate(intBuf.capacity());
          output.put(intBuf);
          output.rewind();
          return output;
        }
      case INT64:
        {
          LongBuffer longBuf = buffer.asLongBuffer();
          LongBuffer output = LongBuffer.allocate(longBuf.capacity());
          output.put(longBuf);
          output.rewind();
          return output;
        }
      case BOOL:
      case INT8:
      case UINT8:
        {
          ByteBuffer output = ByteBuffer.allocate(buffer.capacity());
          output.put(buffer);
          output.rewind();
          return output;
        }
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
    return getIndicesShape(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Gets the shape of the inner indices in a CSRC sparse tensor.
   *
   * @return The indices shape.
   */
  public long[] getInnerIndicesShape() {
    if (sparseTensorType == SparseTensorType.CSRC) {
      return getInnerIndicesShape(OnnxRuntime.ortApiHandle, nativeHandle);
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
    return getValuesShape(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Gets the shape of the (outer) indices.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return The indices shape.
   */
  private native long[] getIndicesShape(long apiHandle, long nativeHandle);

  /**
   * Gets the shape of the inner indices.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return The inner indices shape.
   */
  private native long[] getInnerIndicesShape(long apiHandle, long nativeHandle);

  /**
   * Gets the shape of the values.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return The values shape.
   */
  private native long[] getValuesShape(long apiHandle, long nativeHandle);

  /**
   * Wraps the indices in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the indices.
   */
  private native ByteBuffer getIndicesBuffer(long apiHandle, long nativeHandle);

  /**
   * Wraps the inner indices in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the inner indices.
   */
  private native ByteBuffer getInnerIndicesBuffer(long apiHandle, long nativeHandle);

  /**
   * Wraps the data in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtSparseTensor pointer.
   * @return A ByteBuffer wrapping the indices.
   */
  private native ByteBuffer getValuesBuffer(long apiHandle, long nativeHandle);

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
   * <p>The buffers must be kept alive for the lifetime of the ORT sparse tensor object.
   *
   * @param apiHandle The ORT API pointer.
   * @param allocatorHandle The allocator pointer.
   * @param indicesData The outer indices.
   * @param indicesBufferPos The outer indices position in bytes.
   * @param indicesBufferSize The outer indices buffer size in longs.
   * @param innerIndicesData The inner indices.
   * @param innerIndicesBufferPos The inner indices position in bytes.
   * @param innerIndicesBufferSize The inner indices buffer size in longs.
   * @param values The data.
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
      Buffer indicesData,
      int indicesBufferPos,
      long indicesBufferSize,
      Buffer innerIndicesData,
      int innerIndicesBufferPos,
      long innerIndicesBufferSize,
      Buffer values,
      int bufferPos,
      long[] denseShape,
      long[] valuesShape,
      int onnxType)
      throws OrtException;

  /**
   * Creates a sparse COO or block sparse tensor.
   *
   * <p>The buffers must be kept alive for the lifetime of the ORT sparse tensor object.
   *
   * @param apiHandle The ORT API pointer.
   * @param allocatorHandle The allocator pointer.
   * @param indicesData The indices.
   * @param indicesBufferPos The indices position in bytes.
   * @param indicesBufferSize The indices buffer size in longs.
   * @param values The data.
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
      Buffer indicesData,
      int indicesBufferPos,
      long indicesBufferSize,
      Buffer values,
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
   * Abstract base class for Java sparse tensors
   *
   * <p>Will be sealed to {@link COOTensor}, {@link CSRCTensor} and {@link BlockSparseTensor} one
   * day.
   *
   * @param <T> The type of the indices buffer.
   */
  public abstract static class SparseTensor<T extends Buffer> {
    private final long[] indicesShape;
    private final long[] valuesShape;
    private final long[] denseShape;
    private final OnnxJavaType type;
    private final long numNonZero;

    /** The buffer holding the indices. */
    final T indices;
    /** The buffer holding the values. */
    final Buffer values;

    SparseTensor(
        T indices,
        long[] indicesShape,
        Buffer values,
        long[] valuesShape,
        long[] denseShape,
        OnnxJavaType type,
        long numNonZero) {
      this.indices = indices;
      this.indicesShape = indicesShape;
      this.values = values;
      this.valuesShape = valuesShape;
      this.denseShape = denseShape;
      this.type = type;
      this.numNonZero = numNonZero;
      if (values.remaining() != numNonZero) {
        throw new IllegalArgumentException(
            "Expected numNonZero and data.remaining to be equal, found "
                + numNonZero
                + " and "
                + values.remaining()
                + " respectively");
      }
      if (type == OnnxJavaType.STRING) {
        throw new IllegalArgumentException("String SparseTensors are not supported.");
      }
    }

    /**
     * Gets the dense shape of the sparse tensor.
     *
     * @return The sparse tensor shape.
     */
    public long[] getDenseShape() {
      return denseShape;
    }

    /**
     * The data type of the sparse tensor.
     *
     * @return The sparse tensor data type.
     */
    public OnnxJavaType getType() {
      return type;
    }

    /**
     * The number of non-zero elements.
     *
     * @return The number of non-zero elements.
     */
    public long getNumNonZeroElements() {
      return numNonZero;
    }

    /**
     * Get the indices buffer.
     *
     * @return The indices buffer.
     */
    public T getIndices() {
      return indices;
    }

    /**
     * Get the value buffer.
     *
     * @return The value buffer.
     */
    public Buffer getValues() {
      return values;
    }

    /**
     * Gets the shape of the values of the sparse tensor.
     *
     * @return The sparse tensor value shape.
     */
    public long[] getValuesShape() {
      return valuesShape;
    }

    /**
     * Gets the shape of the indices of the sparse tensor.
     *
     * @return The sparse tensor indices shape.
     */
    public long[] getIndicesShape() {
      return indicesShape;
    }

    /**
     * The sparsity type of the sparse tensor.
     *
     * @return The sparse tensor sparsity type.
     */
    public abstract SparseTensorType getSparsityType();

    /**
     * The indices type of the sparse tensor.
     *
     * <p>Only {@link OnnxJavaType#INT32} and {@link OnnxJavaType#INT64} are supported.
     *
     * @return The sparse tensor indices type.
     */
    public abstract OnnxJavaType getIndicesType();
  }

  /** The Java side representation of a COO sparse tensor. */
  public static final class COOTensor extends SparseTensor<LongBuffer> {
    /**
     * Creates a COO sparse tensor suitable for constructing an ORT Sparse Tensor.
     *
     * @param indices The indices. Should be a 1d vector, or a 2d vector.
     * @param indicesShape The shape of the indices.
     * @param values The data.
     * @param denseShape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public COOTensor(
        LongBuffer indices,
        long[] indicesShape,
        Buffer values,
        long[] denseShape,
        OnnxJavaType type,
        long numNonZero) {
      super(indices, indicesShape, values, new long[] {numNonZero}, denseShape, type, numNonZero);
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
      if (values.remaining() != numNonZero) {
        throw new IllegalArgumentException(
            "Expected data.remaining() - "
                + values.remaining()
                + " to equal numNonZero - "
                + numNonZero);
      }
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
  public static final class CSRCTensor extends SparseTensor<LongBuffer> {
    private final LongBuffer innerIndices;

    /**
     * Creates a CSRC sparse tensor suitable for constructing an ORT Sparse Tensor.
     *
     * @param outerIndices The outer indices.
     * @param innerIndices The inner indices.
     * @param values The data.
     * @param denseShape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public CSRCTensor(
        LongBuffer outerIndices,
        LongBuffer innerIndices,
        Buffer values,
        long[] denseShape,
        OnnxJavaType type,
        long numNonZero) {
      super(
          outerIndices,
          new long[] {outerIndices.remaining()},
          values,
          new long[] {numNonZero},
          denseShape,
          type,
          numNonZero);
      this.innerIndices = innerIndices;
      long expectedRows = denseShape[0] + 1;
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
  public static final class BlockSparseTensor extends SparseTensor<IntBuffer> {
    /**
     * Construct a block sparse tensor.
     *
     * @param indices The indices.
     * @param indicesShape The shape of the indices.
     * @param values The data.
     * @param valuesShape The shape of the data.
     * @param denseShape The dense shape.
     * @param type The data type.
     * @param numNonZero The number of non-zero elements.
     */
    public BlockSparseTensor(
        IntBuffer indices,
        long[] indicesShape,
        Buffer values,
        long[] valuesShape,
        long[] denseShape,
        OnnxJavaType type,
        long numNonZero) {
      super(indices, indicesShape, values, valuesShape, denseShape, type, numNonZero);
      if (OrtUtil.elementCount(valuesShape) != numNonZero) {
        throw new IllegalArgumentException(
            "Expected "
                + numNonZero
                + " entries in the data shape, found "
                + Arrays.toString(valuesShape));
      }
      if (numNonZero != values.remaining()) {
        throw new IllegalArgumentException(
            "Expected " + numNonZero + " elements in the data buffer, found " + values.remaining());
      }
      if (OrtUtil.elementCount(indicesShape) != indices.remaining()) {
        throw new IllegalArgumentException(
            "Expected "
                + OrtUtil.elementCount(indicesShape)
                + " elements in the indices buffer, found "
                + indices.remaining());
      }
      if (valuesShape.length < 3) {
        throw new IllegalArgumentException(
            "Expected [numBlocks, blockSize, blockSize] or larger, but data shape was "
                + Arrays.toString(valuesShape));
      }
      if (indicesShape.length < 2) {
        throw new IllegalArgumentException(
            "Expected [numBlocks, co-ordinates] or larger, but indices shape was "
                + Arrays.toString(indicesShape));
      }
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
