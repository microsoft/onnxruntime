/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

/**
 * A Java object wrapping an OnnxTensor. Tensors are the main input to the library, and can also be
 * returned as outputs.
 */
public class OnnxTensor extends OnnxTensorLike {

  /**
   * This reference is held for OnnxTensors backed by a Java nio buffer to ensure the buffer does
   * not go out of scope while the OnnxTensor exists.
   */
  private final Buffer buffer;

  OnnxTensor(long nativeHandle, long allocatorHandle, TensorInfo info) {
    this(nativeHandle, allocatorHandle, info, null);
  }

  OnnxTensor(long nativeHandle, long allocatorHandle, TensorInfo info, Buffer buffer) {
    super(nativeHandle, allocatorHandle, info);
    this.buffer = buffer;
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_TENSOR;
  }

  /**
   * Either returns a boxed primitive if the Tensor is a scalar, or a multidimensional array of
   * primitives if it has multiple dimensions.
   *
   * <p>Java multidimensional arrays are quite slow for more than 2 dimensions, in that case it is
   * recommended you use the java.nio.Buffer extractors below (e.g. {@link #getFloatBuffer}).
   *
   * @return A Java value.
   * @throws OrtException If the value could not be extracted as the Tensor is invalid, or if the
   *     native code encountered an error.
   */
  @Override
  public Object getValue() throws OrtException {
    if (info.isScalar()) {
      switch (info.type) {
        case FLOAT:
          return getFloat(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case DOUBLE:
          return getDouble(OnnxRuntime.ortApiHandle, nativeHandle);
        case UINT8:
        case INT8:
          return getByte(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case INT16:
          return getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case INT32:
          return getInt(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case INT64:
          return getLong(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value);
        case BOOL:
          return getBool(OnnxRuntime.ortApiHandle, nativeHandle);
        case STRING:
          return getString(OnnxRuntime.ortApiHandle, nativeHandle);
        case FLOAT16:
          return OrtUtil.fp16ToFloat(
              getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value));
        case BFLOAT16:
          return OrtUtil.bf16ToFloat(
              getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value));
        case UNKNOWN:
        default:
          throw new OrtException("Extracting the value of an invalid Tensor.");
      }
    } else {
      Object carrier = info.makeCarrier();
      if (info.getNumElements() > 0) {
        // If the tensor has values copy them out
        getArray(OnnxRuntime.ortApiHandle, nativeHandle, carrier);
      }
      if ((info.type == OnnxJavaType.STRING) && (info.shape.length != 1)) {
        // We read the strings out from native code in a flat array and then reshape
        // to the desired output shape.
        return OrtUtil.reshape((String[]) carrier, info.shape);
      } else {
        return carrier;
      }
    }
  }

  @Override
  public String toString() {
    return "OnnxTensor(info=" + info.toString() + ")";
  }

  /**
   * Closes the tensor, releasing it's underlying memory (if it's not backed by an NIO buffer). If
   * it is backed by a buffer then the memory is released when the buffer is GC'd.
   */
  @Override
  public void close() {
    close(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a ByteBuffer.
   *
   * <p>This method returns null if the OnnxTensor contains Strings as they are stored externally to
   * the OnnxTensor.
   *
   * @return A ByteBuffer copy of the OnnxTensor.
   */
  public ByteBuffer getByteBuffer() {
    if (info.type != OnnxJavaType.STRING) {
      ByteBuffer buffer = getBuffer(OnnxRuntime.ortApiHandle, nativeHandle);
      ByteBuffer output = ByteBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a FloatBuffer if it can be losslessly converted
   * into a float (i.e. it's a float, fp16 or bf16), otherwise it returns null.
   *
   * @return A FloatBuffer copy of the OnnxTensor.
   */
  public FloatBuffer getFloatBuffer() {
    if (info.type == OnnxJavaType.FLOAT) {
      // if it's fp32 use the efficient copy.
      FloatBuffer buffer = getBuffer().asFloatBuffer();
      FloatBuffer output = FloatBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else if (info.type == OnnxJavaType.FLOAT16) {
      // if it's fp16 we need to copy it out by hand.
      ByteBuffer buf = getBuffer();
      ShortBuffer buffer = buf.asShortBuffer();
      return OrtUtil.convertFp16BufferToFloatBuffer(buffer);
    } else if (info.type == OnnxJavaType.BFLOAT16) {
      // if it's bf16 we need to copy it out by hand.
      ByteBuffer buf = getBuffer();
      ShortBuffer buffer = buf.asShortBuffer();
      return OrtUtil.convertBf16BufferToFloatBuffer(buffer);
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a DoubleBuffer if the underlying type is a
   * double, otherwise it returns null.
   *
   * @return A DoubleBuffer copy of the OnnxTensor.
   */
  public DoubleBuffer getDoubleBuffer() {
    if (info.type == OnnxJavaType.DOUBLE) {
      DoubleBuffer buffer = getBuffer().asDoubleBuffer();
      DoubleBuffer output = DoubleBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a ShortBuffer if the underlying type is int16,
   * uint16, fp16 or bf16, otherwise it returns null.
   *
   * @return A ShortBuffer copy of the OnnxTensor.
   */
  public ShortBuffer getShortBuffer() {
    if ((info.type == OnnxJavaType.INT16)
        || (info.type == OnnxJavaType.FLOAT16)
        || (info.type == OnnxJavaType.BFLOAT16)) {
      ShortBuffer buffer = getBuffer().asShortBuffer();
      ShortBuffer output = ShortBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as an IntBuffer if the underlying type is int32 or
   * uint32, otherwise it returns null.
   *
   * @return An IntBuffer copy of the OnnxTensor.
   */
  public IntBuffer getIntBuffer() {
    if (info.type == OnnxJavaType.INT32) {
      IntBuffer buffer = getBuffer().asIntBuffer();
      IntBuffer output = IntBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a LongBuffer if the underlying type is int64 or
   * uint64, otherwise it returns null.
   *
   * @return A LongBuffer copy of the OnnxTensor.
   */
  public LongBuffer getLongBuffer() {
    if (info.type == OnnxJavaType.INT64) {
      LongBuffer buffer = getBuffer().asLongBuffer();
      LongBuffer output = LongBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Wraps the OrtTensor pointer in a direct byte buffer of the native platform endian-ness. Unless
   * you really know what you're doing, you want this one rather than the native call {@link
   * OnnxTensor#getBuffer(long,long)}.
   *
   * @return A ByteBuffer wrapping the data.
   */
  private ByteBuffer getBuffer() {
    return getBuffer(OnnxRuntime.ortApiHandle, nativeHandle).order(ByteOrder.nativeOrder());
  }

  /**
   * Wraps the OrtTensor pointer in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtTensor pointer.
   * @return A ByteBuffer wrapping the data.
   */
  private native ByteBuffer getBuffer(long apiHandle, long nativeHandle);

  private native float getFloat(long apiHandle, long nativeHandle, int onnxType)
      throws OrtException;

  private native double getDouble(long apiHandle, long nativeHandle) throws OrtException;

  private native byte getByte(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native short getShort(long apiHandle, long nativeHandle, int onnxType)
      throws OrtException;

  private native int getInt(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native long getLong(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native String getString(long apiHandle, long nativeHandle) throws OrtException;

  private native boolean getBool(long apiHandle, long nativeHandle) throws OrtException;

  private native void getArray(long apiHandle, long nativeHandle, Object carrier)
      throws OrtException;

  private native void close(long apiHandle, long nativeHandle);

  /**
   * Create a Tensor from a Java primitive, primitive multidimensional array or String
   * multidimensional array. The shape is inferred from the object using reflection. The default
   * allocator is used.
   *
   * @param env The current OrtEnvironment.
   * @param data The data to store in a tensor.
   * @return An OnnxTensor storing the data.
   * @throws OrtException If the onnx runtime threw an error.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, Object data) throws OrtException {
    return createTensor(env, env.defaultAllocator, data);
  }

  /**
   * Create a Tensor from a Java primitive, String, primitive multidimensional array or String
   * multidimensional array. The shape is inferred from the object using reflection.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The data to store in a tensor.
   * @return An OnnxTensor storing the data.
   * @throws OrtException If the onnx runtime threw an error.
   */
  static OnnxTensor createTensor(OrtEnvironment env, OrtAllocator allocator, Object data)
      throws OrtException {
    if (!allocator.isClosed()) {
      TensorInfo info = TensorInfo.constructFromJavaArray(data);
      if (info.type == OnnxJavaType.STRING) {
        if (info.shape.length == 0) {
          return new OnnxTensor(
              createString(OnnxRuntime.ortApiHandle, allocator.handle, (String) data),
              allocator.handle,
              info);
        } else {
          return new OnnxTensor(
              createStringTensor(
                  OnnxRuntime.ortApiHandle,
                  allocator.handle,
                  OrtUtil.flattenString(data),
                  info.shape),
              allocator.handle,
              info);
        }
      } else {
        if (info.shape.length == 0) {
          data = OrtUtil.convertBoxedPrimitiveToArray(info.type, data);
          if (data == null) {
            throw new OrtException(
                "Failed to convert a boxed primitive to an array, this is an error with the ORT Java API, please report this message & stack trace. JavaType = "
                    + info.type
                    + ", object = "
                    + data);
          }
        }
        return new OnnxTensor(
            createTensor(
                OnnxRuntime.ortApiHandle, allocator.handle, data, info.shape, info.onnxType.value),
            allocator.handle,
            info);
      }
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor with a closed OrtAllocator.");
    }
  }

  /**
   * Create a tensor from a flattened string array.
   *
   * <p>Requires the array to be flattened in row-major order. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data
   * @param shape the shape of the tensor
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, String[] data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create a tensor from a flattened string array.
   *
   * <p>Requires the array to be flattened in row-major order.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data
   * @param shape the shape of the tensor
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, String[] data, long[] shape) throws OrtException {
    if (!allocator.isClosed()) {
      TensorInfo info =
          new TensorInfo(
              shape,
              OnnxJavaType.STRING,
              TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
      return new OnnxTensor(
          createStringTensor(OnnxRuntime.ortApiHandle, allocator.handle, data, shape),
          allocator.handle,
          info);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, FloatBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, FloatBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.FLOAT;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, DoubleBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, DoubleBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.DOUBLE;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator. Tells the runtime it's {@link OnnxJavaType#INT8}.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, ByteBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Tells the runtime it's {@link OnnxJavaType#INT8}.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ByteBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, allocator, data, shape, OnnxJavaType.INT8);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator. Tells the runtime it's the specified type.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type to use for the byte buffer elements.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(
      OrtEnvironment env, ByteBuffer data, long[] shape, OnnxJavaType type) throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape, type);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Tells the runtime it's the specified type.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type to use for the byte buffer elements.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ByteBuffer data, long[] shape, OnnxJavaType type)
      throws OrtException {
    if (!allocator.isClosed()) {
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, ShortBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ShortBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.INT16;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, IntBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, IntBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.INT32;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, LongBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, env.defaultAllocator, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the supplied allocator.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, LongBuffer data, long[] shape)
      throws OrtException {
    if (!allocator.isClosed()) {
      OnnxJavaType type = OnnxJavaType.INT64;
      return createTensor(type, allocator, data, shape);
    } else {
      throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
    }
  }

  /**
   * Creates a tensor by wrapping the data in a direct byte buffer and passing it to JNI.
   *
   * <p>Throws IllegalStateException if the buffer is too large to create a direct byte buffer copy,
   * which is more than approximately (Integer.MAX_VALUE - 5) / type.size elements.
   *
   * @param type The buffer type.
   * @param allocator The OrtAllocator.
   * @param data The data.
   * @param shape The tensor shape.
   * @return An OnnxTensor instance.
   * @throws OrtException If the create call failed.
   */
  private static OnnxTensor createTensor(
      OnnxJavaType type, OrtAllocator allocator, Buffer data, long[] shape) throws OrtException {
    OrtUtil.BufferTuple tuple = OrtUtil.prepareBuffer(data, type);
    TensorInfo info = TensorInfo.constructFromBuffer(tuple.data, shape, type);
    return new OnnxTensor(
        createTensorFromBuffer(
            OnnxRuntime.ortApiHandle,
            allocator.handle,
            tuple.data,
            tuple.pos,
            tuple.byteSize,
            shape,
            info.onnxType.value),
        allocator.handle,
        info,
        tuple.data);
  }

  private static native long createTensor(
      long apiHandle, long allocatorHandle, Object data, long[] shape, int onnxType)
      throws OrtException;

  private static native long createTensorFromBuffer(
      long apiHandle,
      long allocatorHandle,
      Buffer data,
      int bufferPos,
      long bufferSize,
      long[] shape,
      int onnxType)
      throws OrtException;

  private static native long createString(long apiHandle, long allocatorHandle, String data)
      throws OrtException;

  private static native long createStringTensor(
      long apiHandle, long allocatorHandle, Object[] data, long[] shape) throws OrtException;
}
