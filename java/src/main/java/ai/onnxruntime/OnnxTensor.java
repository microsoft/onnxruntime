/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A Java object wrapping an OnnxTensor. Tensors are the main input to the library, and can also be
 * returned as outputs.
 */
public class OnnxTensor extends OnnxTensorLike {
  private static final Logger logger = Logger.getLogger(OnnxTensor.class.getName());

  private static final MethodHandle fp16Tofp32;
  private static final MethodHandle fp32ToFp16;

  static {
    MethodHandle tmp16 = null;
    MethodHandle tmp32 = null;
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      // Attempt to lookup the Java 20 fp16 conversion methods.
      tmp16 =
          lookup.findStatic(
              Float.class, "float16ToFloat", MethodType.methodType(float.class, short.class));
      tmp32 =
          lookup.findStatic(
              Float.class, "floatToFloat16", MethodType.methodType(short.class, float.class));
    } catch (IllegalAccessException | NoSuchMethodException e) {
      // Must be on Java 19 or earlier, create handles for our methods.
      try {
        tmp16 =
            lookup.findStatic(
                OnnxTensor.class, "fp16ToFloat", MethodType.methodType(float.class, short.class));
        tmp32 =
            lookup.findStatic(
                OnnxTensor.class, "floatToFp16", MethodType.methodType(short.class, float.class));
      } catch (IllegalAccessException | NoSuchMethodException ex) {
        // Should not happen
        logger.log(Level.SEVERE, "Failed to find fp16 conversion methods on OnnxTensor", e);
      }
    }
    fp16Tofp32 = tmp16;
    fp32ToFp16 = tmp32;
  }

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
          return fp16ToFloat(getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value));
        case BFLOAT16:
          return bf16ToFloat(getShort(OnnxRuntime.ortApiHandle, nativeHandle, info.onnxType.value));
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
      ShortBuffer buffer = getBuffer().asShortBuffer();
      return convertFp16BufferToFloatBuffer(buffer);
    } else if (info.type == OnnxJavaType.BFLOAT16) {
      // if it's bf16 we need to copy it out by hand.
      ShortBuffer buffer = getBuffer().asShortBuffer();
      return convertBf16BufferToFloatBuffer(buffer);
    } else {
      return null;
    }
  }

  static FloatBuffer convertFp16BufferToFloatBuffer(ShortBuffer buf) {
    int bufferCap = buf.capacity();
    FloatBuffer output = FloatBuffer.allocate(bufferCap);
    try {
      for (int i = 0; i < bufferCap; i++) {
        output.put(i, (float) fp16Tofp32.invokeExact(buf.get(i)));
      }
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return output;
  }

  static FloatBuffer convertBf16BufferToFloatBuffer(ShortBuffer buf) {
    int bufferCap = buf.capacity();
    FloatBuffer output = FloatBuffer.allocate(bufferCap);
    try {
      for (int i = 0; i < bufferCap; i++) {
        output.put(i, bf16ToFloat(buf.get(i)));
      }
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return output;
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
   * Returns a copy of the underlying OnnxTensor as a ShortBuffer if the underlying type is int16 or
   * uint16, otherwise it returns null.
   *
   * @return A ShortBuffer copy of the OnnxTensor.
   */
  public ShortBuffer getShortBuffer() {
    if (info.type == OnnxJavaType.INT16) {
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
   * Upcasts a fp16 value to a float. Mirrors the conversion in MLAS.
   *
   * @param input A uint16_t representing an IEEE half precision float.
   * @return A float.
   */
  static float fp16ToFloat(short input) {
    // Port of MLAS_Half2Float from onnxruntime/core/mlas/inc/mlas_float16.h
    final int MAGIC = 113 << 23;
    // exponent mask after shift
    final int SHIFTED_EXP = 0x7c00 << 13;

    // exponent/mantissa bits
    int bits = (input & 0x7fff) << 13;
    // just the exponent
    final int exp = SHIFTED_EXP & bits;
    // exponent adjust
    bits += (127 - 15) << 23;

    // handle exponent special cases
    if (exp == SHIFTED_EXP) {
      // Inf/NaN?
      // extra exp adjust
      bits += (128 - 16) << 23;
    } else if (exp == 0) {
      // Zero/Denormal?
      // extra exp adjust
      bits += (1 << 23);
      // renormalize
      float tmp = Float.intBitsToFloat(bits) - Float.intBitsToFloat(MAGIC);
      bits = Float.floatToIntBits(tmp);
    }

    // sign bit
    bits |= (input & 0x8000) << 16;

    return Float.intBitsToFloat(bits);
  }

  /**
   * Rounds a float value to fp16. Mirrors the conversion in MLAS.
   *
   * @param input A float value.
   * @return The value rounded to an IEEE half precision value.
   */
  static short floatToFp16(float input) {
    // Port of MLAS_Float2Half from onnxruntime/core/mlas/inc/mlas_float16.h
    int bits = Float.floatToIntBits(input);
    final int F32_INFINITY = Float.floatToIntBits(Float.POSITIVE_INFINITY);
    final int F16_MAX = (127 + 16) << 23;
    final int DENORM_MAGIC = ((127 - 15) + (23 - 10) + 1) << 23;
    final int SIGN_MASK = 0x80000000;
    final int ROUNDING_CONST = ((15 - 127) << 23) + 0xfff;

    int sign = bits & SIGN_MASK;
    // mask out sign bit
    bits ^= sign;

    short output;
    if (bits >= F16_MAX) {
      // Inf or NaN (all exponent bits set)
      output = (bits > F32_INFINITY) ? (short) 0x7e00 : (short) 0x7c00;
    } else {
      if (bits < (113 << 23)) {
        // Subnormal or zero
        // use a magic value to align our 10 mantissa bits at the bottom of
        // the float. as long as FP addition is round-to-nearest-even this
        // just works.
        float tmp = Float.intBitsToFloat(bits) + Float.intBitsToFloat(DENORM_MAGIC);

        // and one integer subtract of the bias later, we have our final float!
        output = (short) (Float.floatToIntBits(tmp) - DENORM_MAGIC);
      } else {
        int mant_odd = (bits >> 13) & 1; // resulting mantissa is odd

        // update exponent, rounding bias part 1
        bits += ROUNDING_CONST;
        // rounding bias part 2
        bits += mant_odd;
        // take the bits!
        output = (short) (bits >> 13);
      }
    }

    // Add the sign back in
    output = (short) (output | ((short) (sign >> 16)));

    return output;
  }

  /**
   * Converts a bf16 value stored in a short into a float value.
   *
   * @param input A uint16_t representing a bfloat16 value.
   * @return A float.
   */
  static float bf16ToFloat(short input) {
    int bits = input << 16;
    return Float.intBitsToFloat(bits);
  }

  /**
   * Converts a float into bf16 by truncation. May not produce correct values for subnormal floats.
   *
   * @param input The float input.
   * @return A bfloat16 value which is closest to the float.
   */
  static short floatToBf16(float input) {
    return (short) (Float.floatToIntBits(input) >> 16);
  }

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
