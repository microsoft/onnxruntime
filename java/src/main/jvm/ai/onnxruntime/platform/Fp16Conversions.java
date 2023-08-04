/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.platform;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Conversions between fp16, bfloat16 and fp32. */
public final class Fp16Conversions {
  private static final Logger logger = Logger.getLogger(Fp16Conversions.class.getName());
  private static final MethodHandle fp16ToFp32;
  private static final MethodHandle fp32ToFp16;

  static {
    MethodHandle tmp16 = null;
    MethodHandle tmp32 = null;
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      // Attempt to lookup the Java 20 fp16 conversion methods which can use SIMD intrinsics.
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
                Fp16Conversions.class,
                "mlasFp16ToFloat",
                MethodType.methodType(float.class, short.class));
        tmp32 =
            lookup.findStatic(
                Fp16Conversions.class,
                "mlasFloatToFp16",
                MethodType.methodType(short.class, float.class));
      } catch (IllegalAccessException | NoSuchMethodException ex) {
        // Should not happen
        logger.log(Level.SEVERE, "Failed to find fp16 conversion methods on OnnxTensor", e);
      }
    }
    fp16ToFp32 = tmp16;
    fp32ToFp16 = tmp32;
  }

  /**
   * Rounds a buffer of floats into a buffer containing fp16 values (stored as shorts in Java).
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of floats.
   * @return A buffer of fp16 values stored as shorts.
   */
  public static ShortBuffer convertFloatBufferToFp16Buffer(FloatBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    ShortBuffer output =
        ByteBuffer.allocateDirect(remaining * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, floatToFp16(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Casts a buffer of fp16 values stored as shorts into a buffer of floats.
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of fp16 values stored as shorts.
   * @return A buffer of float values.
   */
  public static FloatBuffer convertFp16BufferToFloatBuffer(ShortBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    FloatBuffer output =
        ByteBuffer.allocateDirect(remaining * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, fp16ToFloat(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Rounds a buffer of floats into a buffer containing bf16 values (stored as shorts in Java).
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of floats.
   * @return A buffer of bf16 values stored as shorts.
   */
  public static ShortBuffer convertFloatBufferToBf16Buffer(FloatBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    ShortBuffer output =
        ByteBuffer.allocateDirect(remaining * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, floatToBf16(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Casts a buffer of bf16 values stored as shorts into a buffer of floats.
   *
   * <p>Respects the position and limit of the input buffer.
   *
   * @param buf The buffer of bf16 values stored as shorts.
   * @return A buffer of float values.
   */
  public static FloatBuffer convertBf16BufferToFloatBuffer(ShortBuffer buf) {
    int pos = buf.position();
    int remaining = buf.remaining();
    FloatBuffer output =
        ByteBuffer.allocateDirect(remaining * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    for (int i = 0; i < remaining; i++) {
      output.put(i, bf16ToFloat(buf.get(i + pos)));
    }
    return output;
  }

  /**
   * Converts a fp16 value stored in a short into a float value.
   *
   * <p>Note on Java 20 or newer this uses {@code Float.float16ToFloat} which may use CPU specific
   * instructions for the conversion, otherwise it uses the conversion operation from ORT's native
   * implementation.
   *
   * @param input The fp16 value.
   * @return The float value.
   */
  public static float fp16ToFloat(short input) {
    try {
      float ret = (float) fp16ToFp32.invokeExact(input);
      return ret;
    } catch (Throwable e) {
      throw new AssertionError("Should not reach here", e);
    }
  }

  /**
   * Converts a float value into a fp16 value stored in a short.
   *
   * <p>Note on Java 20 or newer this uses {@code Float.floatToFloat16} which may use CPU specific
   * instructions for the conversion, otherwise it uses the conversion operation from ORT's native
   * implementation.
   *
   * @param input The float value.
   * @return The fp16 value.
   */
  public static short floatToFp16(float input) {
    try {
      short ret = (short) fp32ToFp16.invokeExact(input);
      return ret;
    } catch (Throwable e) {
      throw new AssertionError("Should not reach here", e);
    }
  }

  /**
   * Upcasts a fp16 value to a float. Mirrors the conversion in MLAS.
   *
   * @param input A uint16_t representing an IEEE half precision float.
   * @return A float.
   */
  public static float mlasFp16ToFloat(short input) {
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
  public static short mlasFloatToFp16(float input) {
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
  public static float bf16ToFloat(short input) {
    int bits = input << 16;
    return Float.intBitsToFloat(bits);
  }

  /**
   * Converts a float into bf16. May not produce correct values for subnormal floats.
   *
   * <p>Rounds to nearest even.
   *
   * @param input The float input.
   * @return A bfloat16 value which is closest to the float.
   */
  public static short floatToBf16(float input) {
    int bits = Float.floatToIntBits(input);
    int lsb = (bits >> 16) & 1;
    int roundingBias = 0x7fff + lsb;
    bits += roundingBias;
    return (short) (bits >> 16);
  }
}
