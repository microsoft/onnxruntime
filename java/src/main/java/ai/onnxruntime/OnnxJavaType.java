/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.TensorInfo.OnnxTensorType;

/** An enum representing ONNX Runtime supported Java primitive types (and String). */
public enum OnnxJavaType {
  /** A 32-bit floating point value. */
  FLOAT(1, float.class, 4),
  /** A 64-bit floating point value. */
  DOUBLE(2, double.class, 8),
  /** An 8-bit signed integer value. */
  INT8(3, byte.class, 1),
  /** A 16-bit signed integer value. */
  INT16(4, short.class, 2),
  /** A 32-bit signed integer value. */
  INT32(5, int.class, 4),
  /** A 64-bit signed integer value. */
  INT64(6, long.class, 8),
  /** A boolean value stored in a single byte. */
  BOOL(7, boolean.class, 1),
  /** A UTF-8 string. */
  STRING(8, String.class, 4),
  /** A 8-bit unsigned integer value. */
  UINT8(9, byte.class, 1),
  /** A IEEE 16-bit floating point value. */
  FLOAT16(10, short.class, 2),
  /** A non-IEEE 16-bit floating point value, with 8 exponent bits and 7 mantissa bits. */
  BFLOAT16(11, short.class, 2),
  /** An unknown type used as an error condition or a sentinel. */
  UNKNOWN(0, Object.class, 0);

  private static final OnnxJavaType[] values;

  static {
    OnnxJavaType[] tmpValues = OnnxJavaType.values();
    values = new OnnxJavaType[tmpValues.length];
    for (OnnxJavaType ot : tmpValues) {
      values[ot.value] = ot;
    }
  }

  /** The native value of the enum. */
  public final int value;
  /** The Java side type used as the carrier. */
  public final Class<?> clazz;
  /** The number of bytes used by a single value of this type. */
  public final int size;

  OnnxJavaType(int value, Class<?> clazz, int size) {
    this.value = value;
    this.clazz = clazz;
    this.size = size;
  }

  /**
   * Maps from an int in native land into an OnnxJavaType instance.
   *
   * @param value The value to lookup.
   * @return The enum instance.
   */
  public static OnnxJavaType mapFromInt(int value) {
    if ((value > 0) && (value < values.length)) {
      return values[value];
    } else {
      return UNKNOWN;
    }
  }

  /**
   * Maps from the {@link OnnxTensorType} enum to the corresponding OnnxJavaType enum, converting
   * types as appropriate.
   *
   * <p>Must match the values from OrtJniUtil.c.
   *
   * @param onnxValue The native value type.
   * @return A OnnxJavaType instance representing the Java type
   */
  public static OnnxJavaType mapFromOnnxTensorType(OnnxTensorType onnxValue) {
    switch (onnxValue) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return OnnxJavaType.UINT8;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return OnnxJavaType.INT8;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return OnnxJavaType.INT16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return OnnxJavaType.INT32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return OnnxJavaType.INT64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return OnnxJavaType.FLOAT16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return OnnxJavaType.BFLOAT16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return OnnxJavaType.FLOAT;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return OnnxJavaType.DOUBLE;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return OnnxJavaType.STRING;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return OnnxJavaType.BOOL;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      default:
        return OnnxJavaType.UNKNOWN;
    }
  }

  /**
   * Maps from a Java class object into the enum type, returning {@link OnnxJavaType#UNKNOWN} for
   * unsupported types.
   *
   * @param clazz The class to use.
   * @return An OnnxJavaType instance.
   */
  public static OnnxJavaType mapFromClass(Class<?> clazz) {
    if (clazz.equals(Byte.TYPE) || clazz.equals(Byte.class)) {
      return OnnxJavaType.INT8;
    } else if (clazz.equals(Short.TYPE) || clazz.equals(Short.class)) {
      return OnnxJavaType.INT16;
    } else if (clazz.equals(Integer.TYPE) || clazz.equals(Integer.class)) {
      return OnnxJavaType.INT32;
    } else if (clazz.equals(Long.TYPE) || clazz.equals(Long.class)) {
      return OnnxJavaType.INT64;
    } else if (clazz.equals(Float.TYPE) || clazz.equals(Float.class)) {
      return OnnxJavaType.FLOAT;
    } else if (clazz.equals(Double.TYPE) || clazz.equals(Double.class)) {
      return OnnxJavaType.DOUBLE;
    } else if (clazz.equals(Boolean.TYPE) || clazz.equals(Boolean.class)) {
      return OnnxJavaType.BOOL;
    } else if (clazz.equals(String.class)) {
      return OnnxJavaType.STRING;
    } else {
      return OnnxJavaType.UNKNOWN;
    }
  }
}
