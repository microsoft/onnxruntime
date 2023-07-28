/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.lang.reflect.Array;
import java.nio.Buffer;
import java.util.Arrays;

/** Describes an {@link OnnxTensor}, including it's size, shape and element type. */
public class TensorInfo implements ValueInfo {

  /** Maximum number of dimensions supported by the Java interface methods. */
  public static final int MAX_DIMENSIONS = 8;

  /** The native element types supported by the ONNX runtime. */
  public enum OnnxTensorType {
    /** An undefined element type. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED(0),
    /** An 8-bit unsigned integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8(1), // maps to c type uint8_t
    /** An 8-bit signed integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8(2), // maps to c type int8_t
    /** A 16-bit unsigned integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16(3), // maps to c type uint16_t
    /** A 16-bit signed integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16(4), // maps to c type int16_t
    /** A 32-bit unsigned integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32(5), // maps to c type uint32_t
    /** A 32-bit signed integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32(6), // maps to c type int32_t
    /** A 64-bit unsigned integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64(7), // maps to c type uint64_t
    /** A 64-bit signed integer. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64(8), // maps to c type int64_t
    /** An IEEE 16-bit floating point number. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16(9), // stored as a uint16_t
    /** An IEEE 32-bit floating point number. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT(10), // maps to c type float
    /** An IEEE 64-bit floating point number. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE(11), // maps to c type double
    /** A UTF-8 string. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING(12), // maps to c++ type std::string
    /** A boolean value stored in a byte. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL(13),
    /** A 64-bit complex number, stored as 2 32-bit values. Not accessible from Java. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64(
        14), // complex with float32 real and imaginary components
    /** A 128-bit complex number, stored as 2 64-bit values. Not accessible from Java. */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128(
        15), // complex with float64 real and imaginary components
    /**
     * A non-IEEE 16-bit floating point value with 8 exponent bits and 7 mantissa bits.
     *
     * <p>See <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">Bfloat16 on
     * Wikipedia</a> for more details.
     */
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16(
        16), // Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN(
        17), // Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ(
        18), // Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2(
        19), // Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ(
        20); // Non-IEEE floating-point format based on IEEE754 single-precision

    /** The int id on the native side. */
    public final int value;

    private static final OnnxTensorType[] values = new OnnxTensorType[21];

    static {
      for (OnnxTensorType ot : OnnxTensorType.values()) {
        values[ot.value] = ot;
      }
    }

    OnnxTensorType(int value) {
      this.value = value;
    }

    /**
     * Maps from the C API's int enum to the Java enum.
     *
     * @param value The index of the Java enum.
     * @return The Java enum.
     */
    public static OnnxTensorType mapFromInt(int value) {
      if ((value > 0) && (value < values.length)) {
        return values[value];
      } else {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      }
    }

    /**
     * Maps a OnnxJavaType into the appropriate native element type.
     *
     * @param type The type of the Java input/output.
     * @return The native element type.
     */
    public static OnnxTensorType mapFromJavaType(OnnxJavaType type) {
      switch (type) {
        case FLOAT:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DOUBLE:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case INT8:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case UINT8:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case INT16:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case INT32:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case INT64:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case BOOL:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        case STRING:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
        case FLOAT16:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case BFLOAT16:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        case UNKNOWN:
        default:
          return OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      }
    }
  }

  /** The shape of the tensor. */
  final long[] shape;

  /** The Java type of this tensor. */
  public final OnnxJavaType type;

  /** The native type of this tensor. */
  public final OnnxTensorType onnxType;

  /** The number of elements in this tensor. */
  final long numElements;

  /**
   * Constructs a TensorInfo with the specified shape, Java type and native type.
   *
   * @param shape The tensor shape.
   * @param type The Java type.
   * @param onnxType The native type.
   */
  TensorInfo(long[] shape, OnnxJavaType type, OnnxTensorType onnxType) {
    this.shape = shape;
    this.type = type;
    this.onnxType = onnxType;
    this.numElements = elementCount(shape);
  }

  /**
   * Constructs a TensorInfo with the specified shape and native type int.
   *
   * <p>Called from JNI.
   *
   * @param shape The tensor shape.
   * @param typeInt The native type int.
   */
  TensorInfo(long[] shape, int typeInt) {
    this.shape = shape;
    this.onnxType = OnnxTensorType.mapFromInt(typeInt);
    this.type = OnnxJavaType.mapFromOnnxTensorType(this.onnxType);
    this.numElements = elementCount(shape);
  }

  /**
   * Get a copy of the tensor's shape.
   *
   * @return A copy of the tensor's shape.
   */
  public long[] getShape() {
    return Arrays.copyOf(shape, shape.length);
  }

  @Override
  public String toString() {
    return "TensorInfo(javaType="
        + type.toString()
        + ",onnxType="
        + onnxType.toString()
        + ",shape="
        + Arrays.toString(shape)
        + ")";
  }

  /**
   * Returns true if the shape represents a scalar value (i.e. it has zero dimensions).
   *
   * @return True if the shape is a scalar.
   */
  public boolean isScalar() {
    return shape.length == 0;
  }

  /**
   * Checks that the shape of this tensor info is valid (i.e. positive and within the bounds of a
   * Java int).
   *
   * @return True if the shape is valid.
   */
  private boolean validateShape() {
    return OrtUtil.validateShape(shape);
  }

  /**
   * Computes the number of elements in this tensor.
   *
   * <p>This replicates {@link OrtUtil#elementCount}, but does not throw on negative values which
   * are used for symbolic dimensions in input and output info objects.
   *
   * @param shape The tensor shape.
   * @return The number of elements.
   */
  private static long elementCount(long[] shape) {
    // Java side tensors must be less than Integer.MAX_VALUE,
    // tensors created in native code can be larger, but are not usable in Java.
    // Tensors should not be able to be created which will overflow a 64-bit long.
    long output = 1;
    for (int i = 0; i < shape.length; i++) {
      output *= shape[i];
    }
    return output;
  }

  /**
   * Returns the number of elements in this tensor.
   *
   * <p>If the returned value is negative, then this tensor info refers to an input or output
   * placeholder which has symbolic dimensions, and the element count cannot be computed without
   * specifying the symbolic dimensions.
   *
   * @return The number of elements.
   */
  public long getNumElements() {
    return numElements;
  }

  /**
   * Constructs an array the right shape and type to hold this tensor.
   *
   * <p>Note for String tensors, this carrier is a single dimensional array with enough space for
   * all elements as that's the expected format of the native code. It can be reshaped to the
   * correct shape using {@link OrtUtil#reshape(String[],long[])}.
   *
   * @return A multidimensional array of the appropriate primitive type (or String).
   * @throws OrtException If the shape isn't representable in Java (i.e. if one of its indices is
   *     greater than an int).
   */
  public Object makeCarrier() throws OrtException {
    // Zero length tensors are allowed to be returned.
    if (!validateShape() && numElements != 0) {
      throw new OrtException(
          "This tensor is not representable in Java, it's too big - shape = "
              + Arrays.toString(shape));
    }
    switch (type) {
      case FLOAT:
        return OrtUtil.newFloatArray(shape);
      case DOUBLE:
        return OrtUtil.newDoubleArray(shape);
      case UINT8:
      case INT8:
        return OrtUtil.newByteArray(shape);
      case INT16:
        return OrtUtil.newShortArray(shape);
      case INT32:
        return OrtUtil.newIntArray(shape);
      case INT64:
        return OrtUtil.newLongArray(shape);
      case BOOL:
        return OrtUtil.newBooleanArray(shape);
      case STRING:
        return new String[(int) OrtUtil.elementCount(shape)];
      case UNKNOWN:
        throw new OrtException("Can't construct a carrier for an invalid type.");
      default:
        throw new OrtException("Unsupported type - " + type);
    }
  }

  /**
   * Constructs a TensorInfo from the supplied multidimensional Java array, used to allocate the
   * appropriate amount of native memory.
   *
   * @param obj The object to inspect.
   * @return A TensorInfo which can be used to make the right size Tensor.
   * @throws OrtException If the supplied Object isn't an array, or is an invalid type.
   */
  public static TensorInfo constructFromJavaArray(Object obj) throws OrtException {
    Class<?> objClass = obj.getClass();
    // Check if it's an array or a scalar.
    if (!objClass.isArray()) {
      // Check if it's a valid non-array type
      OnnxJavaType javaType = OnnxJavaType.mapFromClass(objClass);
      if (javaType == OnnxJavaType.UNKNOWN) {
        throw new OrtException("Cannot convert " + objClass + " to a OnnxTensor.");
      } else {
        // scalar primitive
        return new TensorInfo(new long[0], javaType, OnnxTensorType.mapFromJavaType(javaType));
      }
    }
    // Figure out base type and number of dimensions.
    int dimensions = 0;
    while (objClass.isArray()) {
      objClass = objClass.getComponentType();
      dimensions++;
    }
    if (!objClass.isPrimitive() && !objClass.equals(String.class)) {
      throw new OrtException("Cannot create an OnnxTensor from a base type of " + objClass);
    } else if (dimensions > MAX_DIMENSIONS) {
      throw new OrtException(
          "Cannot create an OnnxTensor with more than "
              + MAX_DIMENSIONS
              + " dimensions. Found "
              + dimensions
              + " dimensions.");
    }
    OnnxJavaType javaType = OnnxJavaType.mapFromClass(objClass);

    // Now we extract the shape and validate that the java array is rectangular (i.e. not ragged).
    // this is pretty nasty as we have to look at every object array recursively.
    // Thanks Java!
    long[] shape = new long[dimensions];
    extractShape(shape, 0, obj);

    return new TensorInfo(shape, javaType, OnnxTensorType.mapFromJavaType(javaType));
  }

  /**
   * Constructs a TensorInfo from the supplied byte buffer.
   *
   * @param buffer The buffer to inspect.
   * @param shape The shape of the tensor.
   * @param type The Java type.
   * @return A TensorInfo for a tensor.
   * @throws OrtException If the supplied buffer doesn't match the shape.
   */
  public static TensorInfo constructFromBuffer(Buffer buffer, long[] shape, OnnxJavaType type)
      throws OrtException {
    if ((type == OnnxJavaType.STRING) || (type == OnnxJavaType.UNKNOWN)) {
      throw new OrtException("Cannot create a tensor from a string or unknown buffer.");
    }

    long elementCount = OrtUtil.elementCount(shape);

    long bufferRemaining = buffer.remaining();

    // Check if size matches
    if (elementCount != bufferRemaining) {
      // if not it could be a ByteBuffer passed in, so check how many bytes there are
      long elemRemaining = bufferRemaining / type.size;
      if (elementCount != elemRemaining) {
        throw new OrtException(
            "Shape "
                + Arrays.toString(shape)
                + ", requires "
                + elementCount
                + " elements but the buffer has "
                + bufferRemaining
                + " elements.");
      }
    }

    return new TensorInfo(
        Arrays.copyOf(shape, shape.length), type, OnnxTensorType.mapFromJavaType(type));
  }

  /**
   * Constructs a TensorInfo from the supplied {@link OnnxSparseTensor.SparseTensor}.
   *
   * @param tensor The sparse tensor.
   * @param <T> The buffer type.
   * @return A TensorInfo for a sparse tensor.
   * @throws OrtException If the supplied tensor has too many elements for it's shape.
   */
  public static <T extends Buffer> TensorInfo constructFromSparseTensor(
      OnnxSparseTensor.SparseTensor<T> tensor) throws OrtException {
    long[] shape = tensor.getDenseShape();

    long elementCount = OrtUtil.elementCount(shape);

    long bufferRemaining = tensor.getValues().remaining();

    if (elementCount < bufferRemaining) {
      throw new OrtException(
          "Shape "
              + Arrays.toString(shape)
              + ", has at most "
              + elementCount
              + " elements but the buffer has "
              + bufferRemaining
              + " elements.");
    }

    return new TensorInfo(
        Arrays.copyOf(shape, shape.length),
        tensor.getType(),
        OnnxTensorType.mapFromJavaType(tensor.getType()));
  }

  /**
   * Extracts the shape from a multidimensional array. Checks to see if the array is ragged or not.
   *
   * @param shape The shape array to write to.
   * @param curDim The current dimension to check.
   * @param obj The multidimensional array to inspect.
   * @throws OrtException If the array has a zero dimension, or is ragged.
   */
  private static void extractShape(long[] shape, int curDim, Object obj) throws OrtException {
    if (shape.length != curDim) {
      int curLength = Array.getLength(obj);
      if (curLength == 0) {
        throw new OrtException(
            "Supplied array has a zero dimension at "
                + curDim
                + ", all dimensions must be positive");
      } else if (shape[curDim] == 0L) {
        shape[curDim] = curLength;
      } else if (shape[curDim] != curLength) {
        throw new OrtException(
            "Supplied array is ragged, expected " + shape[curDim] + ", found " + curLength);
      }
      for (int i = 0; i < curLength; i++) {
        extractShape(shape, curDim + 1, Array.get(obj, i));
      }
    }
  }
}
