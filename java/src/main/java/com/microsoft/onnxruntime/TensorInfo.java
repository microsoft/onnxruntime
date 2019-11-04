/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

import java.lang.reflect.Array;
import java.nio.Buffer;
import java.util.Arrays;

/**
 * Describes a tensor, including it's size, shape and element type.
 */
public class TensorInfo implements ValueInfo {

    /**
     * Maximum number of dimensions supported by these methods.
     */
    public static final int MAX_DIMENSIONS = 8;

    /**
     * The native element types supported by the ONNX runtime.
     */
    public enum ONNXTensorType {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED(0),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8(1),   // maps to c type uint8_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8(2),    // maps to c type int8_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16(3),  // maps to c type uint16_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16(4),   // maps to c type int16_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32(5),      // maps to c type uint32_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32(6),   // maps to c type int32_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64(7),      // maps to c type uint64_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64(8),   // maps to c type int64_t
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16(9),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT(10),   // maps to c type float
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE(11),      // maps to c type double
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING(12),  // maps to c++ type std::string
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL(13),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64(14),   // complex with float32 real and imaginary components
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128(15),  // complex with float64 real and imaginary components
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16(16);    // Non-IEEE floating-point format based on IEEE754 single-precision

        /**
         * The int id on the native side.
         */
        public final int value;

        private static final ONNXTensorType[] values = new ONNXTensorType[17];

        static {
            for (ONNXTensorType ot : ONNXTensorType.values()) {
                values[ot.value] = ot;
            }
        }

        ONNXTensorType(int value) {
            this.value = value;
        }

        /**
         * Maps from the C API's int enum to the Java enum.
         * @param value The index of the Java enum.
         * @return The Java enum.
         */
        public static ONNXTensorType mapFromInt(int value) {
            if ((value > 0) && (value < values.length)) {
                return values[value];
            } else {
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            }
        }

        /**
         * Maps a ONNXJavaType into the appropriate native element type.
         * @param type The type of the Java input/output.
         * @return The native element type.
         */
        public static ONNXTensorType mapFromJavaType(ONNXJavaType type) {
            switch (type) {
                case FLOAT:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
                case DOUBLE:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
                case INT8:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
                case INT16:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
                case INT32:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
                case INT64:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
                case BOOL:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
                case STRING:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
                case UNKNOWN:
                default:
                    return ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            }
        }
    }

    final long[] shape;

    /**
     * The Java type of this tensor.
     */
    public final ONNXJavaType type;

    /**
     * The native type of this tensor.
     */
    public final ONNXTensorType onnxType;

    TensorInfo(long[] shape, ONNXJavaType type, ONNXTensorType onnxType) {
        this.shape = shape;
        this.type = type;
        this.onnxType = onnxType;
    }

    /**
     * Get a copy of the tensor's shape.
     * @return A copy of the tensor's shape.
     */
    public long[] getShape() {
        return Arrays.copyOf(shape,shape.length);
    }

    @Override
    public String toString() {
        return "TensorInfo(javaType="+type.toString()+",onnxType="+onnxType.toString()+",shape="+ Arrays.toString(shape)+")";
    }

    /**
     * Returns true if the shape represents a scalar value (i.e. it has zero dimensions).
     * @return True if the shape is a scalar.
     */
    public boolean isScalar() {
        return shape.length == 0;
    }

    /**
     * Checks that the shape of this tensor info is valid (i.e. positive and within the bounds of a Java int).
     * @return True if the shape is valid.
     */
    private boolean validateShape() {
        return ONNXUtil.validateShape(shape);
    }

    /**
     * Constructs an array the right shape and type to hold this tensor.
     * @return A multidimensional array of the appropriate primitive type (or String).
     * @throws ONNXException If the shape isn't representable in Java (i.e. if one of it's indices is greater than an int).
     */
    public Object makeCarrier() throws ONNXException {
        if (!validateShape()) {
            throw new ONNXException("This tensor is not representable in Java, it's too big - shape = " + Arrays.toString(shape));
        }
        switch (type) {
            case FLOAT:
                return ONNXUtil.newFloatArray(shape);
            case DOUBLE:
                return ONNXUtil.newDoubleArray(shape);
            case INT8:
                return ONNXUtil.newByteArray(shape);
            case INT16:
                return ONNXUtil.newShortArray(shape);
            case INT32:
                return ONNXUtil.newIntArray(shape);
            case INT64:
                return ONNXUtil.newLongArray(shape);
            case BOOL:
                return ONNXUtil.newBooleanArray(shape);
            case STRING:
                return new String[(int)shape[0]];
            case UNKNOWN:
                throw new ONNXException("Can't construct a carrier for an invalid type.");
            default:
                throw new ONNXException("Unsupported type - " + type);
        }
    }

    /**
     * Constructs a TensorInfo from the supplied multidimensional Java array,
     * used to allocate the appropriate amount of native memory.
     * @param obj The object to inspect.
     * @return A TensorInfo which can be used to make the right size Tensor.
     * @throws ONNXException If the supplied Object isn't an array, or is an invalid type.
     */
    public static TensorInfo constructFromJavaArray(Object obj) throws ONNXException {
        Class<?> objClass = obj.getClass();
        if (!objClass.isArray() && !objClass.isPrimitive() && !objClass.equals(String.class)) {
            throw new ONNXException("Cannot convert " + objClass + " to a ONNXTensor.");
        }
        // Figure out base type and number of dimensions.
        int dimensions = 0;
        while (objClass.isArray()) {
            objClass = objClass.getComponentType();
            dimensions++;
        }
        if (!objClass.isPrimitive() && !objClass.equals(String.class)) {
            throw new ONNXException("Cannot create an ONNXTensor from a base type of " + objClass);
        } else if (dimensions > MAX_DIMENSIONS) {
            throw new ONNXException("Cannot create an ONNXTensor with more than " + MAX_DIMENSIONS + " dimensions. Found " + dimensions + " dimensions.");
        } else if ((dimensions > 1) && objClass.equals(String.class)) {
            throw new ONNXException("Cannot create a multidimensional ONNXTensor from Strings. Found " + dimensions + " dimensions.");
        }
        ONNXJavaType javaType = ONNXJavaType.mapFromClass(objClass);

        // Now we extract the shape and validate that the java array is rectangular (i.e. not ragged).
        // this is pretty nasty as we have to look at every object array recursively.
        // Thanks Java!
        long[] shape = new long[dimensions];
        extractShape(shape,0,obj);

        return new TensorInfo(shape,javaType, ONNXTensorType.mapFromJavaType(javaType));
    }

    /**
     * Constructs a TensorInfo from the supplied byte buffer.
     * @param buffer The buffer to inspect.
     * @param shape The shape of the tensor.
     * @param type The Java type.
     * @return A TensorInfo for a tensor.
     * @throws ONNXException If the supplied buffer doesn't match the shape.
     */
    public static TensorInfo constructFromBuffer(Buffer buffer, long[] shape, ONNXJavaType type) throws ONNXException {
        if ((type == ONNXJavaType.STRING) || (type == ONNXJavaType.UNKNOWN)) {
            throw new ONNXException("Cannot create a tensor from a string or unknown buffer.");
        }

        long elementCount = ONNXUtil.elementCount(shape);

        long bufferCapacity = buffer.capacity();

        if (elementCount != bufferCapacity) {
            throw new ONNXException("Shape " + Arrays.toString(shape) + ", requires " + elementCount + " elements but the buffer has " + bufferCapacity + " elements.");
        }

        return new TensorInfo(Arrays.copyOf(shape,shape.length),type,ONNXTensorType.mapFromJavaType(type));
    }

    /**
     * Extracts the shape from a multidimensional array. Checks to see if the array is ragged or not.
     * @param shape The shape array to write to.
     * @param curDim The current dimension to check.
     * @param obj The multidimensional array to inspect.
     * @throws ONNXException If the array has a zero dimension, or is ragged.
     */
    private static void extractShape(long[] shape, int curDim, Object obj) throws ONNXException {
        if (shape.length != curDim) {
            int curLength = Array.getLength(obj);
            if (curLength == 0) {
                throw new ONNXException("Supplied array has a zero dimension at " + curDim + ", all dimensions must be positive");
            } else if (shape[curDim] == 0L) {
                shape[curDim] = curLength;
            } else if (shape[curDim] != curLength) {
                throw new ONNXException("Supplied array is ragged, expected " + shape[curDim] + ", found " + curLength);
            }
            for (int i = 0; i < curLength; i++) {
                extractShape(shape,curDim+1,Array.get(obj,i));
            }
        }
    }
}
