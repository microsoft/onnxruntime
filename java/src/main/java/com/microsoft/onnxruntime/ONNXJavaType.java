/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 */
package com.microsoft.onnxruntime;

import com.microsoft.onnxruntime.TensorInfo.ONNXTensorType;

/**
 * An enum over supported Java primitives (and String).
 */
public enum ONNXJavaType {
    FLOAT(1, float.class, 4),
    DOUBLE(2, double.class, 8),
    INT8(3, byte.class, 1),
    INT16(4, short.class, 2),
    INT32(5, int.class, 4),
    INT64(6, long.class, 8),
    BOOL(7, boolean.class, 1),
    STRING(8, String.class, 4),
    UNKNOWN(0, Object.class, 0);

    private static final ONNXJavaType[] values = new ONNXJavaType[9];

    static {
        for (ONNXJavaType ot : ONNXJavaType.values()) {
            values[ot.value] = ot;
        }
    }

    public final int value;
    public final Class<?> clazz;
    public final int size;

    ONNXJavaType(int value, Class<?> clazz, int size) {
        this.value = value;
        this.clazz = clazz;
        this.size = size;
    }

    /**
     * Maps from an int in native land into an ONNXJavaType instance.
     * @param value The value to lookup.
     * @return The enum instance.
     */
    public static ONNXJavaType mapFromInt(int value) {
        if ((value > 0) && (value < values.length)) {
            return values[value];
        } else {
            return UNKNOWN;
        }
    }

    /**
     * Must match the values from ONNXUtil.c.
     *
     * @param onnxValue The value from ONNXUtil.c
     * @return A JavaDataType instance representing the Java type
     */
    public static ONNXJavaType mapFromONNXTensorType(ONNXTensorType onnxValue) {
        switch (onnxValue) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                return ONNXJavaType.INT8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                return ONNXJavaType.INT16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                return ONNXJavaType.INT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                return ONNXJavaType.INT64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return ONNXJavaType.FLOAT;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                return ONNXJavaType.DOUBLE;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
                return ONNXJavaType.STRING;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                return ONNXJavaType.BOOL;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            default:
                return ONNXJavaType.UNKNOWN;
        }
    }

    /**
     * Maps from a Java class object into the enum type, returning {@link ONNXJavaType#UNKNOWN}
     * for unsupported types.
     * @param clazz The class to use.
     * @return An ONNXJavaType instance.
     */
    public static ONNXJavaType mapFromClass(Class<?> clazz) {
        if (clazz.equals(Byte.TYPE) || clazz.equals(Byte.class)) {
            return ONNXJavaType.INT8;
        } else if (clazz.equals(Short.TYPE) || clazz.equals(Short.class)) {
            return ONNXJavaType.INT16;
        } else if (clazz.equals(Integer.TYPE) || clazz.equals(Integer.class)) {
            return ONNXJavaType.INT32;
        } else if (clazz.equals(Long.TYPE) || clazz.equals(Long.class)) {
            return ONNXJavaType.INT64;
        } else if (clazz.equals(Float.TYPE) || clazz.equals(Float.class)) {
            return ONNXJavaType.FLOAT;
        } else if (clazz.equals(Double.TYPE) || clazz.equals(Double.class)) {
            return ONNXJavaType.DOUBLE;
        } else if (clazz.equals(Boolean.TYPE) || clazz.equals(Boolean.class)) {
            return ONNXJavaType.BOOL;
        } else if (clazz.equals(String.class)) {
            return ONNXJavaType.STRING;
        } else {
            return ONNXJavaType.UNKNOWN;
        }
    }
}
