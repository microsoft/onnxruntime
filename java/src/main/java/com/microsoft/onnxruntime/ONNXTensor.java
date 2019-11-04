/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

import java.io.IOException;
import java.nio.Buffer;

/**
 * A Java object wrapping an ONNX Tensor. Tensors are the main input to the library,
 * and can also be returned as outputs.
 */
public class ONNXTensor implements ONNXValue {

    static {
        try {
            ONNX.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ONNX library",e);
        }
    }

    private final long nativeHandle;

    private final long allocatorHandle;

    private final TensorInfo info;

    private final Buffer buffer; // This reference is held to ensure the Tensor's backing store doesn't go out of scope.

    ONNXTensor(long nativeHandle, long allocatorHandle, TensorInfo info) {
        this(nativeHandle,allocatorHandle,info,null);
    }

    ONNXTensor(long nativeHandle, long allocatorHandle, TensorInfo info, Buffer buffer) {
        this.nativeHandle = nativeHandle;
        this.allocatorHandle = allocatorHandle;
        this.info = info;
        this.buffer = buffer;
    }

    @Override
    public ONNXValueType getType() {
        return ONNXValueType.ONNX_TYPE_TENSOR;
    }

    long getNativeHandle() {
        return nativeHandle;
    }

    /**
     * Either returns a boxed primitive if the Tensor is a scalar, or a multidimensional array of
     * primitives if it has multiple dimensions.
     * @return A Java value.
     * @throws ONNXException If the value could not be extracted as the Tensor is invalid, or if the native code encountered an error.
     */
    @Override
    public Object getValue() throws ONNXException {
        if (info.isScalar()) {
            switch (info.type) {
                case FLOAT:
                    return getFloat(ONNX.ortApiHandle,nativeHandle,info.onnxType.value);
                case DOUBLE:
                    return getDouble(ONNX.ortApiHandle,nativeHandle,info.onnxType.value);
                case INT8:
                    return getByte(ONNX.ortApiHandle,nativeHandle,info.onnxType.value);
                case INT16:
                    return getShort(ONNX.ortApiHandle,nativeHandle,info.onnxType.value);
                case INT32:
                    return getInt(ONNX.ortApiHandle,nativeHandle,info.onnxType.value);
                case INT64:
                    return getLong(ONNX.ortApiHandle,nativeHandle,info.onnxType.value);
                case BOOL:
                    return getBool(ONNX.ortApiHandle,nativeHandle,info.onnxType.value);
                case STRING:
                    return getString(ONNX.ortApiHandle,nativeHandle,allocatorHandle,info.onnxType.value);
                case UNKNOWN:
                default:
                    throw new ONNXException("Extracting the value of an invalid Tensor.");
            }
        } else {
            Object carrier = info.makeCarrier();
            getArray(ONNX.ortApiHandle,nativeHandle, allocatorHandle, carrier);
            return carrier;
        }
    }

    @Override
    public TensorInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "ONNXTensor(info="+info.toString()+")";
    }

    /**
     * Closes the tensor, releasing it's underlying memory (if it's not backed by an NIO buffer).
     */
    @Override
    public void close() {
        close(ONNX.ortApiHandle,nativeHandle);
    }

    private native float getFloat(long apiHandle, long nativeHandle, int onnxType) throws ONNXException;
    private native double getDouble(long apiHandle, long nativeHandle, int onnxType) throws ONNXException;
    private native byte getByte(long apiHandle, long nativeHandle, int onnxType) throws ONNXException;
    private native short getShort(long apiHandle, long nativeHandle, int onnxType) throws ONNXException;
    private native int getInt(long apiHandle, long nativeHandle, int onnxType) throws ONNXException;
    private native long getLong(long apiHandle, long nativeHandle, int onnxType) throws ONNXException;
    private native String getString(long apiHandle, long nativeHandle, long allocatorHandle, int onnxType) throws ONNXException;
    private native boolean getBool(long apiHandle, long nativeHandle, int onnxType) throws ONNXException;

    private native void getArray(long apiHandle, long nativeHandle, long allocatorHandle, Object carrier) throws ONNXException;

    private native void close(long apiHandle, long nativeHandle);

}
