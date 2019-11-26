/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.nio.Buffer;

/**
 * A Java object wrapping an OnnxTensor. Tensors are the main input to the library,
 * and can also be returned as outputs.
 */
public class OnnxTensor implements OnnxValue {

    static {
        try {
            OnnxRuntime.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load onnx-runtime library",e);
        }
    }

    private final long nativeHandle;

    private final long allocatorHandle;

    private final TensorInfo info;

    private final Buffer buffer; // This reference is held to ensure the Tensor's backing store doesn't go out of scope.

    OnnxTensor(long nativeHandle, long allocatorHandle, TensorInfo info) {
        this(nativeHandle,allocatorHandle,info,null);
    }

    OnnxTensor(long nativeHandle, long allocatorHandle, TensorInfo info, Buffer buffer) {
        this.nativeHandle = nativeHandle;
        this.allocatorHandle = allocatorHandle;
        this.info = info;
        this.buffer = buffer;
    }

    @Override
    public OnnxValueType getType() {
        return OnnxValueType.ONNX_TYPE_TENSOR;
    }

    long getNativeHandle() {
        return nativeHandle;
    }

    /**
     * Either returns a boxed primitive if the Tensor is a scalar, or a multidimensional array of
     * primitives if it has multiple dimensions.
     * @return A Java value.
     * @throws OrtException If the value could not be extracted as the Tensor is invalid, or if the native code encountered an error.
     */
    @Override
    public Object getValue() throws OrtException {
        if (info.isScalar()) {
            switch (info.type) {
                case FLOAT:
                    return getFloat(OnnxRuntime.ortApiHandle,nativeHandle,info.onnxType.value);
                case DOUBLE:
                    return getDouble(OnnxRuntime.ortApiHandle,nativeHandle);
                case INT8:
                    return getByte(OnnxRuntime.ortApiHandle,nativeHandle,info.onnxType.value);
                case INT16:
                    return getShort(OnnxRuntime.ortApiHandle,nativeHandle,info.onnxType.value);
                case INT32:
                    return getInt(OnnxRuntime.ortApiHandle,nativeHandle,info.onnxType.value);
                case INT64:
                    return getLong(OnnxRuntime.ortApiHandle,nativeHandle,info.onnxType.value);
                case BOOL:
                    return getBool(OnnxRuntime.ortApiHandle,nativeHandle);
                case STRING:
                    return getString(OnnxRuntime.ortApiHandle,nativeHandle,allocatorHandle);
                case UNKNOWN:
                default:
                    throw new OrtException("Extracting the value of an invalid Tensor.");
            }
        } else {
            Object carrier = info.makeCarrier();
            getArray(OnnxRuntime.ortApiHandle,nativeHandle, allocatorHandle, carrier);
            return carrier;
        }
    }

    @Override
    public TensorInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "OnnxTensor(info="+info.toString()+")";
    }

    /**
     * Closes the tensor, releasing it's underlying memory (if it's not backed by an NIO buffer).
     */
    @Override
    public void close() {
        close(OnnxRuntime.ortApiHandle,nativeHandle);
    }

    private native float getFloat(long apiHandle, long nativeHandle, int onnxType) throws OrtException;
    private native double getDouble(long apiHandle, long nativeHandle) throws OrtException;
    private native byte getByte(long apiHandle, long nativeHandle, int onnxType) throws OrtException;
    private native short getShort(long apiHandle, long nativeHandle, int onnxType) throws OrtException;
    private native int getInt(long apiHandle, long nativeHandle, int onnxType) throws OrtException;
    private native long getLong(long apiHandle, long nativeHandle, int onnxType) throws OrtException;
    private native String getString(long apiHandle, long nativeHandle, long allocatorHandle) throws OrtException;
    private native boolean getBool(long apiHandle, long nativeHandle) throws OrtException;

    private native void getArray(long apiHandle, long nativeHandle, long allocatorHandle, Object carrier) throws OrtException;

    private native void close(long apiHandle, long nativeHandle);

}
