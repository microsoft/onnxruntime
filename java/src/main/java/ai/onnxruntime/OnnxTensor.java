/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

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

    /**
     * Returns a copy of the underlying OnnxTensor as a ByteBuffer.
     *
     * This method returns null if the OnnxTensor contains Strings as they are stored externally to the OnnxTensor.
     *
     * @return A ByteBuffer copy of the OnnxTensor.
     */
    public ByteBuffer getByteBuffer() {
        if (info.type != OnnxJavaType.STRING) {
            ByteBuffer buffer = getBuffer(OnnxRuntime.ortApiHandle,nativeHandle);
            ByteBuffer output = ByteBuffer.allocate(buffer.capacity());
            output.put(buffer);
            output.rewind();
            return output;
        } else {
            return null;
        }
    }

    /**
     * Returns a copy of the underlying OnnxTensor as a FloatBuffer
     * if it can be losslessly converted into a float (i.e. it's a float or fp16), otherwise it
     * returns null.
     * @return A FloatBuffer copy of the OnnxTensor.
     */
    public FloatBuffer getFloatBuffer() {
        if (info.type == OnnxJavaType.FLOAT) {
            if (info.onnxType == TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                // if it's fp16 we need to copy it out by hand.
                ShortBuffer buffer = getBuffer().asShortBuffer();
                int bufferCap = buffer.capacity();
                FloatBuffer output = FloatBuffer.allocate(bufferCap);
                for (int i = 0; i < bufferCap; i++) {
                    output.put(fp16ToFloat(buffer.get(i)));
                }
                output.rewind();
                return output;
            } else {
                // if it's fp32 use the efficient copy.
                FloatBuffer buffer = getBuffer().asFloatBuffer();
                FloatBuffer output = FloatBuffer.allocate(buffer.capacity());
                output.put(buffer);
                output.rewind();
                return output;
            }
        } else {
            return null;
        }
    }

    /**
     * Returns a copy of the underlying OnnxTensor as a DoubleBuffer if the underlying type is
     * a double, otherwise it returns null.
     * the
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
     * Returns a copy of the underlying OnnxTensor as a ShortBuffer if the
     * underlying type is int16 or uint16, otherwise it returns null.
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
     * Returns a copy of the underlying OnnxTensor as an IntBuffer if
     * the underlying type is int32 or uint32, otherwise it returns null.
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
     * Returns a copy of the underlying OnnxTensor as a LongBuffer
     * if the underlying type is int64 or uint64, otherwise it returns null.
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
     * Wraps the OrtTensor pointer in a direct byte buffer of the native platform endian-ness.
     * Unless you really know what you're doing, you want this one rather than the native call {@link OnnxTensor#getBuffer(long,long)}.
     * @return A ByteBuffer wrapping the data.
     */
    private ByteBuffer getBuffer() {
        return getBuffer(OnnxRuntime.ortApiHandle,nativeHandle).order(ByteOrder.nativeOrder());
    }

    /**
     * Wraps the OrtTensor pointer in a direct byte buffer.
     * @param apiHandle The OrtApi pointer.
     * @param nativeHandle The OrtTensor pointer.
     * @return A ByteBuffer wrapping the data.
     */
    private native ByteBuffer getBuffer(long apiHandle, long nativeHandle);

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

    /**
     * Mirrors the conversion in the C code. It's not precise if there are subnormal values,
     * nor does it preserve all the different kinds of NaNs (which aren't representable in Java anyway).
     * @param input A uint16_t representing an IEEE half precision float.
     * @return A float.
     */
    private static float fp16ToFloat(short input) {
        int output = ((input&0x8000)<<16) | (((input&0x7c00)+0x1C000)<<13) | ((input&0x03FF)<<13);
        return Float.intBitsToFloat(output);
    }

}
