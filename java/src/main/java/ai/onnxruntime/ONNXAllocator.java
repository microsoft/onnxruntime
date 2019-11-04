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
 * A wrapper around the onnx memory allocator.
 */
public class ONNXAllocator implements AutoCloseable {

    final long handle;

    private boolean closed = false;

    static {
        try {
            ONNX.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ONNX library",e);
        }
    }

    /**
     * Constructs a CPU allocator.
     * @throws ONNXException If the onnx runtime threw an error.
     */
    public ONNXAllocator() throws ONNXException {
        handle = createAllocator(ONNX.ortApiHandle);
    }

    ONNXAllocator(long handle) {
        this.handle = handle;
    }

    /**
     * Create a Tensor from a Java primitive (or String) multidimensional array.
     * The shape is inferred from the array using reflection.
     * @param data The data to store in a tensor.
     * @return An ONNXTensor storing the data.
     * @throws ONNXException If the onnx runtime threw an error.
     */
    public ONNXTensor createTensor(Object data) throws ONNXException {
        if (!closed) {
            TensorInfo info = TensorInfo.constructFromJavaArray(data);
            if (info.type == ONNXJavaType.STRING) {
                if (info.shape.length == 0) {
                    return new ONNXTensor(createString(ONNX.ortApiHandle, handle,(String)data), handle, info);
                } else {
                    return new ONNXTensor(createStringTensor(ONNX.ortApiHandle, handle, ONNXUtil.flattenString(data), info.shape), handle, info);
                }
            } else {
                if (info.shape.length == 0) {
                    data = convertBoxedPrimitiveToArray(data);
                }
                return new ONNXTensor(createTensor(ONNX.ortApiHandle, handle, data, info.shape, info.onnxType.value), handle, info);
            }
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Create a tensor from a flattened string array.
     * <p>
     * Requires the array to be flattened in row-major order.
     * @param data The tensor data
     * @param shape the shape of the tensor
     * @return An ONNXTensor of the required shape.
     * @throws ONNXException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public ONNXTensor createTensor(String[] data, long[]shape) throws ONNXException {
        if (!closed) {
            TensorInfo info = new TensorInfo(shape, ONNXJavaType.STRING, TensorInfo.ONNXTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
            return new ONNXTensor(createStringTensor(ONNX.ortApiHandle, handle, data, shape), handle, info);
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Create an ONNXTensor backed by a direct FloatBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An ONNXTensor of the required shape.
     * @throws ONNXException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public ONNXTensor createTensor(FloatBuffer data, long[] shape) throws ONNXException {
        if (!closed) {
            ONNXJavaType type = ONNXJavaType.FLOAT;
            int bufferSize = data.capacity()*type.size;
            FloatBuffer tmp;
            if (data.isDirect()) {
                tmp = data;
            } else {
                ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                tmp = buffer.asFloatBuffer();
                tmp.put(data);
            }
            TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
            return new ONNXTensor(createTensorFromBuffer(ONNX.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Create an ONNXTensor backed by a direct DoubleBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An ONNXTensor of the required shape.
     * @throws ONNXException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public ONNXTensor createTensor(DoubleBuffer data, long[] shape) throws ONNXException {
        if (!closed) {
            ONNXJavaType type = ONNXJavaType.DOUBLE;
            int bufferSize = data.capacity()*type.size;
            DoubleBuffer tmp;
            if (data.isDirect()) {
                tmp = data;
            } else {
                ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                tmp = buffer.asDoubleBuffer();
                tmp.put(data);
            }
            TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
            return new ONNXTensor(createTensorFromBuffer(ONNX.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Create an ONNXTensor backed by a direct ByteBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An ONNXTensor of the required shape.
     * @throws ONNXException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public ONNXTensor createTensor(ByteBuffer data, long[] shape) throws ONNXException {
        if (!closed) {
            ONNXJavaType type = ONNXJavaType.INT8;
            int bufferSize = data.capacity()*type.size;
            ByteBuffer tmp;
            if (data.isDirect()) {
                tmp = data;
            } else {
                tmp = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                tmp.put(data);
            }
            TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
            return new ONNXTensor(createTensorFromBuffer(ONNX.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Create an ONNXTensor backed by a direct ShortBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An ONNXTensor of the required shape.
     * @throws ONNXException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public ONNXTensor createTensor(ShortBuffer data, long[] shape) throws ONNXException {
        if (!closed) {
            ONNXJavaType type = ONNXJavaType.INT16;
            int bufferSize = data.capacity()*type.size;
            ShortBuffer tmp;
            if (data.isDirect()) {
                tmp = data;
            } else {
                ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                tmp = buffer.asShortBuffer();
                tmp.put(data);
            }
            TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
            return new ONNXTensor(createTensorFromBuffer(ONNX.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Create an ONNXTensor backed by a direct IntBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An ONNXTensor of the required shape.
     * @throws ONNXException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public ONNXTensor createTensor(IntBuffer data, long[] shape) throws ONNXException {
        if (!closed) {
            ONNXJavaType type = ONNXJavaType.INT32;
            int bufferSize = data.capacity()*type.size;
            IntBuffer tmp;
            if (data.isDirect()) {
                tmp = data;
            } else {
                ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                tmp = buffer.asIntBuffer();
                tmp.put(data);
            }
            TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
            return new ONNXTensor(createTensorFromBuffer(ONNX.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Create an ONNXTensor backed by a direct LongBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An ONNXTensor of the required shape.
     * @throws ONNXException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public ONNXTensor createTensor(LongBuffer data, long[] shape) throws ONNXException {
        if (!closed) {
            ONNXJavaType type = ONNXJavaType.INT64;
            int bufferSize = data.capacity()*type.size;
            LongBuffer tmp;
            if (data.isDirect()) {
                tmp = data;
            } else {
                ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                tmp = buffer.asLongBuffer();
                tmp.put(data);
            }
            TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
            return new ONNXTensor(createTensorFromBuffer(ONNX.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an ONNXTensor on a closed ONNXSession.");
        }
    }

    /**
     * Closes the allocator, must be done after all it's child objects have been closed.
     * @throws ONNXException If it failed to close.
     */
    @Override
    public void close() throws ONNXException {
        if (!closed) {
            //closeAllocator(ONNX.ortApiHandle,handle);
            closed = true;
        } else {
            throw new IllegalStateException("Trying to close an already closed ONNXSession.");
        }
    }

    /**
     * Stores a boxed primitive in a single element array of the boxed type.
     * Otherwise returns the input.
     * @param data The boxed primitive.
     * @return The boxed primitive in an array.
     */
    private Object convertBoxedPrimitiveToArray(Object data) {
        if (data instanceof Boolean) {
            return new Boolean[]{(Boolean)data};
        } else if (data instanceof Byte) {
            return new Byte[]{(Byte)data};
        } else if (data instanceof Short) {
            return new Short[]{(Short)data};
        } else if (data instanceof Integer) {
            return new Integer[]{(Integer)data};
        } else if (data instanceof Long) {
            return new Long[]{(Long)data};
        } else if (data instanceof Float) {
            return new Float[]{(Float)data};
        } else if (data instanceof Double) {
            return new double[]{(Double)data};
        } else {
            return data;
        }
    }

    private native long createAllocator(long apiHandle) throws ONNXException;

    //private native void closeAllocator(long apiHandle, long nativeHandle) throws ONNXException;

    private native long createTensor(long apiHandle, long allocatorHandle, Object data, long[] shape, int onnxType) throws ONNXException;
    private native long createTensorFromBuffer(long apiHandle, long allocatorHandle, Buffer data, long bufferSize, long[] shape, int onnxType) throws ONNXException;

    private native long createString(long apiHandle, long allocatorHandle, String data) throws ONNXException;
    private native long createStringTensor(long apiHandle, long allocatorHandle, Object[] data, long[] shape) throws ONNXException;
}