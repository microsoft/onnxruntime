/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.TensorInfo.OnnxTensorType;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

/**
 * An ONNX Runtime memory allocator.
 */
public class OrtAllocator implements AutoCloseable {

    final long handle;

    private boolean closed = false;

    static {
        try {
            OnnxRuntime.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load onnx-runtime library",e);
        }
    }

    /**
     * Constructs the default CPU allocator.
     * @throws OrtException If the onnx runtime threw an error.
     */
    public OrtAllocator() throws OrtException {
        handle = createAllocator(OnnxRuntime.ortApiHandle);
    }

    OrtAllocator(long handle) {
        this.handle = handle;
    }

    /**
     * Create a Tensor from a Java primitive (or String) multidimensional array.
     * The shape is inferred from the array using reflection.
     * @param data The data to store in a tensor.
     * @return An OnnxTensor storing the data.
     * @throws OrtException If the onnx runtime threw an error.
     */
    public OnnxTensor createTensor(Object data) throws OrtException {
        if (!closed) {
            TensorInfo info = TensorInfo.constructFromJavaArray(data);
            if (info.type == OnnxJavaType.STRING) {
                if (info.shape.length == 0) {
                    return new OnnxTensor(createString(OnnxRuntime.ortApiHandle, handle,(String)data), handle, info);
                } else {
                    return new OnnxTensor(createStringTensor(OnnxRuntime.ortApiHandle, handle, OrtUtil.flattenString(data), info.shape), handle, info);
                }
            } else {
                if (info.shape.length == 0) {
                    data = convertBoxedPrimitiveToArray(data);
                }
                return new OnnxTensor(createTensor(OnnxRuntime.ortApiHandle, handle, data, info.shape, info.onnxType.value), handle, info);
            }
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create a tensor from a flattened string array.
     * <p>
     * Requires the array to be flattened in row-major order.
     * @param data The tensor data
     * @param shape the shape of the tensor
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(String[] data, long[]shape) throws OrtException {
        if (!closed) {
            TensorInfo info = new TensorInfo(shape, OnnxJavaType.STRING, OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
            return new OnnxTensor(createStringTensor(OnnxRuntime.ortApiHandle, handle, data, shape), handle, info);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct FloatBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(FloatBuffer data, long[] shape) throws OrtException {
        if (!closed) {
            OnnxJavaType type = OnnxJavaType.FLOAT;
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct DoubleBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(DoubleBuffer data, long[] shape) throws OrtException {
        if (!closed) {
            OnnxJavaType type = OnnxJavaType.DOUBLE;
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct ByteBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(ByteBuffer data, long[] shape) throws OrtException {
        if (!closed) {
            OnnxJavaType type = OnnxJavaType.INT8;
            int bufferSize = data.capacity()*type.size;
            ByteBuffer tmp;
            if (data.isDirect()) {
                tmp = data;
            } else {
                tmp = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                tmp.put(data);
            }
            TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct ShortBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(ShortBuffer data, long[] shape) throws OrtException {
        if (!closed) {
            OnnxJavaType type = OnnxJavaType.INT16;
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct IntBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(IntBuffer data, long[] shape) throws OrtException {
        if (!closed) {
            OnnxJavaType type = OnnxJavaType.INT32;
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct LongBuffer.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(LongBuffer data, long[] shape) throws OrtException {
        if (!closed) {
            OnnxJavaType type = OnnxJavaType.INT64;
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, handle, tmp, bufferSize, shape, info.onnxType.value), handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Closes the allocator, must be done after all it's child objects have been closed.
     * @throws OrtException If it failed to close.
     */
    @Override
    public void close() throws OrtException {
        if (!closed) {
            // Turned off as it can only construct the default allocator, which cannot be closed.
            // Will need to be enabled when non-default allocators can be constructed.
            //closeAllocator(ONNX.ortApiHandle,handle);
            closed = true;
        } else {
            throw new IllegalStateException("Trying to close an already closed OrtAllocator.");
        }
    }

    /**
     * Stores a boxed primitive in a single element array of the boxed type.
     * Otherwise returns the input.
     * @param data The boxed primitive.
     * @return The boxed primitive in an array.
     */
    private Object convertBoxedPrimitiveToArray(Object data) {
        Object array = Array.newInstance(data.getClass(), 1);
        Array.set(array, 0, data);
        return array;
    }

    private native long createAllocator(long apiHandle) throws OrtException;

    // The default allocator cannot be closed. When support for non-default allocators is added this method will need to be re-enabled.
    //private native void closeAllocator(long apiHandle, long nativeHandle) throws ONNXException;

    private native long createTensor(long apiHandle, long allocatorHandle, Object data, long[] shape, int onnxType) throws OrtException;
    private native long createTensorFromBuffer(long apiHandle, long allocatorHandle, Buffer data, long bufferSize, long[] shape, int onnxType) throws OrtException;

    private native long createString(long apiHandle, long allocatorHandle, String data) throws OrtException;
    private native long createStringTensor(long apiHandle, long allocatorHandle, Object[] data, long[] shape) throws OrtException;
}