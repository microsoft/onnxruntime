/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

/**
 * The host object for the onnx-runtime system. Can create {@link OrtSession}s
 * which encapsulate specific models.
 */
public class OrtEnvironment implements AutoCloseable {

    /**
     * The logging level for messages from the environment and session.
     */
    public enum LoggingLevel {
        ORT_LOGGING_LEVEL_VERBOSE(0),
        ORT_LOGGING_LEVEL_INFO(1),
        ORT_LOGGING_LEVEL_WARNING(2),
        ORT_LOGGING_LEVEL_ERROR(3),
        ORT_LOGGING_LEVEL_FATAL(4);
        private final int value;

        LoggingLevel(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    private static final Logger logger = Logger.getLogger(OrtEnvironment.class.getName());

    public static final String DEFAULT_NAME = "ort-java";

    static {
        try {
            OnnxRuntime.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load onnx-runtime library",e);
        }
    }

    private static volatile OrtEnvironment INSTANCE;

    private static final AtomicInteger refCount = new AtomicInteger();

    private static volatile LoggingLevel curLogLevel;

    private static volatile String curName;

    /**
     * Gets the OrtEnvironment. If there is not an environment currently created, it creates one
     * using {@link OrtEnvironment#DEFAULT_NAME} and {@link LoggingLevel#ORT_LOGGING_LEVEL_WARNING}.
     * @return An onnxruntime environment.
     */
    public static OrtEnvironment getEnvironment() {
        return getEnvironment(LoggingLevel.ORT_LOGGING_LEVEL_WARNING,DEFAULT_NAME);
    }

    /**
     * Gets the OrtEnvironment. If there is not an environment currently created, it creates one
     * using the supplied name and {@link LoggingLevel#ORT_LOGGING_LEVEL_WARNING}.
     * @param name The name of the environment used in logging.
     * @return An onnxruntime environment.
     */
    public static OrtEnvironment getEnvironment(String name) {
        return getEnvironment(LoggingLevel.ORT_LOGGING_LEVEL_WARNING,name);
    }

    /**
     * Gets the OrtEnvironment. If there is not an environment currently created, it creates one
     * using the {@link OrtEnvironment#DEFAULT_NAME} and the supplied logging level.
     * @param logLevel The logging level to use.
     * @return An onnxruntime environment.
     */
    public static OrtEnvironment getEnvironment(LoggingLevel logLevel) {
        return getEnvironment(logLevel,DEFAULT_NAME);
    }

    /**
     * Gets the OrtEnvironment. If there is not an environment currently created, it creates one
     * using the supplied name and logging level. If an environment already exists with a different name,
     * that environment is returned and a warning is logged.
     * @param loggingLevel The logging level to use.
     * @param name The name to log.
     * @return The OrtEnvironment singleton.
     */
    public static synchronized OrtEnvironment getEnvironment(LoggingLevel loggingLevel, String name) {
        if (INSTANCE == null) {
            try {
                INSTANCE = new OrtEnvironment(loggingLevel, name);
                curLogLevel = loggingLevel;
                curName = name;
            } catch (OrtException e) {
                throw new IllegalStateException("Failed to create OrtEnvironment",e);
            }
        } else {
            if ((loggingLevel.value != curLogLevel.value) || (!name.equals(curName))) {
                logger.warning("Tried to change OrtEnvironment's logging level or name while a reference exists.");
            }
        }
        refCount.incrementAndGet();
        return INSTANCE;
    }

    final long nativeHandle;

    final OrtAllocator defaultAllocator;

    private boolean closed = false;

    /**
     * Create an OrtEnvironment using a default name.
     * @throws OrtException If the environment couldn't be created.
     */
    private OrtEnvironment() throws OrtException {
        this(LoggingLevel.ORT_LOGGING_LEVEL_WARNING,"java-default");
    }

    /**
     * Create an OrtEnvironment using the specified name and log level.
     * @param loggingLevel The logging level to use.
     * @param name The environment name.
     * @throws OrtException If the environment couldn't be created.
     */
    private OrtEnvironment(LoggingLevel loggingLevel, String name) throws OrtException {
        nativeHandle = createHandle(OnnxRuntime.ortApiHandle,loggingLevel.getValue(),name);
        defaultAllocator = new OrtAllocator(getDefaultAllocator(OnnxRuntime.ortApiHandle),true);
    }

    /**
     * Create a Tensor from a Java primitive or String multidimensional array.
     * The shape is inferred from the array using reflection.
     * The default allocator is used.
     * @param data The data to store in a tensor.
     * @return An OnnxTensor storing the data.
     * @throws OrtException If the onnx runtime threw an error.
     */
    public OnnxTensor createTensor(Object data) throws OrtException {
        return createTensor(defaultAllocator,data);
    }

    /**
     * Create a Tensor from a Java primitive or String multidimensional array.
     * The shape is inferred from the array using reflection.
     * @param allocator The allocator to use.
     * @param data The data to store in a tensor.
     * @return An OnnxTensor storing the data.
     * @throws OrtException If the onnx runtime threw an error.
     */
    OnnxTensor createTensor(OrtAllocator allocator, Object data) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
            TensorInfo info = TensorInfo.constructFromJavaArray(data);
            if (info.type == OnnxJavaType.STRING) {
                if (info.shape.length == 0) {
                    return new OnnxTensor(createString(OnnxRuntime.ortApiHandle, allocator.handle,(String)data), allocator.handle, info);
                } else {
                    return new OnnxTensor(createStringTensor(OnnxRuntime.ortApiHandle, allocator.handle, OrtUtil.flattenString(data), info.shape), allocator.handle, info);
                }
            } else {
                if (info.shape.length == 0) {
                    data = convertBoxedPrimitiveToArray(data);
                }
                return new OnnxTensor(createTensor(OnnxRuntime.ortApiHandle, allocator.handle, data, info.shape, info.onnxType.value), allocator.handle, info);
            }
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor with a closed OrtAllocator.");
        }
    }

    /**
     * Create a tensor from a flattened string array.
     * <p>
     * Requires the array to be flattened in row-major order. Uses the default allocator.
     * @param data The tensor data
     * @param shape the shape of the tensor
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(String[] data, long[] shape) throws OrtException {
        return createTensor(defaultAllocator,data,shape);
    }

    /**
     * Create a tensor from a flattened string array.
     * <p>
     * Requires the array to be flattened in row-major order.
     * @param allocator The allocator to use.
     * @param data The tensor data
     * @param shape the shape of the tensor
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    OnnxTensor createTensor(OrtAllocator allocator, String[] data, long[] shape) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
            TensorInfo info = new TensorInfo(shape, OnnxJavaType.STRING, OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
            return new OnnxTensor(createStringTensor(OnnxRuntime.ortApiHandle, allocator.handle, data, shape), allocator.handle, info);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor. Uses the default allocator.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(FloatBuffer data, long[] shape) throws OrtException {
        return createTensor(defaultAllocator,data,shape);
    }

    /**
     * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param allocator The allocator to use.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    OnnxTensor createTensor(OrtAllocator allocator, FloatBuffer data, long[] shape) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, allocator.handle, tmp, bufferSize, shape, info.onnxType.value), allocator.handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor. Uses the default allocator.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(DoubleBuffer data, long[] shape) throws OrtException {
        return createTensor(defaultAllocator,data,shape);
    }

    /**
     * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param allocator The allocator to use.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    OnnxTensor createTensor(OrtAllocator allocator, DoubleBuffer data, long[] shape) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, allocator.handle, tmp, bufferSize, shape, info.onnxType.value), allocator.handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor. Uses the default allocator.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(ByteBuffer data, long[] shape) throws OrtException {
        return createTensor(defaultAllocator,data,shape);
    }

    /**
     * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param allocator The allocator to use.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    OnnxTensor createTensor(OrtAllocator allocator, ByteBuffer data, long[] shape) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, allocator.handle, tmp, bufferSize, shape, info.onnxType.value), allocator.handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor. Uses the default allocator.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(ShortBuffer data, long[] shape) throws OrtException {
        return createTensor(defaultAllocator,data,shape);
    }

    /**
     * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param allocator The allocator to use.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    OnnxTensor createTensor(OrtAllocator allocator, ShortBuffer data, long[] shape) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, allocator.handle, tmp, bufferSize, shape, info.onnxType.value), allocator.handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor. Uses the default allocator.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(IntBuffer data, long[] shape) throws OrtException {
        return createTensor(defaultAllocator,data,shape);
    }

    /**
     * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor.
     * @param allocator The allocator to use.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    OnnxTensor createTensor(OrtAllocator allocator, IntBuffer data, long[] shape) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, allocator.handle, tmp, bufferSize, shape, info.onnxType.value), allocator.handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor. Uses the default allocator.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    public OnnxTensor createTensor(LongBuffer data, long[] shape) throws OrtException {
        return createTensor(defaultAllocator,data,shape);
    }

    /**
     * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
     *
     * If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
     * of the tensor. Uses the supplied allocator.
     * @param allocator The allocator to use.
     * @param data The tensor data.
     * @param shape The shape of tensor.
     * @return An OnnxTensor of the required shape.
     * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
     */
    OnnxTensor createTensor(OrtAllocator allocator, LongBuffer data, long[] shape) throws OrtException {
        if ((!closed) && (!allocator.isClosed())) {
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
            return new OnnxTensor(createTensorFromBuffer(OnnxRuntime.ortApiHandle, allocator.handle, tmp, bufferSize, shape, info.onnxType.value), allocator.handle, info, tmp);
        } else {
            throw new IllegalStateException("Trying to create an OnnxTensor on a closed OrtAllocator.");
        }
    }

    /**
     * Create a session using the specified {@link SessionOptions}, model and the default memory allocator.
     * @param modelPath Path on disk to load the model from.
     * @param options The session options.
     * @return An {@link OrtSession} with the specified model.
     * @throws OrtException If the model failed to load, wasn't compatible or caused an error.
     */
    public OrtSession createSession(String modelPath, SessionOptions options) throws OrtException {
        return createSession(modelPath,defaultAllocator,options);
    }

    /**
     * Create a session using the specified {@link SessionOptions} and model.
     * @param modelPath Path on disk to load the model from.
     * @param allocator The memory allocator to use.
     * @param options The session options.
     * @return An {@link OrtSession} with the specified model.
     * @throws OrtException If the model failed to load, wasn't compatible or caused an error.
     */
    OrtSession createSession(String modelPath, OrtAllocator allocator, SessionOptions options) throws OrtException {
        if (!closed) {
            return new OrtSession(this,modelPath,allocator,options);
        } else {
            throw new IllegalStateException("Trying to create an OrtSession on a closed OrtEnvironment.");
        }
    }

    /**
     * Create a session using the specified {@link SessionOptions}, model and the default memory allocator.
     * @param modelArray Byte array representing an ONNX model.
     * @param options The session options.
     * @return An {@link OrtSession} with the specified model.
     * @throws OrtException If the model failed to parse, wasn't compatible or caused an error.
     */
    public OrtSession createSession(byte[] modelArray, SessionOptions options) throws OrtException {
        return createSession(modelArray,defaultAllocator,options);
    }

    /**
     * Create a session using the specified {@link SessionOptions} and model.
     * @param modelArray Byte array representing an ONNX model.
     * @param allocator The memory allocator to use.
     * @param options The session options.
     * @return An {@link OrtSession} with the specified model.
     * @throws OrtException If the model failed to parse, wasn't compatible or caused an error.
     */
    OrtSession createSession(byte[] modelArray, OrtAllocator allocator, SessionOptions options) throws OrtException {
        if (!closed) {
            return new OrtSession(this, modelArray, allocator, options);
        } else {
            throw new IllegalStateException("Trying to create an OrtSession on a closed OrtEnvironment.");
        }
    }

    /**
     * Turns on or off the telemetry.
     * @param sendTelemetry If true then send telemetry on onnxruntime usage.
     * @throws OrtException If the call failed.
     */
    public void setTelemetry(boolean sendTelemetry) throws OrtException {
        setTelemetry(OnnxRuntime.ortApiHandle,nativeHandle,sendTelemetry);
    }

    @Override
    public String toString() {
        return "OrtEnvironment(name="+curName+",logLevel="+curLogLevel+")";
    }

    /**
     * Closes the OrtEnvironment. If this is the last reference to the environment then it closes the native handle.
     * @throws OrtException If the close failed.
     */
    @Override
    public synchronized void close() throws OrtException {
        closed = true;
        synchronized (refCount) {
            int curCount = refCount.get();
            if (curCount != 0) {
                refCount.decrementAndGet();
            }
            if (curCount == 1) {
                close(OnnxRuntime.ortApiHandle, nativeHandle);
                INSTANCE = null;
            }
        }
    }

    /**
     * Stores a boxed primitive in a single element array of the boxed type.
     * Otherwise returns the input.
     * @param data The boxed primitive.
     * @return The boxed primitive in an array.
     */
    private static Object convertBoxedPrimitiveToArray(Object data) {
        Object array = Array.newInstance(data.getClass(), 1);
        Array.set(array, 0, data);
        return array;
    }

    /**
     * Creates the native object.
     * @param apiHandle The API pointer.
     * @param loggingLevel The logging level.
     * @param name The name of the environment.
     * @return The pointer to the native object.
     * @throws OrtException If the creation failed.
     */
    private static native long createHandle(long apiHandle, int loggingLevel, String name) throws OrtException;

    /**
     * Gets a reference to the default allocator.
     * @param apiHandle The API handle to use.
     * @return A pointer to the default allocator.
     * @throws OrtException If it failed to get the allocator.
     */
    private static native long getDefaultAllocator(long apiHandle) throws OrtException;

    /**
     * Closes the OrtEnvironment, frees the handle.
     * @param apiHandle The API pointer.
     * @param nativeHandle The handle to free.
     * @throws OrtException If an error was caused by freeing the handle.
     */
    private static native void close(long apiHandle, long nativeHandle) throws OrtException;

    /**
     * Enables or disables the telemetry.
     * @param apiHandle The API pointer.
     * @param nativeHandle The native handle for the environment.
     * @param sendTelemetry Turn on or off the telemetry.
     * @throws OrtException If an error was caused when setting the telemetry status.
     */
    private static native void setTelemetry(long apiHandle, long nativeHandle, boolean sendTelemetry) throws OrtException;

    private static native long createTensor(long apiHandle, long allocatorHandle, Object data, long[] shape, int onnxType) throws OrtException;
    private static native long createTensorFromBuffer(long apiHandle, long allocatorHandle, Buffer data, long bufferSize, long[] shape, int onnxType) throws OrtException;

    private static native long createString(long apiHandle, long allocatorHandle, String data) throws OrtException;
    private static native long createStringTensor(long apiHandle, long allocatorHandle, Object[] data, long[] shape) throws OrtException;

}
