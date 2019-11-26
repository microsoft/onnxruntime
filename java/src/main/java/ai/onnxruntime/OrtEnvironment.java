/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;

import java.io.IOException;

/**
 * The host object for the onnx-runtime system. Can create {@link OrtSession}s
 * which encapsulate specific models.
 */
public class OrtEnvironment implements AutoCloseable {

    static {
        try {
            OnnxRuntime.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load onnx-runtime library",e);
        }
    }

    final long nativeHandle;

    /**
     * Create an OrtEnvironment using a default name.
     * @throws OrtException If the environment couldn't be created.
     */
    public OrtEnvironment() throws OrtException {
        this("java-default");
    }

    /**
     * Create an OrtEnvironment using the specified name and default log level of {@link LoggingLevel#ORT_LOGGING_LEVEL_WARNING}.
     * @param name The environment name.
     * @throws OrtException If the environment couldn't be created.
     */
    public OrtEnvironment(String name) throws OrtException {
        this(LoggingLevel.ORT_LOGGING_LEVEL_WARNING,name);
    }

    /**
     * Create an OrtEnvironment using the specified name and log level.
     * @param loggingLevel The logging level to use.
     * @param name The environment name.
     * @throws OrtException If the environment couldn't be created.
     */
    public OrtEnvironment(LoggingLevel loggingLevel, String name) throws OrtException {
        nativeHandle = createHandle(OnnxRuntime.ortApiHandle,loggingLevel.getValue(),name);
    }

    /**
     * Create a session using the specified {@link SessionOptions} and model.
     * @param modelPath Path on disk to load the model from.
     * @param allocator The memory allocator to use.
     * @param options The session options.
     * @return An {@link OrtSession} with the specified model.
     * @throws OrtException If the model failed to load, wasn't compatible or caused an error.
     */
    public OrtSession createSession(String modelPath, OrtAllocator allocator, SessionOptions options) throws OrtException {
        return new OrtSession(this,modelPath,allocator,options);
    }

    /**
     * Create a session using the specified {@link SessionOptions} and model.
     * @param modelArray Byte array representing an ONNX model.
     * @param allocator The memory allocator to use.
     * @param options The session options.
     * @return An {@link OrtSession} with the specified model.
     * @throws OrtException If the model failed to parse, wasn't compatible or caused an error.
     */
    public OrtSession createSession(byte[] modelArray, OrtAllocator allocator, SessionOptions options) throws OrtException {
        return new OrtSession(this,modelArray,allocator,options);
    }

    /**
     * Closes the OrtEnvironment freeing it's resources.
     * @throws OrtException If the close failed.
     */
    @Override
    public void close() throws OrtException {
        close(OnnxRuntime.ortApiHandle,nativeHandle);
    }

    /**
     * Creates the native object.
     * @param apiHandle The API pointer.
     * @param loggingLevel The logging level.
     * @param name The name of the environment.
     * @return The pointer to the native object.
     * @throws OrtException If the creation failed.
     */
    private native long createHandle(long apiHandle, int loggingLevel, String name) throws OrtException;

    /**
     * Closes the OrtEnvironment, frees the handle.
     * @param apiHandle The API pointer.
     * @param nativeHandle The handle to free.
     * @throws OrtException If an error was caused by freeing the handle.
     */
    private native void close(long apiHandle, long nativeHandle) throws OrtException;

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
}
