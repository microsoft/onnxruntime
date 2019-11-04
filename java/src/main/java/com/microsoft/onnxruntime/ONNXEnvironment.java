/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

import com.microsoft.onnxruntime.ONNXSession.SessionOptions;

import java.io.IOException;

/**
 * The host object for an ONNX system. Can create {@link ONNXSession}s
 * which encapsulate specific models.
 */
public class ONNXEnvironment implements AutoCloseable {

    static {
        try {
            ONNX.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ONNX library",e);
        }
    }

    final long nativeHandle;

    /**
     * Create an ONNXEnvironment using a default name.
     * @throws ONNXException If the environment couldn't be created.
     */
    public ONNXEnvironment() throws ONNXException {
        this("java-default");
    }

    /**
     * Create an ONNXEnvironment using the specified name and default log level of {@link LoggingLevel#ORT_LOGGING_LEVEL_WARNING}.
     * @param name The environment name.
     * @throws ONNXException If the environment couldn't be created.
     */
    public ONNXEnvironment(String name) throws ONNXException {
        this(LoggingLevel.ORT_LOGGING_LEVEL_WARNING,name);
    }

    /**
     * Create an ONNXEnvironment using the specified name and log level.
     * @param loggingLevel The logging level to use.
     * @param name The environment name.
     * @throws ONNXException If the environment couldn't be created.
     */
    public ONNXEnvironment(LoggingLevel loggingLevel, String name) throws ONNXException {
        nativeHandle = createHandle(ONNX.ortApiHandle,loggingLevel.getValue(),name);
    }

    /**
     * Create a session using the specified {@link SessionOptions} and model.
     * @param modelPath Path on disk to load the model from.
     * @param allocator The memory allocator to use.
     * @param options The session options.
     * @return An ONNXSession with the specified model.
     * @throws ONNXException If the model failed to load, wasn't compatible or caused an error.
     */
    public ONNXSession createSession(String modelPath, ONNXAllocator allocator, SessionOptions options) throws ONNXException {
        return new ONNXSession(this,modelPath,allocator,options);
    }

    /**
     * Create a session using the specified {@link SessionOptions} and model.
     * @param modelArray Byte array representing an ONNX model.
     * @param allocator The memory allocator to use.
     * @param options The session options.
     * @return An ONNXSession with the specified model.
     * @throws ONNXException If the model failed to parse, wasn't compatible or caused an error.
     */
    public ONNXSession createSession(byte[] modelArray, ONNXAllocator allocator, SessionOptions options) throws ONNXException {
        return new ONNXSession(this,modelArray,allocator,options);
    }

    /**
     * Closes the ONNXEnvironment freeing it's resources.
     * @throws ONNXException If the close failed.
     */
    @Override
    public void close() throws ONNXException {
        close(ONNX.ortApiHandle,nativeHandle);
    }

    /**
     * Creates the native object.
     * @param apiHandle The API pointer.
     * @param loggingLevel The logging level.
     * @param name The name of the environment.
     * @return The pointer to the native object.
     * @throws ONNXException If the creation failed.
     */
    private native long createHandle(long apiHandle, int loggingLevel, String name) throws ONNXException;

    /**
     * Closes the ONNX environment, frees the handle.
     * @param apiHandle The API pointer.
     * @param nativeHandle The handle to free.
     * @throws ONNXException If an error was caused by freeing the handle.
     */
    private native void close(long apiHandle, long nativeHandle) throws ONNXException;

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
