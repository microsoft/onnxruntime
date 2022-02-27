/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/**
 * An abstract base class for execution provider options classes.
 */
// Note this lives in ai.onnxruntime to allow subclasses to access the OnnxRuntime.ortApiHandle package private field.
public abstract class OrtProviderOptions implements AutoCloseable {

    protected final long nativeHandle;

    /**
     * Constructs a OrtProviderOptions wrapped around a native pointer.
     * @param nativeHandle The native pointer.
     */
    protected OrtProviderOptions(long nativeHandle) {
        this.nativeHandle = nativeHandle;
    }

    /**
     * Allow access to the api handle pointer for subclasses.
     * @return The api handle.
     */
    protected static long getApiHandle() {
        return OnnxRuntime.ortApiHandle;
    }

    @Override
    public void close() {
        close(OnnxRuntime.ortApiHandle, nativeHandle);
    }

    /**
     * Native close method.
     * @param apiHandle The api pointer.
     * @param nativePointer The native options pointer.
     */
    protected abstract void close(long apiHandle, long nativePointer);
}
