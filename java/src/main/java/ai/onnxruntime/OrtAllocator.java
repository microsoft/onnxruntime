/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;

/**
 * An ONNX Runtime memory allocator.
 */
class OrtAllocator implements AutoCloseable {

    final long handle;

    private final boolean isDefault;

    private boolean closed = false;

    static {
        try {
            OnnxRuntime.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load onnx-runtime library",e);
        }
    }

    OrtAllocator(long handle, boolean isDefault) {
        this.handle = handle;
        this.isDefault = isDefault;
    }

    public boolean isClosed() {
        return closed;
    }

    /**
     * Closes the allocator, must be done after all it's child objects have been closed.
     * @throws OrtException If it failed to close.
     */
    @Override
    public void close() throws OrtException {
        if (!closed) {
            if (!isDefault) {
                // Can only close custom allocators.
                closeAllocator(OnnxRuntime.ortApiHandle, handle);
                closed = true;
            }
        } else {
            throw new IllegalStateException("Trying to close an already closed OrtAllocator.");
        }
    }

    // The default allocator cannot be closed, this is guarded in the close method above.
    private native void closeAllocator(long apiHandle, long nativeHandle) throws OrtException;
}