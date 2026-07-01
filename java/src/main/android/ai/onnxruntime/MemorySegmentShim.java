/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Wrapper for java.lang.foreign.MemorySegment instances which throws {@link java.lang.UnsupportedOperationException}
 * as FFM is not supported on Android.
 */
final class MemorySegmentShim {
    private static final Logger logger = Logger.getLogger(MemorySegmentShim.class.getName());

    /**
     * Constructor which wraps a MemorySegment. Always throws on Android.
     *
     * @param segment The memory segment.
     * @throws UnsupportedOperationException If java.lang.foreign.MemorySegment is not available in the running JDK.
     */
    MemorySegmentShim(Object segment) {
        throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }

    /**
     * Constructor which builds a MemorySegment using the supplied arguments. Always throws on Android.
     *
     * @param address The address of the memory.
     * @param byteSize The size of the memory.
     * @throws UnsupportedOperationException If java.lang.foreign.MemorySegment is not available in the running JDK.
     */
    MemorySegmentShim(long address, long byteSize) {
        throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }

    /**
     * Always throws {@link UnsupportedOperationException} on Android.
     * @return The MemorySegment.
     */
    Object get() {
        throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }

    /**
     * Always throws {@link UnsupportedOperationException} on Android.
     * @return The address of the MemorySegment.
     */
    long address() {
        throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }

    /**
     * Always throws {@link UnsupportedOperationException} on Android.
     * @return The size of the MemorySegment in bytes.
     */
    long byteSize() {
        throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }

}
