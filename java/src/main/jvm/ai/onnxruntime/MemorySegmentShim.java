/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.logging.Logger;

/**
 * Wrapper for java.lang.foreign.MemorySegment instances which uses reflection to access the methods
 * so it can be compiled on Java 21 and earlier. Requires Java 22 or newer to use MemorySegments,
 * when run on earlier versions all methods throw {@link UnsupportedOperationException}.
 */
final class MemorySegmentShim {
  private static final Logger logger = Logger.getLogger(MemorySegmentShim.class.getName());

  // Class is null if java.lang.foreign.MemorySegment is not available.
  private static final Class<?> memorySegmentClass;

  /*
   * Method handles that bind to methods on java.lang.foreign.MemorySegment.
   */
  private static final MethodHandle ofAddress;
  private static final MethodHandle reinterpret;
  private static final MethodHandle address;
  private static final MethodHandle byteSize;
  private static final MethodHandle isNative;
  private static final MethodHandle set; // only used in tests
  private static final Object floatLayout;

  static {
    Class<?> segmentClass;
    MethodHandle tmpOfAddress;
    MethodHandle tmpReinterpret;
    MethodHandle tmpAddress;
    MethodHandle tmpByteSize;
    MethodHandle tmpIsNative;
    MethodHandle tmpSet;
    Object tmpLayout;
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      segmentClass = Class.forName("java.lang.foreign.MemorySegment");
      Class<?> valueLayoutClass = Class.forName("java.lang.foreign.ValueLayout");
      Class<?> floatValueLayoutClass = Class.forName("java.lang.foreign.ValueLayout$OfFloat");
      // Attempt to lookup the Java 22 memory segment methods.
      tmpOfAddress =
          lookup.findStatic(
              segmentClass, "ofAddress", MethodType.methodType(segmentClass, long.class));
      tmpReinterpret =
          lookup.findVirtual(
              segmentClass, "reinterpret", MethodType.methodType(segmentClass, long.class));
      tmpAddress = lookup.findVirtual(segmentClass, "address", MethodType.methodType(long.class));
      tmpByteSize = lookup.findVirtual(segmentClass, "byteSize", MethodType.methodType(long.class));
      tmpIsNative =
          lookup.findVirtual(segmentClass, "isNative", MethodType.methodType(boolean.class));
      tmpSet =
          lookup.findVirtual(
              segmentClass,
              "set",
              MethodType.methodType(void.class, floatValueLayoutClass, long.class, float.class));
      tmpLayout =
          lookup.findStaticGetter(valueLayoutClass, "JAVA_FLOAT", floatValueLayoutClass).invoke();
    } catch (IllegalAccessException
        | NoSuchMethodException
        | ClassNotFoundException
        | NoSuchFieldException e) {
      logger.fine("Running on Java 21 or earlier, MemorySegment not available");
      segmentClass = null;
      tmpOfAddress = null;
      tmpReinterpret = null;
      tmpAddress = null;
      tmpByteSize = null;
      tmpIsNative = null;
      tmpSet = null;
      tmpLayout = null;
    } catch (Throwable e) {
      logger.severe(
          "Failed to load float value layout, while other Java 22 features were available.");
      segmentClass = null;
      tmpOfAddress = null;
      tmpReinterpret = null;
      tmpAddress = null;
      tmpByteSize = null;
      tmpIsNative = null;
      tmpSet = null;
      tmpLayout = null;
    }
    memorySegmentClass = segmentClass;
    ofAddress = tmpOfAddress;
    reinterpret = tmpReinterpret;
    address = tmpAddress;
    byteSize = tmpByteSize;
    isNative = tmpIsNative;
    set = tmpSet;
    floatLayout = tmpLayout;
  }

  // Only holds java.lang.foreign.MemorySegment instances
  private final Object segment;

  /**
   * Constructor which wraps a MemorySegment.
   *
   * @param segment The memory segment.
   * @throws IllegalArgumentException If the supplied argument was not a
   *     java.lang.foreign.MemorySegment.
   * @throws UnsupportedOperationException If java.lang.foreign.MemorySegment is not available in
   *     the running JDK.
   */
  MemorySegmentShim(Object segment) {
    if (memorySegmentClass != null) {
      if (memorySegmentClass.isInstance(segment)) {
        this.segment = segment;
      } else {
        throw new IllegalArgumentException(
            "Segment argument was not a java.lang.foreign.MemorySegment, found "
                + segment.getClass());
      }
    } else {
      throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }
  }

  /**
   * Constructor which builds a MemorySegment using the supplied arguments.
   *
   * @param address The address of the memory.
   * @param byteSize The size of the memory.
   * @throws IllegalArgumentException If the supplied argument was not a valid memory region (i.e.,
   *     positive address and non-negative size).
   * @throws UnsupportedOperationException If java.lang.foreign.MemorySegment is not available in
   *     the running JDK.
   */
  MemorySegmentShim(long address, long byteSize) {
    if (memorySegmentClass != null) {
      if (address > 0 && byteSize >= 0) {
        try {
          Object segment = ofAddress.invoke(address);
          segment = reinterpret.invoke(segment, byteSize);
          this.segment = segment;
        } catch (Throwable e) {
          throw new AssertionError("Should not reach here", e);
        }
      } else {
        throw new IllegalArgumentException(
            "Invalid segment, found a non-positive address or a negative size, address = "
                + address
                + ", byteSize = "
                + byteSize);
      }
    } else {
      throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }
  }

  /**
   * Returns the MemorySegment instance.
   *
   * @return The MemorySegment.
   */
  Object get() {
    return segment;
  }

  /**
   * Returns the address of the MemorySegment.
   *
   * @return The address of the MemorySegment.
   */
  long address() {
    if (memorySegmentClass != null) {
      try {
        long ret = (long) address.invoke(segment);
        return ret;
      } catch (Throwable e) {
        throw new AssertionError("Should not reach here", e);
      }
    } else {
      throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }
  }

  /**
   * Returns the size of the MemorySegment in bytes.
   *
   * @return The size of the MemorySegment in bytes.
   */
  long byteSize() {
    if (memorySegmentClass != null) {
      try {
        long ret = (long) byteSize.invoke(segment);
        return ret;
      } catch (Throwable e) {
        throw new AssertionError("Should not reach here", e);
      }
    } else {
      throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }
  }

  /**
   * Returns true if this segment is backed by native memory, and false if it's backed by memory on
   * the Java heap.
   *
   * @return True if the segment is native.
   */
  boolean isNative() {
    if (memorySegmentClass != null) {
      try {
        boolean ret = (boolean) isNative.invoke(segment);
        return ret;
      } catch (Throwable e) {
        throw new AssertionError("Should not reach here", e);
      }
    } else {
      throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }
  }

  /**
   * Sets a float value on this memory segment at the specified index.
   *
   * <p>Only used in the tests, should not be used in user code as invoke is slower than
   * invokeExact.
   *
   * @param idx The index to write to.
   * @param value The value.
   */
  void set(long idx, float value) {
    if (memorySegmentClass != null) {
      try {
        set.invoke(segment, floatLayout, idx, value);
      } catch (Throwable e) {
        throw new AssertionError("Should not reach here", e);
      }
    } else {
      throw new UnsupportedOperationException("java.lang.foreign.MemorySegment is not available.");
    }
  }
}
