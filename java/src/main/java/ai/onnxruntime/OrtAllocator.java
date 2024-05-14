/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;

/** An ONNX Runtime memory allocator. */
class OrtAllocator implements AutoCloseable {

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  /** The native pointer. */
  final long handle;

  private final boolean isDefault;

  private boolean closed = false;

  /**
   * Constructs an OrtAllocator wrapped around a native reference.
   *
   * @param handle The reference to a native OrtAllocator.
   * @param isDefault Is this the default allocator.
   */
  OrtAllocator(long handle, boolean isDefault) {
    this.handle = handle;
    this.isDefault = isDefault;
  }

  /**
   * Is this allocator closed?
   *
   * @return True if the allocator is closed.
   */
  public boolean isClosed() {
    return closed;
  }

  /**
   * Is this the default allocator?
   *
   * @return True if it is the default allocator.
   */
  public boolean isDefault() {
    return isDefault;
  }

  /**
   * Closes the allocator, must be done after all its child objects have been closed.
   *
   * <p>The default allocator is not closeable, and this operation is a no-op on that allocator.
   *
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
