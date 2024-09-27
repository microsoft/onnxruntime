/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;

/**
 * A container for an adapter which can be supplied to {@link
 * OrtSession.RunOptions#addActiveLoraAdapter(OrtLoraAdapter)} to apply the adapter to a specific
 * execution of a model.
 */
public final class OrtLoraAdapter implements AutoCloseable {
  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  private final long nativeHandle;

  private boolean closed = false;

  private OrtLoraAdapter(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Creates an instance of OrtLoraAdapter.
   *
   * @param absoluteAdapterPath path to the adapter file that is going to be memory mapped.
   * @throws OrtException If the native call failed.
   */
  public static OrtLoraAdapter create(String absoluteAdapterPath) throws OrtException {
    return create(absoluteAdapterPath, null);
  }

  /**
   * Creates an instance of OrtLoraAdapter.
   *
   * @param absoluteAdapterPath path to the adapter file that is going to be memory mapped.
   * @param allocator optional allocator or null. If supplied, adapter parameters are copied to the
   *     allocator memory.
   * @throws OrtException If the native call failed.
   */
  static OrtLoraAdapter create(String absoluteAdapterPath, OrtAllocator allocator)
      throws OrtException {
    long allocatorHandle = allocator == null ? 0 : allocator.handle;
    return new OrtLoraAdapter(
        createLoraAdapter(OnnxRuntime.ortApiHandle, absoluteAdapterPath, allocatorHandle));
  }

  /**
   * Package accessor for native pointer.
   *
   * @return The native pointer.
   */
  long getNativeHandle() {
    return nativeHandle;
  }

  /** Checks if the LoraAdapter is closed, if so throws {@link IllegalStateException}. */
  void checkClosed() {
    if (closed) {
      throw new IllegalStateException("Trying to use a closed LoraAdapter");
    }
  }

  @Override
  public void close() {
    if (!closed) {
      close(OnnxRuntime.ortApiHandle, nativeHandle);
      closed = true;
    } else {
      throw new IllegalStateException("Trying to close an already closed LoraAdapter");
    }
  }

  private static native long createLoraAdapter(
      long apiHandle, String adapterPath, long allocatorHandle) throws OrtException;

  private static native void close(long apiHandle, long nativeHandle);
}
