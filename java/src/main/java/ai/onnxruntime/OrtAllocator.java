/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/** An ONNX Runtime memory allocator. */
class OrtAllocator extends NativeObject {

  static final OrtAllocator DEFAULT_ALLOCATOR;

  static {
    try {
      DEFAULT_ALLOCATOR = new OrtAllocator(getDefaultAllocator(OnnxRuntime.ortApiHandle), true);
    } catch (OrtException e) {
      throw new RuntimeException("Failed to create OrtEnvironment defaults", e);
    }
  }

  private final boolean isDefault;

  /**
   * Constructs an OrtAllocator wrapped around a native reference.
   *
   * @param handle The reference to a native OrtAllocator.
   * @param isDefault Is this the default allocator.
   */
  OrtAllocator(long handle, boolean isDefault) {
    super(handle);
    this.isDefault = isDefault;
  }

  /**
   * Is this the default allocator?
   *
   * @return True if it is the default allocator.
   */
  public boolean isDefault() {
    return isDefault;
  }

  /** Frees all resources. Only non-default allocators can be closed. */
  @Override
  public void close() {
    if (!isDefault) {
      super.close();
    }
  }

  @Override
  protected void doClose(long handle) {
    closeAllocator(OnnxRuntime.ortApiHandle, handle);
  }

  /**
   * Gets a reference to the default allocator.
   *
   * @param apiHandle The API handle to use.
   * @return A pointer to the default allocator.
   * @throws OrtException If it failed to get the allocator.
   */
  private static native long getDefaultAllocator(long apiHandle) throws OrtException;

  private native void closeAllocator(long apiHandle, long nativeHandle);
}
