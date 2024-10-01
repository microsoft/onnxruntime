/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;

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
   * Creates an instance of OrtLoraAdapter from a byte array.
   *
   * @param loraArray The LoRA stored in a byte array.
   * @throws OrtException If the native call failed.
   * @return An OrtLoraAdapter instance.
   */
  public static OrtLoraAdapter create(byte[] loraArray) throws OrtException {
    return create(loraArray, null);
  }

  /**
   * Creates an instance of OrtLoraAdapter from a byte array.
   *
   * @param loraArray The LoRA stored in a byte array.
   * @param allocator optional allocator or null. If supplied, adapter parameters are copied to the
   *     allocator memory.
   * @throws OrtException If the native call failed.
   * @return An OrtLoraAdapter instance.
   */
  static OrtLoraAdapter create(byte[] loraArray, OrtAllocator allocator) throws OrtException {
    Objects.requireNonNull(loraArray, "LoRA array must not be null");
    long allocatorHandle = allocator == null ? 0 : allocator.handle;
    return new OrtLoraAdapter(
        createLoraAdapterFromArray(OnnxRuntime.ortApiHandle, loraArray, allocatorHandle));
  }

  /**
   * Creates an instance of OrtLoraAdapter from a direct ByteBuffer.
   *
   * @param loraBuffer The buffer to load.
   * @throws OrtException If the native call failed.
   * @return An OrtLoraAdapter instance.
   */
  public static OrtLoraAdapter create(ByteBuffer loraBuffer) throws OrtException {
    return create(loraBuffer, null);
  }

  /**
   * Creates an instance of OrtLoraAdapter from a direct ByteBuffer.
   *
   * @param loraBuffer The buffer to load.
   * @param allocator optional allocator or null. If supplied, adapter parameters are copied to the
   *     allocator memory.
   * @throws OrtException If the native call failed.
   * @return An OrtLoraAdapter instance.
   */
  static OrtLoraAdapter create(ByteBuffer loraBuffer, OrtAllocator allocator) throws OrtException {
    Objects.requireNonNull(loraBuffer, "LoRA buffer must not be null");
    if (loraBuffer.remaining() == 0) {
      throw new OrtException("Invalid LoRA buffer, no elements remaining.");
    } else if (!loraBuffer.isDirect()) {
      throw new OrtException("ByteBuffer is not direct.");
    }
    long allocatorHandle = allocator == null ? 0 : allocator.handle;
    return new OrtLoraAdapter(
        createLoraAdapterFromBuffer(
            OnnxRuntime.ortApiHandle,
            loraBuffer,
            loraBuffer.position(),
            loraBuffer.remaining(),
            allocatorHandle));
  }

  /**
   * Creates an instance of OrtLoraAdapter.
   *
   * @param adapterPath path to the adapter file that is going to be memory mapped.
   * @throws OrtException If the native call failed.
   * @return An OrtLoraAdapter instance.
   */
  public static OrtLoraAdapter create(String adapterPath) throws OrtException {
    return create(adapterPath, null);
  }

  /**
   * Creates an instance of OrtLoraAdapter.
   *
   * @param adapterPath path to the adapter file that is going to be memory mapped.
   * @param allocator optional allocator or null. If supplied, adapter parameters are copied to the
   *     allocator memory.
   * @throws OrtException If the native call failed.
   * @return An OrtLoraAdapter instance.
   */
  static OrtLoraAdapter create(String adapterPath, OrtAllocator allocator) throws OrtException {
    long allocatorHandle = allocator == null ? 0 : allocator.handle;
    return new OrtLoraAdapter(
        createLoraAdapter(OnnxRuntime.ortApiHandle, adapterPath, allocatorHandle));
  }

  /**
   * Package accessor for native pointer.
   *
   * @return The native pointer.
   */
  long getNativeHandle() {
    return nativeHandle;
  }

  /** Checks if the OrtLoraAdapter is closed, if so throws {@link IllegalStateException}. */
  void checkClosed() {
    if (closed) {
      throw new IllegalStateException("Trying to use a closed OrtLoraAdapter");
    }
  }

  @Override
  public void close() {
    if (!closed) {
      close(OnnxRuntime.ortApiHandle, nativeHandle);
      closed = true;
    } else {
      throw new IllegalStateException("Trying to close an already closed OrtLoraAdapter");
    }
  }

  private static native long createLoraAdapter(
      long apiHandle, String adapterPath, long allocatorHandle) throws OrtException;

  private static native long createLoraAdapterFromArray(
      long apiHandle, byte[] loraBytes, long allocatorHandle) throws OrtException;

  private static native long createLoraAdapterFromBuffer(
      long apiHandle, ByteBuffer loraBuffer, int bufferPos, int bufferSize, long allocatorHandle)
      throws OrtException;

  private static native void close(long apiHandle, long nativeHandle);
}
