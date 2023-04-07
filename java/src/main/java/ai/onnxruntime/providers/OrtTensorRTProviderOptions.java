/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;

/**
 * Options for configuring the TensorRT execution provider.
 *
 * <p>Supported options are listed on the <a
 * href="https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options">ORT
 * website</a>.
 */
public final class OrtTensorRTProviderOptions extends StringConfigProviderOptions {
  private static final OrtProvider PROVIDER = OrtProvider.TENSOR_RT;

  /**
   * Constructs TensorRT execution provider options for device 0.
   *
   * @throws OrtException If TensorRT is unavailable.
   */
  public OrtTensorRTProviderOptions() throws OrtException {
    this(0);
  }

  /**
   * Constructs TensorRT execution provider options for the specified non-negative device id.
   *
   * @param deviceId The device id.
   * @throws OrtException If TensorRT is unavailable.
   */
  public OrtTensorRTProviderOptions(int deviceId) throws OrtException {
    super(loadLibraryAndCreate(PROVIDER, () -> create(getApiHandle())));
    if (deviceId < 0) {
      close();
      throw new IllegalArgumentException("Device id must be non-negative, received " + deviceId);
    }

    String id = "" + deviceId;
    this.options.put("device_id", id);
    add(getApiHandle(), this.nativeHandle, "device_id", id);
  }

  @Override
  public OrtProvider getProvider() {
    return PROVIDER;
  }

  /**
   * Creates a native OrtTensorRTProviderOptionsV2.
   *
   * @param apiHandle The ONNX Runtime api handle.
   * @return A pointer to a new OrtTensorRTProviderOptionsV2.
   * @throws OrtException If the creation failed (usually due to TensorRT being unavailable).
   */
  private static native long create(long apiHandle) throws OrtException;

  /**
   * Adds an option to this options instance.
   *
   * @param apiHandle The api pointer.
   * @param nativeHandle The native options pointer.
   * @param key The option key.
   * @param value The option value.
   * @throws OrtException If the addition failed.
   */
  @Override
  protected native void add(long apiHandle, long nativeHandle, String key, String value)
      throws OrtException;

  /**
   * Closes this options instance.
   *
   * @param apiHandle The api pointer.
   * @param nativeHandle The native options pointer.
   */
  @Override
  protected native void close(long apiHandle, long nativeHandle);
}
