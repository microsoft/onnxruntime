/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.Map;

/** A tuple of Execution Provider information and the hardware device. */
public final class OrtEpDevice {

  private final long nativeHandle;

  private final String epName;
  private final String epVendor;
  private final Map<String, String> epMetadata;
  private final Map<String, String> epOptions;
  private final OrtHardwareDevice device;

  /**
   * Construct an OrtEpDevice tuple from the native pointer.
   *
   * @param nativeHandle The native pointer.
   */
  OrtEpDevice(long nativeHandle) {
    this.nativeHandle = nativeHandle;
    this.epName = getEpName(OnnxRuntime.ortApiHandle, nativeHandle);
    this.epVendor = getEpVendor(OnnxRuntime.ortApiHandle, nativeHandle);
    String[][] metadata = getEpMetadata(OnnxRuntime.ortApiHandle, nativeHandle);
    this.epMetadata = OrtUtil.convertToMap(metadata);
    String[][] options = getEpOptions(OnnxRuntime.ortApiHandle, nativeHandle);
    this.epOptions = OrtUtil.convertToMap(options);
    this.device = new OrtHardwareDevice(getDeviceHandle(OnnxRuntime.ortApiHandle, nativeHandle));
  }

  /**
   * Return the native pointer.
   *
   * @return The native pointer.
   */
  long getNativeHandle() {
    return nativeHandle;
  }

  /**
   * Gets the Execution Provider name.
   *
   * @return The EP name.
   */
  public String getEpName() {
    return epName;
  }

  /**
   * Gets the Execution Provider vendor name.
   *
   * @return The EP vendor name.
   */
  public String getEpVendor() {
    return epVendor;
  }

  /**
   * Gets an unmodifiable view on the Execution Provider metadata.
   *
   * @return The EP metadata.
   */
  public Map<String, String> getEpMetadata() {
    return epMetadata;
  }

  /**
   * Gets an unmodifiable view on the Execution Provider options.
   *
   * @return The EP options.
   */
  public Map<String, String> getEpOptions() {
    return epOptions;
  }

  /**
   * Gets the device information.
   *
   * @return The device information.
   */
  public OrtHardwareDevice getDevice() {
    return device;
  }

  @Override
  public String toString() {
    return "OrtEpDevice{"
        + "epName='"
        + epName
        + '\''
        + ", epVendor='"
        + epVendor
        + '\''
        + ", epMetadata="
        + epMetadata
        + ", epOptions="
        + epOptions
        + ", device="
        + device
        + '}';
  }

  private static native String getEpName(long apiHandle, long nativeHandle);

  private static native String getEpVendor(long apiHandle, long nativeHandle);

  private static native String[][] getEpMetadata(long apiHandle, long nativeHandle);

  private static native String[][] getEpOptions(long apiHandle, long nativeHandle);

  private static native long getDeviceHandle(long apiHandle, long nativeHandle);
}
