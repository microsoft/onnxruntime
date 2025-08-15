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
    this.epName = getName(OnnxRuntime.ortApiHandle, nativeHandle);
    this.epVendor = getVendor(OnnxRuntime.ortApiHandle, nativeHandle);
    String[][] metadata = getMetadata(OnnxRuntime.ortApiHandle, nativeHandle);
    this.epMetadata = OrtUtil.convertToMap(metadata);
    String[][] options = getOptions(OnnxRuntime.ortApiHandle, nativeHandle);
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
   * Gets the EP name.
   *
   * @return The EP name.
   */
  public String getName() {
    return epName;
  }

  /**
   * Gets the vendor name.
   *
   * @return The vendor name.
   */
  public String getVendor() {
    return epVendor;
  }

  /**
   * Gets an unmodifiable view on the EP metadata.
   *
   * @return The EP metadata.
   */
  public Map<String, String> getMetadata() {
    return epMetadata;
  }

  /**
   * Gets an unmodifiable view on the EP options.
   *
   * @return The EP options.
   */
  public Map<String, String> getOptions() {
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

  private static native String getName(long apiHandle, long nativeHandle);

  private static native String getVendor(long apiHandle, long nativeHandle);

  private static native String[][] getMetadata(long apiHandle, long nativeHandle);

  private static native String[][] getOptions(long apiHandle, long nativeHandle);

  private static native long getDeviceHandle(long apiHandle, long nativeHandle);
}
