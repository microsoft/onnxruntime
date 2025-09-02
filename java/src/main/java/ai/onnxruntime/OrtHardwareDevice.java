/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.Map;
import java.util.logging.Logger;

/** Hardware information for a specific device. */
public final class OrtHardwareDevice {

  /** The hardware device types. */
  // Must be updated in concert with the native OrtHardwareDeviceType enum in the C API
  public enum OrtHardwareDeviceType {
    /** A CPU device. */
    CPU(0),
    /** A GPU device. */
    GPU(1),
    /** A NPU (Neural Processing Unit) device. */
    NPU(2);
    private final int value;

    private static final Logger logger = Logger.getLogger(OrtHardwareDeviceType.class.getName());
    private static final OrtHardwareDeviceType[] values = new OrtHardwareDeviceType[3];

    static {
      for (OrtHardwareDeviceType ot : OrtHardwareDeviceType.values()) {
        values[ot.value] = ot;
      }
    }

    OrtHardwareDeviceType(int value) {
      this.value = value;
    }

    /**
     * Gets the native value associated with this device type.
     *
     * @return The native value.
     */
    public int getValue() {
      return value;
    }

    /**
     * Maps from the C API's int enum to the Java enum.
     *
     * @param deviceType The index of the Java enum.
     * @return The Java enum.
     */
    public static OrtHardwareDeviceType mapFromInt(int deviceType) {
      if ((deviceType >= 0) && (deviceType < values.length)) {
        return values[deviceType];
      } else {
        logger.warning("Unknown device type '" + deviceType + "' setting to CPU");
        return CPU;
      }
    }
  }

  private final long nativeHandle;

  private final OrtHardwareDeviceType type;
  private final int vendorId;
  private final String vendor;
  private final int deviceId;
  private final Map<String, String> metadata;

  OrtHardwareDevice(long nativeHandle) {
    this.nativeHandle = nativeHandle;
    this.type =
        OrtHardwareDeviceType.mapFromInt(getDeviceType(OnnxRuntime.ortApiHandle, nativeHandle));
    this.vendorId = getVendorId(OnnxRuntime.ortApiHandle, nativeHandle);
    this.vendor = getVendor(OnnxRuntime.ortApiHandle, nativeHandle);
    this.deviceId = getDeviceId(OnnxRuntime.ortApiHandle, nativeHandle);
    String[][] metadata = getMetadata(OnnxRuntime.ortApiHandle, nativeHandle);
    this.metadata = OrtUtil.convertToMap(metadata);
  }

  long getNativeHandle() {
    return nativeHandle;
  }

  /**
   * Gets the device type.
   *
   * @return The device type.
   */
  public OrtHardwareDeviceType getType() {
    return type;
  }

  /**
   * Gets the vendor ID number.
   *
   * @return The vendor ID number.
   */
  public int getVendorId() {
    return vendorId;
  }

  /**
   * Gets the device ID number.
   *
   * @return The device ID number.
   */
  public int getDeviceId() {
    return deviceId;
  }

  /**
   * Gets an unmodifiable view on the device metadata.
   *
   * @return The device metadata.
   */
  public Map<String, String> getMetadata() {
    return metadata;
  }

  /**
   * Gets the vendor name.
   *
   * @return The vendor name.
   */
  public String getVendor() {
    return vendor;
  }

  @Override
  public String toString() {
    return "OrtHardwareDevice{"
        + "type="
        + type
        + ", vendorId="
        + vendorId
        + ", vendor='"
        + vendor
        + '\''
        + ", deviceId="
        + deviceId
        + ", metadata="
        + metadata
        + '}';
  }

  private static native String getVendor(long apiHandle, long nativeHandle);

  private static native String[][] getMetadata(long apiHandle, long nativeHandle);

  private static native int getDeviceType(long apiHandle, long nativeHandle);

  private static native int getDeviceId(long apiHandle, long nativeHandle);

  private static native int getVendorId(long apiHandle, long nativeHandle);
}
