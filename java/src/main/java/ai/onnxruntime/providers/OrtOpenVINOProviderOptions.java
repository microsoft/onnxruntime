/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

/** Options for configuring the OpenVINO provider. */
public final class OrtOpenVINOProviderOptions {

  public final OpenVINODeviceType deviceType;
  public final boolean enableVPUFastCompile;
  public final String deviceID;
  public final int numOfThreads;

  public OrtOpenVINOProviderOptions(
      OpenVINODeviceType deviceType,
      boolean enableVPUFastCompile,
      String deviceID,
      int numOfThreads) {
    this.deviceType = deviceType;
    this.enableVPUFastCompile = enableVPUFastCompile;
    this.deviceID = deviceID;
    this.numOfThreads = numOfThreads;
  }

  /** Type of OpenVINO device. */
  public enum OpenVINODeviceType {
    CPU_FP32("CPU_FP32"),
    GPU_FP32("GPU_FP32"),
    GPU_FP16("GPU_FP16"),
    MYRIAD_FP16("MYRIAD_FP16"),
    VAD_M_FP16("VAD-M_FP16"),
    VAD_F_FP32("VAD-F_FP32");

    public final String value;

    OpenVINODeviceType(String value) {
      this.value = value;
    }
  }
}
