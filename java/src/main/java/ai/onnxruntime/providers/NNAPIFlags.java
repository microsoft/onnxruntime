/*
 * Copyright (c) 2021, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

/** Flags for the NNAPI provider. */
public enum NNAPIFlags implements OrtFlags {
  /** Enables fp16 support. */
  USE_FP16(1), // NNAPI_FLAG_USE_FP16(0x001)
  /**
   * Uses channels first format. Only recommended for developer usage to validate code changes to
   * the execution provider implementation.
   */
  USE_NCHW(2), // NNAPI_FLAG_USE_NCHW(0x002)
  /**
   * Disables NNAPI from using CPU. If an operator could be assigned to NNAPI, but NNAPI only has a
   * CPU implementation of that operator on the current device, model load will fail.
   */
  CPU_DISABLED(4), // NNAPI_FLAG_CPU_DISABLED(0x004)
  /**
   * NNAPI will only use CPU. Only recommended for developer usage as it significantly impacts
   * performance.
   */
  CPU_ONLY(8); // NNAPI_FLAG_CPU_ONLY(0x008)

  /** The native value of the enum. */
  public final int value;

  NNAPIFlags(int value) {
    this.value = value;
  }

  @Override
  public int getValue() {
    return value;
  }
}
