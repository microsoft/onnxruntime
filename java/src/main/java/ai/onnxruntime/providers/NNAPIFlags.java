/*
 * Copyright (c) 2021, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

/** Flags for the NNAPI provider. */
public enum NNAPIFlags implements OrtFlags {
  /** Enables fp16 support. */
  USE_FP16(1), // NNAPI_FLAG_USE_FP16(0x001)
  /** Uses channels first format. */
  USE_NCHW(2), // NNAPI_FLAG_USE_NCHW(0x002)
  /** Disables CPU ops. */
  CPU_DISABLED(4), // NNAPI_FLAG_CPU_DISABLED(0x004)
  /** Only runs on CPU. */
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
