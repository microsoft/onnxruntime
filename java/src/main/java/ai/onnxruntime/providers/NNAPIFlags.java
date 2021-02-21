/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

/** Flags for the NNAPI provider. */
public enum NNAPIFlags implements OrtFlags {
  USE_FP16(1), // NNAPI_FLAG_USE_FP16(0x001)
  USE_NCHW(2), // NNAPI_FLAG_USE_NCHW(0x002)
  CPU_DISABLED(4); // NNAPI_FLAG_CPU_DISABLED(0x004)

  public final int value;

  NNAPIFlags(int value) {
    this.value = value;
  }

  @Override
  public int getValue() {
    return value;
  }
}
