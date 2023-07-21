/*
 * Copyright (c) 2021, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

/** Flags for the CoreML provider. */
public enum CoreMLFlags implements OrtFlags {
  /**
   * Use only the CPU, disables the GPU and Apple Neural Engine. Only recommended for developer
   * usage as it significantly impacts performance.
   */
  CPU_ONLY(1), // COREML_FLAG_USE_CPU_ONLY(0x001)
  /** Enables CoreML on subgraphs. */
  ENABLE_ON_SUBGRAPH(2), // COREML_FLAG_ENABLE_ON_SUBGRAPH(0x002)
  /** Only enable usage of CoreML if the device has an Apple Neural Engine. */
  ONLY_ENABLE_DEVICE_WITH_ANE(4); // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE(0x004),

  /** The native value of the enum. */
  public final int value;

  CoreMLFlags(int value) {
    this.value = value;
  }

  @Override
  public int getValue() {
    return value;
  }
}
