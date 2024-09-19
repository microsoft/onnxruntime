/*
 * Copyright (c) 2021, 2024, Oracle and/or its affiliates. All rights reserved.
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
  ONLY_ENABLE_DEVICE_WITH_ANE(4), // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE(0x004)
  /**
   * Only allow CoreML EP to take nodes with inputs with static shapes. By default it will also
   * allow inputs with dynamic shapes. However, the performance may be negatively impacted if inputs
   * have dynamic shapes.
   */
  ONLY_ALLOW_STATIC_INPUT_SHAPES(8), // COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES(0x008)
  /**
   * Create an MLProgram. By default it will create a NeuralNetwork model. Requires Core ML 5 or
   * later.
   */
  CREATE_MLPROGRAM(16); // COREML_FLAG_CREATE_MLPROGRAM(0x010)

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
