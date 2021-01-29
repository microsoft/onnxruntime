/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

/** Flags for the CoreML provider. */
public enum CoreMLFlags implements OrtFlags {
  CPU_ONLY(1), // COREML_FLAG_USE_CPU_ONLY(0x001)
  ENABLE_ON_SUBGRAPH(2); // COREML_FLAG_ENABLE_ON_SUBGRAPH(0x002)

  public final int value;

  CoreMLFlags(int value) {
    this.value = value;
  }

  @Override
  public int getValue() {
    return value;
  }
}
