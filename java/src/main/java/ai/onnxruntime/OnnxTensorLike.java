/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;

/**
 * Currently implemented by {@link OnnxTensor}, {@link OnnxSparseTensor}. Will be sealed to these
 * types one day.
 */
public abstract class OnnxTensorLike implements OnnxValue {
  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  protected final long nativeHandle;

  protected final long allocatorHandle;

  protected final TensorInfo info;

  OnnxTensorLike(long nativeHandle, long allocatorHandle, TensorInfo info) {
    this.nativeHandle = nativeHandle;
    this.allocatorHandle = allocatorHandle;
    this.info = info;
  }

  /**
   * Returns the native pointer.
   *
   * @return The native pointer.
   */
  long getNativeHandle() {
    return nativeHandle;
  }
}
