/*
 * Copyright (c) 2022, 2023, Oracle and/or its affiliates. All rights reserved.
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

  /** The native pointer. */
  protected final long nativeHandle;

  /** The pointer to the native memory allocator. */
  protected final long allocatorHandle;

  /** The size and shape information for this tensor. */
  protected final TensorInfo info;

  /**
   * Constructs a tensor-like (the base class of OnnxTensor and OnnxSparseTensor).
   *
   * @param nativeHandle The pointer to the tensor.
   * @param allocatorHandle The pointer to the memory allocator.
   * @param info The tensor info.
   */
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

  /**
   * Returns a {@link TensorInfo} for this tensor.
   *
   * @return The tensor info.
   */
  @Override
  public TensorInfo getInfo() {
    return info;
  }
}
