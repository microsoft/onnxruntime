/*
 * Copyright (c) 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/**
 * An optional ONNX value returned from a computation.
 *
 * <p>Check {@link #isPresent()} first, if the value is not present many of the methods throw {@link
 * IllegalStateException}.
 *
 * <p>The {@link #getValue()} method returns the appropriate Onnx type which can be further
 * inspected.
 */
public final class OnnxOptional implements OnnxValue {

  private final long nativeHandle;

  private final long allocatorHandle;

  private final boolean present;

  OnnxOptional(long nativeHandle, long allocatorHandle, boolean present) {
    this.nativeHandle = nativeHandle;
    this.allocatorHandle = allocatorHandle;
    this.present = present;
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_OPTIONAL;
  }

  @Override
  public OnnxValue getValue() throws OrtException {
    if (present) {
      OnnxValueType innerType = getInnerType(nativeHandle);
      switch (innerType) {
        case ONNX_TYPE_TENSOR:
          TensorInfo tensorInfo = getTensorInfo(nativeHandle);
          return new OnnxTensor(nativeHandle, allocatorHandle, tensorInfo);
        case ONNX_TYPE_SEQUENCE:
          SequenceInfo sequenceInfo = getSequenceInfo(nativeHandle);
          return new OnnxSequence(nativeHandle, allocatorHandle, sequenceInfo);
        case ONNX_TYPE_MAP:
          MapInfo mapInfo = getMapInfo(nativeHandle);
          return new OnnxMap(nativeHandle, allocatorHandle, mapInfo);
        case ONNX_TYPE_SPARSETENSOR:
          // TensorInfo sparseTensorInfo = getSparseTensorInfo(nativeHandle);
          // return new OnnxSparseTensor(nativeHandle,allocatorHandle,sparseTensorInfo);
        case ONNX_TYPE_UNKNOWN:
        case ONNX_TYPE_OPAQUE:
        case ONNX_TYPE_OPTIONAL:
        default:
          throw new IllegalStateException("Found unexpected ONNX value type " + innerType);
      }
    } else {
      throw new IllegalStateException("getValue called on empty OnnxOptional.");
    }
  }

  @Override
  public ValueInfo getInfo() {
    if (present) {
      OnnxValueType innerType = getInnerType(nativeHandle);
      switch (innerType) {
        case ONNX_TYPE_TENSOR:
          return getTensorInfo(nativeHandle);
        case ONNX_TYPE_SEQUENCE:
          return getSequenceInfo(nativeHandle);
        case ONNX_TYPE_MAP:
          return getMapInfo(nativeHandle);
        case ONNX_TYPE_SPARSETENSOR:
          return getSparseTensorInfo(nativeHandle);
        case ONNX_TYPE_UNKNOWN:
        case ONNX_TYPE_OPAQUE:
        case ONNX_TYPE_OPTIONAL:
        default:
          throw new IllegalStateException("Found unexpected ONNX value type " + innerType);
      }
    } else {
      throw new IllegalStateException("getInfo called on empty OnnxOptional.");
    }
  }

  @Override
  public void close() {
    close(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Is the value present in this optional?
   *
   * <p>When the value is not present many of the methods will return null.
   *
   * @return True if the value is present.
   */
  public boolean isPresent() {
    return present;
  }

  private native OnnxValueType getInnerType(long nativeHandle);

  private native TensorInfo getTensorInfo(long nativeHandle);

  private native SequenceInfo getSequenceInfo(long nativeHandle);

  private native MapInfo getMapInfo(long nativeHandle);

  private native TensorInfo getSparseTensorInfo(long nativeHandle);

  private native void close(long apiHandle, long nativeHandle);
}
