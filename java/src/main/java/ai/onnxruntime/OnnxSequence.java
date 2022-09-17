/*
 * Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

/**
 * A sequence of {@link OnnxValue}s all of the same type.
 *
 * <p>Supports the types mentioned in "onnxruntime_c_api.h", currently
 *
 * <ul>
 *   <li>OnnxTensor&lt;String&gt;
 *   <li>OnnxTensor&lt;Long&gt;
 *   <li>OnnxTensor&lt;Float&gt;
 *   <li>OnnxTensor&lt;Double&gt;
 *   <li>OnnxMap&lt;String,Float&gt;
 *   <li>OnnxMap&lt;Long,Float&gt;
 * </ul>
 */
public class OnnxSequence implements OnnxValue {

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  final long nativeHandle;

  private final long allocatorHandle;

  private final SequenceInfo info;

  /**
   * Creates the wrapper object for a native sequence.
   *
   * <p>Called from native code.
   *
   * @param nativeHandle The reference to the native sequence object.
   * @param allocatorHandle The reference to the allocator.
   * @param info The sequence type information.
   */
  OnnxSequence(long nativeHandle, long allocatorHandle, SequenceInfo info) {
    this.nativeHandle = nativeHandle;
    this.allocatorHandle = allocatorHandle;
    this.info = info;
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_SEQUENCE;
  }

  /**
   * Extracts a Java object from the native ONNX type.
   *
   * <p>Returns either a {@link List} of either {@link OnnxTensor} or {@link java.util.Map}.
   *
   * @return A Java object containing the value.
   * @throws OrtException If the runtime failed to read an element.
   */
  @Override
  public List<Object> getValue() throws OrtException {
    if (info.sequenceOfMaps) {
      List<Object> outputSequence = new ArrayList<>(info.length);
      for (int i = 0; i < info.length; i++) {
        Object[] keys = getMapKeys(i);
        Object[] values = getMapValues(i);
        HashMap<Object, Object> map = new HashMap<>(OrtUtil.capacityFromSize(keys.length));
        for (int j = 0; j < keys.length; j++) {
          map.put(keys[j], values[j]);
        }
        outputSequence.add(map);
      }
      return Collections.unmodifiableList(outputSequence);
    } else {
      switch (info.sequenceType) {
        case STRING:
        case INT64:
        case FLOAT:
        case DOUBLE:
          Object[] tensors = getTensors(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle);
          return Collections.unmodifiableList(Arrays.asList(tensors));
        case BOOL:
        case UINT8:
        case INT8:
        case INT16:
        case INT32:
        case UNKNOWN:
        default:
          throw new OrtException("Unsupported type in a sequence, found " + info.sequenceType);
      }
    }
  }

  @Override
  public SequenceInfo getInfo() {
    return info;
  }

  @Override
  public String toString() {
    return "OnnxSequence(info=" + info.toString() + ")";
  }

  /** Closes this sequence, releasing the native memory backing it and it's elements. */
  @Override
  public void close() {
    close(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Extract the keys for the map at the specified index.
   *
   * @param index The index to extract.
   * @return The map keys as an array.
   * @throws OrtException If the native code failed to read the keys.
   */
  private Object[] getMapKeys(int index) throws OrtException {
    if (info.mapInfo.keyType == OnnxJavaType.STRING) {
      return getStringKeys(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle, index);
    } else {
      return Arrays.stream(
              getLongKeys(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle, index))
          .boxed()
          .toArray();
    }
  }

  /**
   * Extract the values for the map at the specified index.
   *
   * @param index The index to extract.
   * @return The map values as an array.
   * @throws OrtException If the native code failed to read the values.
   */
  private Object[] getMapValues(int index) throws OrtException {
    switch (info.mapInfo.valueType) {
      case STRING:
        {
          return getStringValues(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle, index);
        }
      case INT64:
        {
          return Arrays.stream(
                  getLongValues(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle, index))
              .boxed()
              .toArray();
        }
      case FLOAT:
        {
          float[] floats =
              getFloatValues(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle, index);
          Float[] boxed = new Float[floats.length];
          for (int i = 0; i < floats.length; i++) {
            // cast float to Float
            boxed[i] = floats[i];
          }
          return boxed;
        }
      case DOUBLE:
        {
          return Arrays.stream(
                  getDoubleValues(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle, index))
              .boxed()
              .toArray();
        }
      default:
        throw new RuntimeException("Invalid or unknown valueType: " + info.mapInfo.valueType);
    }
  }

  private native String[] getStringKeys(
      long apiHandle, long nativeHandle, long allocatorHandle, int index) throws OrtException;

  private native long[] getLongKeys(
      long apiHandle, long nativeHandle, long allocatorHandle, int index) throws OrtException;

  private native String[] getStringValues(
      long apiHandle, long nativeHandle, long allocatorHandle, int index) throws OrtException;

  private native long[] getLongValues(
      long apiHandle, long nativeHandle, long allocatorHandle, int index) throws OrtException;

  private native float[] getFloatValues(
      long apiHandle, long nativeHandle, long allocatorHandle, int index) throws OrtException;

  private native double[] getDoubleValues(
      long apiHandle, long nativeHandle, long allocatorHandle, int index) throws OrtException;

  private native OnnxTensor[] getTensors(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native void close(long apiHandle, long nativeHandle);
}
