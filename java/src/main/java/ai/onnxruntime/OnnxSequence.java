/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A sequence of {@link OnnxValue}s all of the same type.
 *
 * <p>Supports the types mentioned in "onnxruntime_c_api.h", currently String, Long, Float, Double,
 * Map&gt;String,Float&lt;, Map&gt;Long,Float&lt;.
 */
public class OnnxSequence extends NativeObject implements OnnxValue {

  private final OrtAllocator allocator;

  private final SequenceInfo info;

  /**
   * Creates the wrapper object for a native sequence.
   *
   * <p>Called from native code.
   *
   * @param nativeHandle The reference to the native sequence object.
   * @param allocator The allocator used when extracting data from this object.
   * @param info The sequence type information.
   */
  OnnxSequence(long nativeHandle, OrtAllocator allocator, SequenceInfo info) {
    super(nativeHandle);
    this.allocator = allocator;
    this.info = info;
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_SEQUENCE;
  }

  /**
   * Extracts a Java object from the native ONNX type.
   *
   * <p>Returns either a {@link List} of boxed primitives, {@link String}s, or {@link
   * java.util.Map}s.
   *
   * @return A Java object containing the value.
   * @throws OrtException If the runtime failed to read an element.
   */
  @Override
  public List<Object> getValue() throws OrtException {
    try (NativeUsage sequenceReference = use();
        NativeUsage allocatorReference = allocator.use()) {
      if (info.sequenceOfMaps) {
        List<Object> outputSequence = new ArrayList<>(info.length);
        for (int i = 0; i < info.length; i++) {
          Object[] keys = getMapKeys(i);
          Object[] values = getMapValues(i);
          HashMap<Object, Object> map = new HashMap<>(keys.length);
          for (int j = 0; j < keys.length; j++) {
            map.put(keys[j], values[j]);
          }
          outputSequence.add(map);
        }
        return outputSequence;
      } else {
        switch (info.sequenceType) {
          case FLOAT:
            float[] floats =
                getFloats(
                    OnnxRuntime.ortApiHandle,
                    sequenceReference.handle(),
                    allocatorReference.handle());
            ArrayList<Object> boxed = new ArrayList<>(floats.length);
            for (float aFloat : floats) {
              // box float to Float
              boxed.add(aFloat);
            }
            return boxed;
          case DOUBLE:
            return Arrays.stream(
                    getDoubles(
                        OnnxRuntime.ortApiHandle,
                        sequenceReference.handle(),
                        allocatorReference.handle()))
                .boxed()
                .collect(Collectors.toList());
          case INT64:
            return Arrays.stream(
                    getLongs(
                        OnnxRuntime.ortApiHandle,
                        sequenceReference.handle(),
                        allocatorReference.handle()))
                .boxed()
                .collect(Collectors.toList());
          case STRING:
            String[] strings =
                getStrings(
                    OnnxRuntime.ortApiHandle,
                    sequenceReference.handle(),
                    allocatorReference.handle());
            ArrayList<Object> list = new ArrayList<>(strings.length);
            list.addAll(Arrays.asList(strings));
            return list;
          case BOOL:
          case INT8:
          case INT16:
          case INT32:
          case UNKNOWN:
          default:
            throw new OrtException("Unsupported type in a sequence, found " + info.sequenceType);
        }
      }
    }
  }

  @Override
  public SequenceInfo getInfo() {
    return info;
  }

  @Override
  public String toString() {
    return super.toString() + "(info=" + info.toString() + ")";
  }

  /** Closes this sequence, releasing the native memory backing it and it's elements. */
  @Override
  protected void doClose(long handle) {
    close(OnnxRuntime.ortApiHandle, handle);
  }

  /**
   * Extract the keys for the map at the specified index.
   *
   * @param index The index to extract.
   * @return The map keys as an array.
   * @throws OrtException If the native code failed to read the keys.
   */
  private Object[] getMapKeys(int index) throws OrtException {
    try (NativeUsage sequenceReference = use();
        NativeUsage allocatorReference = allocator.use()) {
      if (info.mapInfo.keyType == OnnxJavaType.STRING) {
        return getStringKeys(
            OnnxRuntime.ortApiHandle,
            sequenceReference.handle(),
            allocatorReference.handle(),
            index);
      } else {
        return Arrays.stream(
                getLongKeys(
                    OnnxRuntime.ortApiHandle,
                    sequenceReference.handle(),
                    allocatorReference.handle(),
                    index))
            .boxed()
            .toArray();
      }
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
    try (NativeUsage sequenceReference = use();
        NativeUsage allocatorReference = allocator.use()) {
      switch (info.mapInfo.valueType) {
        case STRING:
          {
            return getStringValues(
                OnnxRuntime.ortApiHandle,
                sequenceReference.handle(),
                allocatorReference.handle(),
                index);
          }
        case INT64:
          {
            return Arrays.stream(
                    getLongValues(
                        OnnxRuntime.ortApiHandle,
                        sequenceReference.handle(),
                        allocatorReference.handle(),
                        index))
                .boxed()
                .toArray();
          }
        case FLOAT:
          {
            float[] floats =
                getFloatValues(
                    OnnxRuntime.ortApiHandle,
                    sequenceReference.handle(),
                    allocatorReference.handle(),
                    index);
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
                    getDoubleValues(
                        OnnxRuntime.ortApiHandle,
                        sequenceReference.handle(),
                        allocatorReference.handle(),
                        index))
                .boxed()
                .toArray();
          }
        default:
          throw new RuntimeException("Invalid or unknown valueType: " + info.mapInfo.valueType);
      }
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

  private native String[] getStrings(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native long[] getLongs(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native float[] getFloats(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native double[] getDoubles(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native void close(long apiHandle, long nativeHandle);
}
