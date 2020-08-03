/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A container for a map returned by {@link OrtSession#run(Map)}.
 *
 * <p>Supported types are those mentioned in "onnxruntime_c_api.h", keys: String and Long, values:
 * String, Long, Float, Double.
 */
public class OnnxMap extends NativeObject implements OnnxValue {

  /** An enum representing the Java type of the values stored in an {@link OnnxMap}. */
  public enum OnnxMapValueType {
    INVALID(0),
    STRING(1),
    LONG(2),
    FLOAT(3),
    DOUBLE(4);

    final int value;

    OnnxMapValueType(int value) {
      this.value = value;
    }

    private static final OnnxMapValueType[] values = new OnnxMapValueType[5];

    static {
      for (OnnxMapValueType ot : OnnxMapValueType.values()) {
        values[ot.value] = ot;
      }
    }

    /**
     * Gets the enum type from it's integer id. Used by native code.
     *
     * @param value The integer id.
     * @return The enum instance.
     */
    public static OnnxMapValueType mapFromInt(int value) {
      if ((value > 0) && (value < values.length)) {
        return values[value];
      } else {
        return INVALID;
      }
    }

    /**
     * Maps a {@link OnnxJavaType} into a map value type. If it's not a valid map type return {@link
     * OnnxMapValueType#INVALID}.
     *
     * @param type The Java type.
     * @return The equivalent Map value type.
     */
    public static OnnxMapValueType mapFromOnnxJavaType(OnnxJavaType type) {
      switch (type) {
        case FLOAT:
          return OnnxMapValueType.FLOAT;
        case DOUBLE:
          return OnnxMapValueType.DOUBLE;
        case INT64:
          return OnnxMapValueType.LONG;
        case STRING:
          return OnnxMapValueType.STRING;
        case INT8:
        case INT16:
        case INT32:
        case BOOL:
        case UNKNOWN:
        default:
          return OnnxMapValueType.INVALID;
      }
    }
  }

  private final OrtAllocator allocator;

  private final MapInfo info;

  private final boolean stringKeys;

  private final OnnxMapValueType valueType;

  /**
   * Constructs an OnnxMap containing a reference to the native map along with the type information.
   *
   * <p>Called from native code.
   *
   * @param nativeHandle The reference to the native map object.
   * @param allocator The allocator used when extracting data from this object.
   * @param info The type information.
   */
  OnnxMap(long nativeHandle, OrtAllocator allocator, MapInfo info) {
    super(nativeHandle);
    this.allocator = allocator;
    this.info = info;
    this.stringKeys = info.keyType == OnnxJavaType.STRING;
    this.valueType = OnnxMapValueType.mapFromOnnxJavaType(info.valueType);
  }

  /**
   * The number of entries in the map.
   *
   * @return The number of entries.
   */
  public int size() {
    return info.size;
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_MAP;
  }

  /**
   * Returns a weakly typed Map containing all the elements.
   *
   * @return A map.
   * @throws OrtException If the onnx runtime failed to read the entries.
   */
  @Override
  public Map<Object, Object> getValue() throws OrtException {
    Object[] keys = getMapKeys();
    Object[] values = getMapValues();
    HashMap<Object, Object> map = new HashMap<>(keys.length);
    for (int i = 0; i < keys.length; i++) {
      map.put(keys[i], values[i]);
    }
    return map;
  }

  /**
   * Extracts the map keys, boxing the primitives as necessary.
   *
   * @return The keys from the map as an array.
   * @throws OrtException If the onnxruntime failed to read the keys.
   */
  private Object[] getMapKeys() throws OrtException {
    try (NativeUsage mapReference = use();
        NativeUsage allocatorReference = allocator.use()) {
      if (stringKeys) {
        return getStringKeys(
            OnnxRuntime.ortApiHandle, mapReference.handle(), allocatorReference.handle());
      } else {
        return Arrays.stream(
                getLongKeys(
                    OnnxRuntime.ortApiHandle, mapReference.handle(), allocatorReference.handle()))
            .boxed()
            .toArray();
      }
    }
  }

  /**
   * Extracts the map values, boxing primitives as necessary.
   *
   * @return The values from the map as an array.
   * @throws OrtException If the onnxruntime failed to read the values.
   */
  private Object[] getMapValues() throws OrtException {
    try (NativeUsage mapReference = use();
        NativeUsage allocatorReference = allocator.use()) {
      switch (valueType) {
        case STRING:
          {
            return getStringValues(
                OnnxRuntime.ortApiHandle, mapReference.handle(), allocatorReference.handle());
          }
        case LONG:
          {
            return Arrays.stream(
                    getLongValues(
                        OnnxRuntime.ortApiHandle,
                        mapReference.handle(),
                        allocatorReference.handle()))
                .boxed()
                .toArray();
          }
        case FLOAT:
          {
            float[] floats =
                getFloatValues(
                    OnnxRuntime.ortApiHandle, mapReference.handle(), allocatorReference.handle());
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
                        mapReference.handle(),
                        allocatorReference.handle()))
                .boxed()
                .toArray();
          }
        default:
          throw new RuntimeException("Invalid or unknown valueType: " + valueType);
      }
    }
  }

  @Override
  public MapInfo getInfo() {
    return info;
  }

  @Override
  public String toString() {
    return super.toString() + "(size=" + size() + ",info=" + info.toString() + ")";
  }

  /** Closes this map, releasing the native memory backing it and it's elements. */
  @Override
  protected void doClose(long handle) {
    close(OnnxRuntime.ortApiHandle, handle);
  }

  private native String[] getStringKeys(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native long[] getLongKeys(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native String[] getStringValues(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native long[] getLongValues(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native float[] getFloatValues(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native double[] getDoubleValues(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native void close(long apiHandle, long nativeHandle);
}
