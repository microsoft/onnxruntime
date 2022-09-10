/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.TensorInfo.OnnxTensorType;

/** Describes an {@link OnnxMap} object or output node. */
public class MapInfo implements ValueInfo {

  /** The number of entries in this map. */
  public final int size;

  /** The Java type of the keys. */
  public final OnnxJavaType keyType;

  /** The Java type of the values. */
  public final OnnxJavaType valueType;

  /**
   * Construct a MapInfo with the specified key type and value type. The size is unknown and set to
   * -1.
   *
   * @param keyType The Java type of the keys.
   * @param valueType The Java type of the values.
   */
  MapInfo(OnnxJavaType keyType, OnnxJavaType valueType) {
    this.size = -1;
    this.keyType = keyType;
    this.valueType = valueType;
  }

  /**
   * Construct a MapInfo with the specified size, key type and value type.
   *
   * @param size The size.
   * @param keyType The Java type of the keys.
   * @param valueType The Java type of the values.
   */
  MapInfo(int size, OnnxJavaType keyType, OnnxJavaType valueType) {
    this.size = size;
    this.keyType = keyType;
    this.valueType = valueType;
  }

  /**
   * Construct a MapInfo with the specified size, key type and value type.
   *
   * <p>Called from JNI.
   *
   * @param size The size.
   * @param keyTypeInt The int representing the {@link OnnxTensorType} of the keys.
   * @param valueTypeInt The int representing the {@link OnnxTensorType} of the values.
   */
  MapInfo(int size, int keyTypeInt, int valueTypeInt) {
    this.size = size;
    this.keyType = OnnxJavaType.mapFromOnnxTensorType(OnnxTensorType.mapFromInt(keyTypeInt));
    this.valueType = OnnxJavaType.mapFromOnnxTensorType(OnnxTensorType.mapFromInt(valueTypeInt));
  }

  @Override
  public String toString() {
    String initial = size == -1 ? "MapInfo(size=UNKNOWN" : "MapInfo(size=" + size;
    return initial + ",keyType=" + keyType.toString() + ",valueType=" + valueType.toString() + ")";
  }
}
