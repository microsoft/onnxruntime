/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/** Describes an {@link OnnxSequence}, including it's element type if known. */
public class SequenceInfo implements ValueInfo {

  /** Is this a sequence of maps. */
  public final boolean sequenceOfMaps;

  /**
   * The type of the sequence if it does not contain a map, {@link OnnxJavaType#UNKNOWN} if it does.
   */
  public final OnnxJavaType sequenceType;

  /** The type of the map if it contains a map, null otherwise. */
  public final MapInfo mapInfo;

  /** The number of elements in this sequence. */
  public final int length;

  /**
   * Construct a sequence of known length, with the specified type. This sequence does not contain
   * maps.
   *
   * @param length The length of the sequence.
   * @param sequenceType The element type of the sequence.
   */
  SequenceInfo(int length, OnnxJavaType sequenceType) {
    this.length = length;
    this.sequenceType = sequenceType;
    this.sequenceOfMaps = false;
    this.mapInfo = null;
  }

  /**
   * Construct a sequence of known length containing maps.
   *
   * @param length The length of the sequence.
   * @param mapInfo The map type information.
   */
  SequenceInfo(int length, MapInfo mapInfo) {
    this.length = length;
    this.sequenceOfMaps = true;
    this.mapInfo = mapInfo;
    this.sequenceType = OnnxJavaType.UNKNOWN;
  }

  /**
   * Constructs a sequence of known length containing maps.
   *
   * @param length The length of the sequence.
   * @param keyType The map key type.
   * @param valueType The map value type.
   */
  SequenceInfo(int length, OnnxJavaType keyType, OnnxJavaType valueType) {
    this.length = length;
    this.sequenceType = OnnxJavaType.UNKNOWN;
    this.sequenceOfMaps = true;
    this.mapInfo = new MapInfo(keyType, valueType);
  }

  /**
   * Is this a sequence of maps?
   *
   * @return True if it's a sequence of maps.
   */
  public boolean isSequenceOfMaps() {
    return sequenceOfMaps;
  }

  @Override
  public String toString() {
    String initial = "SequenceInfo(length=" + (length == -1 ? "UNKNOWN" : length);
    if (sequenceOfMaps) {
      return initial + ",type=" + mapInfo.toString() + ")";
    } else {
      return initial + ",type=" + sequenceType.toString() + ")";
    }
  }
}
