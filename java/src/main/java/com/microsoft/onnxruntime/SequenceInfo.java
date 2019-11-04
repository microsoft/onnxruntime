/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

/**
 * Describes a sequence, including it's element type.
 */
public class SequenceInfo implements ValueInfo {

    /**
     * Is this a sequence of maps.
     */
    public final boolean sequenceOfMaps;

    /**
     * The type of the sequence if it does not contain a map, {@link ONNXJavaType#UNKNOWN} if it does.
     */
    public final ONNXJavaType sequenceType;

    /**
     * The type of the map if it contains a map, null otherwise.
     */
    public final MapInfo mapInfo;

    /**
     * The number of elements in this sequence.
     */
    public final int length;

    SequenceInfo(int length, ONNXJavaType sequenceType) {
        this.length = length;
        this.sequenceType = sequenceType;
        this.sequenceOfMaps = false;
        this.mapInfo = null;
    }

    SequenceInfo(int length, MapInfo mapInfo) {
        this.length = length;
        this.sequenceOfMaps = true;
        this.mapInfo = mapInfo;
        this.sequenceType = ONNXJavaType.UNKNOWN;
    }

    SequenceInfo(int length, ONNXJavaType keyType, ONNXJavaType valueType) {
        this.length = length;
        this.sequenceType = ONNXJavaType.UNKNOWN;
        this.sequenceOfMaps = true;
        this.mapInfo = new MapInfo(keyType,valueType);
    }

    public boolean isSequenceOfMaps() {
        return sequenceOfMaps;
    }

    @Override
    public String toString() {
        String initial = "SequenceInfo(length=" + (length == -1 ? "UNKNOWN" : length);
        if (sequenceOfMaps) {
            return initial+",type=" + mapInfo.toString() + ")";
        } else {
            return initial+",type=" + sequenceType.toString() + ")";
        }
    }

}
