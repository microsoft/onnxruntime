/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

/**
 * Describes an {@link ONNXMap} object or output node.
 */
public class MapInfo implements ValueInfo {

    /**
     * The number of entries in this map.
     */
    public final int size;

    /**
     * The Java type of the keys.
     */
    public final ONNXJavaType keyType;

    /**
     * The Java type of the values.
     */
    public final ONNXJavaType valueType;

    MapInfo(ONNXJavaType keyType, ONNXJavaType valueType) {
        this.size = -1;
        this.keyType = keyType;
        this.valueType = valueType;
    }

    MapInfo(int size, ONNXJavaType keyType, ONNXJavaType valueType) {
        this.size = size;
        this.keyType = keyType;
        this.valueType = valueType;
    }

    @Override
    public String toString() {
        String initial = size == -1 ? "MapInfo(size=UNKNOWN" : "MapInfo(size="+size;
        return initial + ",keyType=" + keyType.toString() + ",valueType=" + valueType.toString() + ")";
    }
}
