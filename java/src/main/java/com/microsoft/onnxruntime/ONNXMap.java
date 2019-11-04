/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A container for a map returned by an ONNX call.
 * <p>
 * Supported types are those mentioned in "onnxruntime_c_api.h",
 * keys: String and Long, values: String, Long, Float, Double.
 */
public class ONNXMap implements ONNXValue {

    static {
        try {
            ONNX.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ONNX library",e);
        }
    }

    /**
     * An enum representing the type of the values stored in an {@link ONNXMap}.
     */
    public enum ONNXMapValueType {
        INVALID(0), STRING(1), LONG(2), FLOAT(3), DOUBLE(4);
        final int value;
        ONNXMapValueType(int value) {
            this.value = value;
        }
        private static final ONNXMapValueType[] values = new ONNXMapValueType[4];
        static {
            for (ONNXMapValueType ot : ONNXMapValueType.values()) {
                values[ot.value] = ot;
            }
        }

        /**
         * Gets the enum type from it's integer id. Used by native code.
         * @param value The integer id.
         * @return The enum instance.
         */
        public static ONNXMapValueType mapFromInt(int value) {
            if ((value > 0) && (value < values.length)) {
                return values[value];
            } else {
                return INVALID;
            }
        }

        /**
         * Maps a {@link ONNXJavaType} into a map value type. If it's not a valid
         * map type return {@link ONNXMapValueType#INVALID}.
         * @param type The Java type.
         * @return The equivalent Map value type.
         */
        public static ONNXMapValueType mapFromONNXJavaType(ONNXJavaType type) {
            switch (type) {
                case FLOAT:
                    return ONNXMapValueType.FLOAT;
                case DOUBLE:
                    return ONNXMapValueType.DOUBLE;
                case INT64:
                    return ONNXMapValueType.LONG;
                case STRING:
                    return ONNXMapValueType.STRING;
                case INT8:
                case INT16:
                case INT32:
                case BOOL:
                case UNKNOWN:
                default:
                    return ONNXMapValueType.INVALID;
            }
        }
    }

    final long nativeHandle;

    final long allocatorHandle;

    private final MapInfo info;

    private final boolean stringKeys;

    private final ONNXMapValueType valueType;

    ONNXMap(long nativeHandle, long allocatorHandle, MapInfo info) {
        this.nativeHandle = nativeHandle;
        this.allocatorHandle = allocatorHandle;
        this.info = info;
        this.stringKeys = info.keyType == ONNXJavaType.STRING;
        this.valueType = ONNXMapValueType.mapFromONNXJavaType(info.valueType);
    }

    /**
     * The number of entries in the map.
     * @return The number of entries.
     */
    public int size() {
        return info.size;
    }

    @Override
    public ONNXValueType getType() {
        return ONNXValueType.ONNX_TYPE_MAP;
    }

    /**
     * Returns a weakly typed Map containing all the elements.
     * @return A map.
     * @throws ONNXException If the onnx runtime failed to read the entries.
     */
    @Override
    public Map<Object,Object> getValue() throws ONNXException {
        HashMap<Object,Object> map = new HashMap<>();
        Object[] keys = getMapKeys();
        Object[] values = getMapValues();
        for (int i = 0; i < keys.length; i++) {
            map.put(keys[i],values[i]);
        }
        return map;
    }

    /**
     * Extracts the map keys, boxing the longs if necessary.
     * @return The keys from the map as an array.
     * @throws ONNXException If the onnx runtime failed to read the keys.
     */
    private Object[] getMapKeys() throws ONNXException {
        if (stringKeys) {
            return getStringKeys(ONNX.ortApiHandle,nativeHandle,allocatorHandle);
        } else {
            return Arrays.stream(getLongKeys(ONNX.ortApiHandle,nativeHandle,allocatorHandle)).boxed().toArray();
        }
    }

    /**
     * Extracts the map values, boxing primitives if necessary.
     * @return The values from the map as an array.
     * @throws ONNXException If the onnx runtime failed to read the values.
     */
    private Object[] getMapValues() throws ONNXException {
        switch (valueType) {
            case STRING: {
                return getStringValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle);
            }
            case LONG:{
                return Arrays.stream(getLongValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle)).boxed().toArray();
            }
            case FLOAT:{
                float[] floats = getFloatValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle);
                Float[] boxed = new Float[floats.length];
                for (int i = 0; i < floats.length; i++) {
                    // cast float to Float
                    boxed[i] = floats[i];
                }
                return boxed;
            }
            case DOUBLE:{
                return Arrays.stream(getDoubleValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle)).boxed().toArray();
            }
            default:
                throw new RuntimeException("Invalid or unknown valueType: " + valueType);
        }
    }

    @Override
    public MapInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "ONNXMap(size="+size()+",info="+info.toString()+")";
    }

    /**
     * Closes this map, releasing the native memory backing it and it's elements.
     */
    @Override
    public void close() {
        close(ONNX.ortApiHandle,nativeHandle);
    }

    private native String[] getStringKeys(long apiHandle,long nativeHandle,long allocatorHandle) throws ONNXException;
    private native long[] getLongKeys(long apiHandle,long nativeHandle,long allocatorHandle) throws ONNXException;

    private native String[] getStringValues(long apiHandle,long nativeHandle,long allocatorHandle) throws ONNXException;
    private native long[] getLongValues(long apiHandle,long nativeHandle,long allocatorHandle) throws ONNXException;
    private native float[] getFloatValues(long apiHandle,long nativeHandle,long allocatorHandle) throws ONNXException;
    private native double[] getDoubleValues(long apiHandle,long nativeHandle,long allocatorHandle) throws ONNXException;

    private native void close(long apiHandle,long nativeHandle);
}
