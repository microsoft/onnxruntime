/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 */
package com.microsoft.onnxruntime;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A sequence of ONNXValue all of the same type.
 *
 * Supports the types mentioned in "onnxruntime_c_api.h",
 * currently String, Long, Float, Double, Map&gt;String,Float&lt;, Map&gt;Long,Float&lt;.
 */
public class ONNXSequence implements ONNXValue {

    static {
        try {
            ONNX.init();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ONNX library",e);
        }
    }

    final long nativeHandle;

    private final long allocatorHandle;

    private final SequenceInfo info;

    ONNXSequence(long nativeHandle, long allocatorHandle, SequenceInfo info) {
        this.nativeHandle = nativeHandle;
        this.allocatorHandle = allocatorHandle;
        this.info = info;
    }

    @Override
    public ONNXValueType getType() {
        return ONNXValueType.ONNX_TYPE_SEQUENCE;
    }

    /**
     * Extracts a Java object from the native ONNX type.
     *
     * Returns either a {@link List} of primitives, {@link String}s, or {@link java.util.Map}s.
     * @return A Java object containing the value.
     * @throws ONNXException If the runtime failed to read an element.
     */
    @Override
    public List<Object> getValue() throws ONNXException {
        if (info.sequenceOfMaps) {
            List<Object> outputSequence = new ArrayList<>();
            for (int i = 0; i < info.length; i++) {
                HashMap<Object,Object> map = new HashMap<>();
                Object[] keys = getMapKeys(i);
                Object[] values = getMapValues(i);
                for (int j = 0; j < keys.length; j++) {
                    map.put(keys[j],values[j]);
                }
                outputSequence.add(map);
            }
            return outputSequence;
        } else {
            switch (info.sequenceType) {
                case FLOAT:
                    float[] floats = getFloats(ONNX.ortApiHandle,nativeHandle,allocatorHandle);
                    ArrayList<Object> boxed = new ArrayList<>(floats.length);
                    for (int i = 0; i < floats.length; i++) {
                        // cast float to Float
                        boxed.add(floats[i]);
                    }
                    return boxed;
                case DOUBLE:
                    return Arrays.stream(getDoubles(ONNX.ortApiHandle,nativeHandle,allocatorHandle)).boxed().collect(Collectors.toList());
                case INT64:
                    return Arrays.stream(getLongs(ONNX.ortApiHandle,nativeHandle,allocatorHandle)).boxed().collect(Collectors.toList());
                case STRING:
                    String[] strings = getStrings(ONNX.ortApiHandle,nativeHandle,allocatorHandle);
                    ArrayList<Object> list = new ArrayList<>(strings.length);
                    list.addAll(Arrays.asList(strings));
                    return list;
                case BOOL:
                case INT8:
                case INT16:
                case INT32:
                case UNKNOWN:
                default:
                    throw new ONNXException("Unsupported type in a sequence, found " + info.sequenceType);
            }
        }
    }

    @Override
    public SequenceInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "ONNXSequence(info="+info.toString()+")";
    }

    /**
     * Closes this sequence, releasing the native memory backing it and it's elements.
     */
    @Override
    public void close() {
        close(ONNX.ortApiHandle,nativeHandle);
    }

    private Object[] getMapKeys(int index) throws ONNXException {
        if (info.mapInfo.keyType == ONNXJavaType.STRING) {
            return getStringKeys(ONNX.ortApiHandle,nativeHandle,allocatorHandle,index);
        } else {
            return Arrays.stream(getLongKeys(ONNX.ortApiHandle,nativeHandle,allocatorHandle,index)).boxed().toArray();
        }
    }

    private Object[] getMapValues(int index) throws ONNXException {
        switch (info.mapInfo.valueType) {
            case STRING: {
                return getStringValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle,index);
            }
            case INT64:{
                return Arrays.stream(getLongValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle,index)).boxed().toArray();
            }
            case FLOAT:{
                float[] floats = getFloatValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle,index);
                Float[] boxed = new Float[floats.length];
                for (int i = 0; i < floats.length; i++) {
                    // cast float to Float
                    boxed[i] = floats[i];
                }
                return boxed;
            }
            case DOUBLE:{
                return Arrays.stream(getDoubleValues(ONNX.ortApiHandle,nativeHandle,allocatorHandle,index)).boxed().toArray();
            }
            default:
                throw new RuntimeException("Invalid or unknown valueType: " + info.mapInfo.valueType);
        }
    }


    private native String[] getStringKeys(long apiHandle, long nativeHandle,long allocatorHandle, int index) throws ONNXException;
    private native long[] getLongKeys(long apiHandle, long nativeHandle,long allocatorHandle, int index) throws ONNXException;

    private native String[] getStringValues(long apiHandle, long nativeHandle,long allocatorHandle, int index) throws ONNXException;
    private native long[] getLongValues(long apiHandle, long nativeHandle,long allocatorHandle, int index) throws ONNXException;
    private native float[] getFloatValues(long apiHandle, long nativeHandle,long allocatorHandle, int index) throws ONNXException;
    private native double[] getDoubleValues(long apiHandle, long nativeHandle,long allocatorHandle, int index) throws ONNXException;

    private native String[] getStrings(long apiHandle, long nativeHandle,long allocatorHandle) throws ONNXException;
    private native long[] getLongs(long apiHandle, long nativeHandle,long allocatorHandle) throws ONNXException;
    private native float[] getFloats(long apiHandle, long nativeHandle,long allocatorHandle) throws ONNXException;
    private native double[] getDoubles(long apiHandle, long nativeHandle,long allocatorHandle) throws ONNXException;

    private native void close(long apiHandle, long nativeHandle);
}
