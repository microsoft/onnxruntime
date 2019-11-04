/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package com.microsoft.onnxruntime;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

/**
 * Test helpers for manipulating primitive arrays.
 */
class TestHelpers {

    static boolean[] toPrimitiveBoolean(List<Boolean> input) {
        boolean[] output = new boolean[input.size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    static byte[] toPrimitiveByte(List<Byte> input) {
        byte[] output = new byte[input.size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    static short[] toPrimitiveShort(List<Short> input) {
        short[] output = new short[input.size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    static int[] toPrimitiveInteger(List<Integer> input) {
        int[] output = new int[input.size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    static long[] toPrimitiveLong(List<Long> input) {
        long[] output = new long[input.size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    static float[] toPrimitiveFloat(List<Float> input) {
        float[] output = new float[input.size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    static double[] toPrimitiveDouble(List<Double> input) {
        double[] output = new double[input.size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    static boolean[] flattenBoolean(Object o) {
        List<Boolean> output = new ArrayList<>();

        flatten((Object[]) o,output, boolean.class);

        return toPrimitiveBoolean(output);
    }

    static byte[] flattenByte(Object o) {
        List<Byte> output = new ArrayList<>();

        flatten((Object[]) o,output, byte.class);

        return toPrimitiveByte(output);
    }

    static short[] flattenShort(Object o) {
        List<Short> output = new ArrayList<>();

        flatten((Object[]) o,output, short.class);

        return toPrimitiveShort(output);
    }

    static int[] flattenInteger(Object o) {
        List<Integer> output = new ArrayList<>();

        flatten((Object[]) o,output, int.class);

        return toPrimitiveInteger(output);
    }

    static long[] flattenLong(Object o) {
        List<Long> output = new ArrayList<>();

        flatten((Object[]) o,output, long.class);

        return toPrimitiveLong(output);
    }

    static float[] flattenFloat(Object o) {
        List<Float> output = new ArrayList<>();

        flatten((Object[]) o,output, float.class);

        return toPrimitiveFloat(output);
    }

    static double[] flattenDouble(Object o) {
        List<Double> output = new ArrayList<>();

        flatten((Object[]) o,output, double.class);

        return toPrimitiveDouble(output);
    }

    static void flatten(Object[] input, List output, Class<?> primitiveClazz) {
        for (Object i : input) {
            Class<?> iClazz = i.getClass();
            if (iClazz.isArray()) {
                if (iClazz.getComponentType().isArray()) {
                    flatten((Object[]) i, output, primitiveClazz);
                } else if (iClazz.getComponentType().isPrimitive() && iClazz.getComponentType().equals(primitiveClazz)){
                    flattenBase(i,output,primitiveClazz);
                } else {
                    throw new IllegalStateException("Found a non-primitive, non-array element type, " + iClazz);
                }
            } else {
                throw new IllegalStateException("Found an element type where there should have been an array. Class = " + iClazz);
            }
        }
    }

    static void flattenBase(Object input, List output, Class<?> primitiveClass) {
        if (primitiveClass.equals(boolean.class)) {
            flattenBooleanBase((boolean[])input,output);
        } else if (primitiveClass.equals(byte.class)) {
            flattenByteBase((byte[])input,output);
        } else if (primitiveClass.equals(short.class)) {
            flattenShortBase((short[])input,output);
        } else if (primitiveClass.equals(int.class)) {
            flattenIntBase((int[])input,output);
        } else if (primitiveClass.equals(long.class)) {
            flattenLongBase((long[])input,output);
        } else if (primitiveClass.equals(float.class)) {
            flattenFloatBase((float[])input,output);
        } else if (primitiveClass.equals(double.class)) {
            flattenDoubleBase((double[])input,output);
        } else {
            throw new IllegalStateException("Flattening a non-primitive class");
        }
    }

    static void flattenBooleanBase(boolean[] input, List<Boolean> output) {
        for (int i = 0; i < input.length; i++) {
            output.add(input[i]);
        }
    }

    static void flattenByteBase(byte[] input, List<Byte> output) {
        for (int i = 0; i < input.length; i++) {
            output.add(input[i]);
        }
    }

    static void flattenShortBase(short[] input, List<Short> output) {
        for (int i = 0; i < input.length; i++) {
            output.add(input[i]);
        }
    }

    static void flattenIntBase(int[] input, List<Integer> output) {
        for (int i = 0; i < input.length; i++) {
            output.add(input[i]);
        }
    }

    static void flattenLongBase(long[] input, List<Long> output) {
        for (int i = 0; i < input.length; i++) {
            output.add(input[i]);
        }
    }

    static void flattenFloatBase(float[] input, List<Float> output) {
        for (int i = 0; i < input.length; i++) {
            output.add(input[i]);
        }
    }

    static void flattenDoubleBase(double[] input, List<Double> output) {
        for (int i = 0; i < input.length; i++) {
            output.add(input[i]);
        }
    }

    public static Object reshape(boolean[] input, long[] shape) {
        Object output = ONNXUtil.newBooleanArray(shape);
        reshape(input,output,0);
        return output;
    }

    public static Object reshape(byte[] input, long[] shape) {
        Object output = ONNXUtil.newByteArray(shape);
        reshape(input,output,0);
        return output;
    }

    public static Object reshape(short[] input, long[] shape) {
        Object output = ONNXUtil.newShortArray(shape);
        reshape(input,output,0);
        return output;
    }

    public static Object reshape(int[] input, long[] shape) {
        Object output = ONNXUtil.newIntArray(shape);
        reshape(input,output,0);
        return output;
    }

    public static Object reshape(long[] input, long[] shape) {
        Object output = ONNXUtil.newLongArray(shape);
        reshape(input,output,0);
        return output;
    }

    public static Object reshape(float[] input, long[] shape) {
        Object output = ONNXUtil.newFloatArray(shape);
        reshape(input,output,0);
        return output;
    }

    public static Object reshape(double[] input, long[] shape) {
        Object output = ONNXUtil.newDoubleArray(shape);
        reshape(input,output,0);
        return output;
    }

    static int reshape(Object input, Object output, int position) {
        if (output.getClass().isArray()) {
            Object[] outputArray = (Object[]) output;
            for (Object i : outputArray) {
                Class<?> iClazz = i.getClass();
                if (iClazz.isArray()) {
                    if (iClazz.getComponentType().isPrimitive()) {
                        int length = Array.getLength(i);
                        System.arraycopy(input,position,i,0,length);
                        position += length;
                    } else {
                        position += reshape(input,i,position);
                    }
                } else {
                    throw new IllegalStateException("Found element type when expecting an array. Class " + iClazz);
                }
            }
        } else {
            throw new IllegalStateException("Found element type when expecting an array. Class " + output.getClass());
        }

        return position;
    }

    static void writeBase(double[] input, int position, double[] output) {
        System.arraycopy(input, position, output, 0, output.length);
    }

}
