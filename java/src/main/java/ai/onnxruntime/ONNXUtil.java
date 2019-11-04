/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Util code for interacting with shape arrays.
 */
public class ONNXUtil {
    /**
     * Creates a new primitive boolean array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A boolean array.
     */
    public static Object newBooleanArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new boolean[(int) shape[0]];
            case 2:
                return new boolean[(int) shape[0]][(int) shape[1]];
            case 3:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new boolean[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive byte array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A byte array.
     */
    public static Object newByteArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new byte[(int) shape[0]];
            case 2:
                return new byte[(int) shape[0]][(int) shape[1]];
            case 3:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new byte[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive short array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A short array.
     */
    public static Object newShortArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new short[(int) shape[0]];
            case 2:
                return new short[(int) shape[0]][(int) shape[1]];
            case 3:
                return new short[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new short[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new short[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new short[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new short[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new short[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive int array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A int array.
     */
    public static Object newIntArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new int[(int) shape[0]];
            case 2:
                return new int[(int) shape[0]][(int) shape[1]];
            case 3:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new int[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive long array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A long array.
     */
    public static Object newLongArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new long[(int) shape[0]];
            case 2:
                return new long[(int) shape[0]][(int) shape[1]];
            case 3:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new long[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive float array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A float array.
     */
    public static Object newFloatArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new float[(int) shape[0]];
            case 2:
                return new float[(int) shape[0]][(int) shape[1]];
            case 3:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new float[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Creates a new primitive double array of up to 8 dimensions, using the supplied shape.
     * <p>
     * Does not check the shape to see if all it's elements are positive.
     *
     * @param shape The shape of array to create.
     * @return A double array.
     */
    public static Object newDoubleArray(long[] shape) {
        switch (shape.length) {
            case 1:
                return new double[(int) shape[0]];
            case 2:
                return new double[(int) shape[0]][(int) shape[1]];
            case 3:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]];
            case 4:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]];
            case 5:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]];
            case 6:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]];
            case 7:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]];
            case 8:
                return new double[(int) shape[0]][(int) shape[1]][(int) shape[2]][(int) shape[3]][(int) shape[4]][(int) shape[5]][(int) shape[6]][(int) shape[7]];
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Reshapes an int array into the desired n-dimensional array, assuming the int array is stored in n-dimensional row-major order.
     * Throws {@link IllegalArgumentException} if the number of elements doesn't match between the shape and the input.
     * @param input The int array.
     * @param shape The desired shape.
     * @return An n-dimensional int array.
     */
    public static Object reshape(int[] input, long[] shape) {
        long elementCount = elementCount(shape);
        // This check implicitly checks if it's a valid Java array, as otherwise it won't be comparable to input.length.
        if (elementCount != input.length) {
            throw new IllegalArgumentException("Expected " + elementCount + " elements to fill this shape, received " + input.length);
        }

        // This could be further optimised with use of System.arraycopy.
        switch (shape.length) {
            case 1: {
                return Arrays.copyOf(input,input.length);
            }
            case 2: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int[][] output = new int[dimOne][dimTwo];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        output[i][j] = input[count];
                        count++;
                    }
                }

                return output;
            }
            case 3: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int[][][] output = new int[dimOne][dimTwo][dimThree];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            output[i][j][k] = input[count];
                            count++;
                        }
                    }
                }

                return output;
            }
            case 4: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int[][][][] output = new int[dimOne][dimTwo][dimThree][dimFour];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                output[i][j][k][l] = input[count];
                                count++;
                            }
                        }
                    }
                }

                return output;
            }
            case 5: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                int[][][][][] output = new int[dimOne][dimTwo][dimThree][dimFour][dimFive];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    output[i][j][k][l][m] = input[count];
                                    count++;
                                }
                            }
                        }
                    }
                }

                return output;
            }
            case 6: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                int dimSix = (int) shape[5];
                int[][][][][][] output = new int[dimOne][dimTwo][dimThree][dimFour][dimFive][dimSix];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    for (int n = 0; n < dimSix; n++) {
                                        output[i][j][k][l][m][n] = input[count];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }

                return output;
            }
            case 7: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                int dimSix = (int) shape[5];
                int dimSeven = (int) shape[6];
                int[][][][][][][] output = new int[dimOne][dimTwo][dimThree][dimFour][dimFive][dimSix][dimSeven];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    for (int n = 0; n < dimSix; n++) {
                                        for (int o = 0; o < dimSeven; o++) {
                                            output[i][j][k][l][m][n][o] = input[count];
                                            count++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return output;
            }
            case 8: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                int dimSix = (int) shape[5];
                int dimSeven = (int) shape[6];
                int dimEight = (int) shape[7];
                int[][][][][][][][] output = new int[dimOne][dimTwo][dimThree][dimFour][dimFive][dimSix][dimSeven][dimEight];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    for (int n = 0; n < dimSix; n++) {
                                        for (int o = 0; o < dimSeven; o++) {
                                            for (int p = 0; p < dimEight; p++) {
                                                output[i][j][k][l][m][n][o][p] = input[count];
                                                count++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return output;
            }
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Reshapes a float array into the desired n-dimensional array.
     * Throws {@link IllegalArgumentException} if the number of elements doesn't match between the shape and the input.
     * @param input The float array.
     * @param shape The desired shape.
     * @return An n-dimensional float array.
     */
    public static Object reshape(float[] input, long[] shape) {
        long elementCount = elementCount(shape);
        // This check implicitly checks if it's a valid Java array, as otherwise it won't be comparable to input.length.
        if (elementCount != input.length) {
            throw new IllegalArgumentException("Expected " + elementCount + " elements to fill this shape, received " + input.length);
        }

        // This could be further optimised with use of System.arraycopy.
        switch (shape.length) {
            case 1: {
                return Arrays.copyOf(input,input.length);
            }
            case 2: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                float[][] output = new float[dimOne][dimTwo];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        output[i][j] = input[count];
                        count++;
                    }
                }

                return output;
            }
            case 3: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                float[][][] output = new float[dimOne][dimTwo][dimThree];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            output[i][j][k] = input[count];
                            count++;
                        }
                    }
                }

                return output;
            }
            case 4: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                float[][][][] output = new float[dimOne][dimTwo][dimThree][dimFour];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                output[i][j][k][l] = input[count];
                                count++;
                            }
                        }
                    }
                }

                return output;
            }
            case 5: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                float[][][][][] output = new float[dimOne][dimTwo][dimThree][dimFour][dimFive];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    output[i][j][k][l][m] = input[count];
                                    count++;
                                }
                            }
                        }
                    }
                }

                return output;
            }
            case 6: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                int dimSix = (int) shape[5];
                float[][][][][][] output = new float[dimOne][dimTwo][dimThree][dimFour][dimFive][dimSix];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    for (int n = 0; n < dimSix; n++) {
                                        output[i][j][k][l][m][n] = input[count];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }

                return output;
            }
            case 7: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                int dimSix = (int) shape[5];
                int dimSeven = (int) shape[6];
                float[][][][][][][] output = new float[dimOne][dimTwo][dimThree][dimFour][dimFive][dimSix][dimSeven];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    for (int n = 0; n < dimSix; n++) {
                                        for (int o = 0; o < dimSeven; o++) {
                                            output[i][j][k][l][m][n][o] = input[count];
                                            count++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return output;
            }
            case 8: {
                int dimOne = (int) shape[0];
                int dimTwo = (int) shape[1];
                int dimThree = (int) shape[2];
                int dimFour = (int) shape[3];
                int dimFive = (int) shape[4];
                int dimSix = (int) shape[5];
                int dimSeven = (int) shape[6];
                int dimEight = (int) shape[7];
                float[][][][][][][][] output = new float[dimOne][dimTwo][dimThree][dimFour][dimFive][dimSix][dimSeven][dimEight];

                int count = 0;
                for (int i = 0; i < dimOne; i++) {
                    for (int j = 0; j < dimTwo; j++) {
                        for (int k = 0; k < dimThree; k++) {
                            for (int l = 0; l < dimFour; l++) {
                                for (int m = 0; m < dimFive; m++) {
                                    for (int n = 0; n < dimSix; n++) {
                                        for (int o = 0; o < dimSeven; o++) {
                                            for (int p = 0; p < dimEight; p++) {
                                                output[i][j][k][l][m][n][o][p] = input[count];
                                                count++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return output;
            }
            default:
                throw new IllegalArgumentException("Arrays with less than 1 and more than 8 dimensions are not supported.");
        }
    }

    /**
     * Counts the number of elements stored in a Tensor of this shape.
     * <p>
     * Multiplies all the elements together if they are positive, throws an {@link IllegalArgumentException} otherwise.
     * </p>
     * @param shape The shape to use.
     * @return The number of elements.
     */
    public static long elementCount(long[] shape) {
        long count = 1;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] > 0) {
                count *= shape[i];
            } else {
                throw new IllegalArgumentException("Received non-positive value in shape " + Arrays.toString(shape) + " .");
            }
        }
        return count;
    }

    /**
     * Checks that the shape is a valid shape for a Java array (i.e. that the values are all positive and representable by an int).
     * @param shape The shape to check.
     * @return True if the shape is valid.
     */
    public static boolean validateShape(long[] shape) {
        boolean valid = true;
        for (int i = 0; i < shape.length; i++) {
            valid &= shape[i] > 0;
            valid &= ((int)shape[i]) == shape[i];
        }
        return valid && shape.length <= TensorInfo.MAX_DIMENSIONS;
    }

    public static String[] flattenString(Object o) {
        List<String> output = new ArrayList<>();

        flattenString((Object[]) o,output);

        return output.toArray(new String[0]);
    }

    private static void flattenString(Object[] input, List output) {
        for (Object i : input) {
            Class<?> iClazz = i.getClass();
            if (iClazz.isArray()) {
                if (iClazz.getComponentType().isArray()) {
                    flattenString((Object[]) i, output);
                } else if (iClazz.getComponentType().equals(String.class)) {
                    output.addAll(Arrays.asList((String[])i));
                } else {
                    throw new IllegalStateException("Found a non-String, non-array element type, " + iClazz);
                }
            } else {
                throw new IllegalStateException("Found an element type where there should have been an array. Class = " + iClazz);
            }
        }
    }
}
