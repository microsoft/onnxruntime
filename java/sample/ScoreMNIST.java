/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Demo code, supporting both a pytorch CNN trained on MNIST and a scikit-learn model trained on MNIST.
 */
public class ScoreMNIST {

    private static final Logger logger = Logger.getLogger(ScoreMNIST.class.getName());

    /**
     * A named tuple for sparse classification data.
     */
    private static class SparseData {
        public final int[] labels;
        public final List<int[]> indices;
        public final List<float[]> values;

        public SparseData(int[] labels, List<int[]> indices, List<float[]> values) {
            this.labels = labels;
            this.indices = indices;
            this.values = values;
        }
    }

    /**
     * Deserialises the data and puts it in a named tuple.
     * @param path The path to load the data from.
     * @return A named tuple containing the data.
     * @throws IOException If it failed to read the file.
     * @throws ClassNotFoundException If a class wasn't found (only uses JDK types so this would be very odd).
     */
    @SuppressWarnings("unchecked")
    private static SparseData load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int[] labels = (int[]) ois.readObject();
            List<int[]> indices = (List<int[]>) ois.readObject();
            List<float[]> values = (List<float[]>) ois.readObject();
            return new SparseData(labels,indices,values);
        }
    }

    /**
     * Naively takes the softmax of the input.
     * @param input The input array.
     * @return The softmax of the input.
     */
    public static float[] softmax(float[] input) {
        double[] tmp = new double[input.length];
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            double val = Math.exp(input[i]);
            sum += val;
            tmp[i] = val;
        }

        float[] output = new float[input.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = (float) (tmp[i]/sum);
        }

        return output;
    }

    /**
     * Zeros the supplied array.
     * @param data The array to zero.
     */
    public static void zeroData(float[][][][] data) {
        // Zero the array
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                for (int k = 0; k < data[i][j].length; k++) {
                    Arrays.fill(data[i][j][k],0.0f);
                }
            }
        }
    }

    /**
     * Writes out sparse data into the last two dimensions of the supplied 4d array.
     * @param data The 4d array to write to.
     * @param indices The indices of the sparse data.
     * @param values The values of the sparse data.
     */
    public static void writeData(float[][][][] data, int[] indices, float[] values) {
        zeroData(data);

        for (int m = 0; m < indices.length; m++) {
            int i = (indices[m]) / 28;
            int j = (indices[m]) % 28;
            data[0][0][i][j] = values[m]/255;
        }

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                data[0][0][i][j] = (data[0][0][i][j] - 0.1307f) / 0.3081f;
            }
        }
    }

    /**
     * Zeros the array used by the scikit-learn model.
     * @param data The array to zero.
     */
    public static void zeroDataSKL(float[][] data) {
        // Zero the array
        for (int i = 0; i < data.length; i++) {
            Arrays.fill(data[i],0.0f);
        }
    }

    /**
     * Writes out sparse data to the last dimension of the supplied 2d array.
     * @param data The 2d array to write to.
     * @param indices The indices of the sparse data.
     * @param values THe values of the sparse data.
     */
    public static void writeDataSKL(float[][] data, int[] indices, float[] values) {
        zeroDataSKL(data);

        for (int m = 0; m < indices.length; m++) {
            data[0][indices[m]] = values[m];
        }
    }

    /**
     * Find the maximum probability and return it's index.
     * @param probabilities The probabilites.
     * @return The index of the max.
     */
    public static int pred(float[] probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
                idx = i;
            }
        }
        return idx;
    }

    public static void main(String[] args) throws OrtException, IOException, ClassNotFoundException {
        if (args.length < 2 || args.length > 3) {
            System.out.println("Usage: ScoreMNIST <model-path> <test-data> <optional:scikit-learn-flag>");
            System.out.println("The test data input format is a Java serialized file containing an array of int labels, a list of int[] feature indices, and a list of float[] feature values");
            return;
        }

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession.SessionOptions opts = new SessionOptions()) {

            opts.setOptimizationLevel(OptLevel.BASIC_OPT);

            logger.info("Loading model from " + args[0]);
            try (OrtSession session = env.createSession(args[0], opts)) {

                logger.info("Inputs:");
                for (NodeInfo i : session.getInputInfo().values()) {
                    logger.info(i.toString());
                }

                logger.info("Outputs:");
                for (NodeInfo i : session.getOutputInfo().values()) {
                    logger.info(i.toString());
                }

                SparseData data = load(args[1]);

                float[][][][] testData = new float[1][1][28][28];
                float[][] testDataSKL = new float[1][780];

                int correctCount = 0;
                int[][] confusionMatrix = new int[10][10];

                String inputName = session.getInputNames().iterator().next();

                for (int i = 0; i < data.labels.length; i++) {
                    if (args.length==3) {
                        writeDataSKL(testDataSKL, data.indices.get(i), data.values.get(i));
                    } else {
                        writeData(testData, data.indices.get(i), data.values.get(i));
                    }

                    try (OnnxTensor test = OnnxTensor.createTensor(env,args.length==3?testDataSKL:testData);
                         Result output = session.run(Collections.singletonMap(inputName,test))) {

                        int predLabel;

                        if (args.length==3) {
                            long[] labels = (long[]) output.get(0).getValue();
                            predLabel = (int) labels[0];
                        } else {
                            float[][] outputProbs = (float[][]) output.get(0).getValue();
                            predLabel = pred(outputProbs[0]);
                        }
                        if (predLabel == data.labels[i]) {
                            correctCount++;
                        }

                        confusionMatrix[data.labels[i]][predLabel]++;

                        if (i % 500 == 0) {
                            logger.log(Level.INFO, "Cur accuracy = " + ((float)correctCount)/(i+1));
                            logger.log(Level.INFO, "Output type = " + output.get(0).toString());
                            if (args.length == 3) {
                                logger.log(Level.INFO, "Output type = " + output.get(1).toString());
                                logger.log(Level.INFO, "Output value = " + output.get(1).getValue().toString());
                            }
                        }
                    }
                }

                logger.info("Final accuracy = " + ((float)correctCount)/data.labels.length);

                StringBuilder sb = new StringBuilder();
                sb.append("Label");
                for (int i = 0; i < confusionMatrix.length; i++) {
                    sb.append(String.format("%1$5s", ""+i));
                }
                sb.append("\n");

                for (int i = 0; i < confusionMatrix.length; i++) {
                    sb.append(String.format("%1$5s", ""+i));
                    for (int j = 0; j < confusionMatrix[i].length; j++) {
                        sb.append(String.format("%1$5s", ""+confusionMatrix[i][j]));
                    }
                    sb.append("\n");
                }

                System.out.println(sb.toString());
            }
        }

        logger.info("Done!");
    }
}
