/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Tests for the onnx-runtime Java interface.
 */
public class InferenceTest {
    private static final Pattern LOAD_PATTERN = Pattern.compile("[,\\[\\] ]");
    private static Path resourcePath;
    private static Path otherTestPath;

    static {
        if (System.getProperty("GRADLE_TEST") != null) {
            resourcePath = Paths.get("..","csharp","testdata");
            otherTestPath = Paths.get("..","onnxruntime","test", "testdata");
        } else {
            resourcePath = Paths.get("csharp","testdata");
            otherTestPath = Paths.get("onnxruntime","test", "testdata");
        }
    }

    @Test
    public void createSessionFromPath() {
        String modelPath = resourcePath.resolve("squeezenet.onnx").toString();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment("createSessionFromPath");
             OrtSession.SessionOptions options = new SessionOptions()) {
            try (OrtSession session = env.createSession(modelPath,options)) {
                assertNotNull(session);
                assertEquals(1, session.getNumInputs()); // 1 input node
                Map<String,NodeInfo> inputInfoList = session.getInputInfo();
                assertNotNull(inputInfoList);
                assertEquals(1,inputInfoList.size());
                NodeInfo input = inputInfoList.get("data_0");
                assertEquals("data_0",input.getName()); // input node name
                assertTrue(input.getInfo() instanceof TensorInfo);
                TensorInfo inputInfo = (TensorInfo) input.getInfo();
                assertEquals(OnnxJavaType.FLOAT, inputInfo.type);
                int[] expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                assertEquals(expectedInputDimensions.length, inputInfo.shape.length);
                for (int i = 0; i < expectedInputDimensions.length; i++) {
                    assertEquals(expectedInputDimensions[i], inputInfo.shape[i]);
                }

                assertEquals(1, session.getNumOutputs()); // 1 output node
                Map<String,NodeInfo> outputInfoList = session.getOutputInfo();
                assertNotNull(outputInfoList);
                assertEquals(1,outputInfoList.size());
                NodeInfo output = outputInfoList.get("softmaxout_1");
                assertEquals("softmaxout_1",output.getName()); // output node name
                assertTrue(output.getInfo() instanceof TensorInfo);
                TensorInfo outputInfo = (TensorInfo) output.getInfo();
                assertEquals(OnnxJavaType.FLOAT, outputInfo.type);
                int[] expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                assertEquals(expectedOutputDimensions.length, outputInfo.shape.length);
                for (int i = 0; i < expectedOutputDimensions.length; i++) {
                    assertEquals(expectedOutputDimensions[i], outputInfo.shape[i]);
                }
            }
        } catch (OrtException e) {
            fail("Exception thrown - " + e);
        }
    }
    @Test
    public void createSessionFromByteArray() throws IOException {
        Path modelPath = resourcePath.resolve("squeezenet.onnx");
        byte[] modelBytes = Files.readAllBytes(modelPath);
        try (OrtEnvironment env = OrtEnvironment.getEnvironment("createSessionFromByteArray");
             OrtSession.SessionOptions options = new SessionOptions()) {
            try (OrtSession session = env.createSession(modelBytes,options)) {
                assertNotNull(session);
                assertEquals(1, session.getNumInputs()); // 1 input node
                Map<String,NodeInfo> inputInfoList = session.getInputInfo();
                assertNotNull(inputInfoList);
                assertEquals(1,inputInfoList.size());
                NodeInfo input = inputInfoList.get("data_0");
                assertEquals("data_0",input.getName()); // input node name
                assertTrue(input.getInfo() instanceof TensorInfo);
                TensorInfo inputInfo = (TensorInfo) input.getInfo();
                assertEquals(OnnxJavaType.FLOAT, inputInfo.type);
                int[] expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                assertEquals(expectedInputDimensions.length, inputInfo.shape.length);
                for (int i = 0; i < expectedInputDimensions.length; i++) {
                    assertEquals(expectedInputDimensions[i], inputInfo.shape[i]);
                }

                assertEquals(1, session.getNumOutputs()); // 1 output node
                Map<String,NodeInfo> outputInfoList = session.getOutputInfo();
                assertNotNull(outputInfoList);
                assertEquals(1,outputInfoList.size());
                NodeInfo output = outputInfoList.get("softmaxout_1");
                assertEquals("softmaxout_1",output.getName()); // output node name
                assertTrue(output.getInfo() instanceof TensorInfo);
                TensorInfo outputInfo = (TensorInfo) output.getInfo();
                assertEquals(OnnxJavaType.FLOAT, outputInfo.type);
                int[] expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                assertEquals(expectedOutputDimensions.length, outputInfo.shape.length);
                for (int i = 0; i < expectedOutputDimensions.length; i++) {
                    assertEquals(expectedOutputDimensions[i], outputInfo.shape[i]);
                }
            }
        } catch (OrtException e) {
            fail("Exception thrown - " + e);
        }
    }

    @Test
    public void inferenceTest() {
        canRunInferenceOnAModel(OptLevel.NO_OPT,ExecutionMode.PARALLEL);
        canRunInferenceOnAModel(OptLevel.NO_OPT,ExecutionMode.SEQUENTIAL);
        canRunInferenceOnAModel(OptLevel.ALL_OPT,ExecutionMode.PARALLEL);
        canRunInferenceOnAModel(OptLevel.ALL_OPT,ExecutionMode.SEQUENTIAL);
    }

    private void canRunInferenceOnAModel(OptLevel graphOptimizationLevel, ExecutionMode exectionMode) {
        String modelPath = resourcePath.resolve("squeezenet.onnx").toString();

        // Set the graph optimization level for this session.
        try (OrtEnvironment env = OrtEnvironment.getEnvironment("canRunInferenceOnAModel");
             SessionOptions options = new SessionOptions()) {
            options.setOptimizationLevel(graphOptimizationLevel);
            options.setExecutionMode(exectionMode);

            try (OrtSession session = env.createSession(modelPath, options)) {
                Map<String,NodeInfo> inputMetaMap = session.getInputInfo();
                Map<String,OnnxTensor> container = new HashMap<>();
                NodeInfo inputMeta = inputMetaMap.values().iterator().next();

                float[] inputData = loadTensorFromFile(resourcePath.resolve("bench.in"));
                // this is the data for only one input tensor for this model
                Object tensorData = OrtUtil.reshape(inputData,((TensorInfo) inputMeta.getInfo()).getShape());
                OnnxTensor inputTensor = OnnxTensor.createTensor(env,tensorData);
                container.put(inputMeta.getName(),inputTensor);

                // Run the inference
                try (OrtSession.Result results = session.run(container)) {
                    assertEquals(1, results.size());

                    float[] expectedOutput = loadTensorFromFile(resourcePath.resolve("bench.expected_out"));
                    // validate the results
                    // Only iterates once
                    for (Map.Entry<String, OnnxValue> r : results) {
                        OnnxValue resultValue = r.getValue();
                        assertTrue(resultValue instanceof OnnxTensor);
                        OnnxTensor resultTensor = (OnnxTensor) resultValue;
                        int[] expectedDimensions = new int[]{1, 1000, 1, 1};  // hardcoded for now for the test data
                        long[] resultDimensions = resultTensor.getInfo().getShape();
                        assertEquals(expectedDimensions.length, resultDimensions.length);

                        for (int i = 0; i < expectedDimensions.length; i++) {
                            assertEquals(expectedDimensions[i], resultDimensions[i]);
                        }

                        float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
                        assertEquals(expectedOutput.length, resultArray.length);
                        assertArrayEquals(expectedOutput, resultArray, 1e-6f);
                    }
                } finally {
                    inputTensor.close();
                }
            }
        } catch (OrtException e) {
            fail("Exception thrown - " + e);
        }
    }

    @Test
    public void throwWrongInputType() throws OrtException {
        SqueezeNetTuple tuple = openSessionSqueezeNet();
        try (OrtEnvironment env = tuple.env;
             OrtSession session = tuple.session) {

            float[] inputData = tuple.inputData;
            NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            long[] inputShape = ((TensorInfo)inputMeta.getInfo()).shape;
            int[] inputDataInt = new int[inputData.length];
            for (int i = 0; i < inputData.length; i++) {
                inputDataInt[i] = (int) inputData[i];
            }
            Object tensor = OrtUtil.reshape(inputDataInt,inputShape);
            container.put(inputMeta.getName(),OnnxTensor.createTensor(env,tensor));
            try {
                session.run(container);
                OnnxValue.close(container.values());
                fail("Should throw exception for incorrect type.");
            } catch (OrtException e) {
                OnnxValue.close(container.values());
                String msg = e.getMessage();
                assertTrue(msg.contains("Unexpected input data type"));
            }
        }
    }

    @Test
    public void testMultiThreads() throws OrtException, InterruptedException {
        int numThreads = 10;
        int loop = 10;
        SqueezeNetTuple tuple = openSessionSqueezeNet();
        try (OrtEnvironment env = tuple.env;
             OrtSession session = tuple.session) {

            float[] inputData = tuple.inputData;
            float[] expectedOutput = tuple.outputData;
            NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            long[] inputShape = ((TensorInfo)inputMeta.getInfo()).shape;
            Object tensor = OrtUtil.reshape(inputData, inputShape);
            container.put(inputMeta.getName(),OnnxTensor.createTensor(env,tensor));
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            for (int i = 0; i < numThreads; i++) {
                executor.submit(() -> {
                    for (int j = 0; j < loop; j++) {
                        try (OrtSession.Result result = session.run(container)) {
                            OnnxValue resultTensor = result.get(0);
                            float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
                            assertEquals(expectedOutput.length, resultArray.length);
                            assertArrayEquals(expectedOutput, resultArray, 1e-6f);
                        } catch (OrtException e) {
                            throw new IllegalStateException("Failed to execute a scoring operation",e);
                        }
                    }
                });
            }
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
            OnnxValue.close(container.values());
            assertTrue(executor.isTerminated());
        }
    }

    /*
    @Disabled
    @Test
    public void testPreTrainedModelsOpset7And8() throws IOException, OrtException {
        Set<String> skipModels = new HashSet<>(Arrays.asList(
                "mxnet_arcface",  // Model not supported by CPU execution provider
                "tf_inception_v2",  // TODO: Debug failing model, skipping for now
                "fp16_inception_v1",  // 16-bit float not supported type in Java.
                "fp16_shufflenet",  // 16-bit float not supported type in Java.
                "fp16_tiny_yolov2", // 16-bit float not supported type in Java.
                "test_tiny_yolov2"));

        String[] opsets = new String[]{"opset7", "opset8"};
        Path modelsDir = getTestModelsDir();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testPreTrainedModelsOpset7And8");
             SessionOptions options = new SessionOptions()) {
            for (String opset : opsets) {
                Path modelRoot = modelsDir.resolve(opset);
                for (Path modelDir : Files.newDirectoryStream(modelRoot)) {
                    if (!modelDir.toFile().isDirectory()) {
                        continue;
                    }
                    String onnxModelFileName = null;

                    if (skipModels.contains(modelDir.getFileName().toString())) {
                        continue;
                    }

                    try {
                        File[] onnxModelFiles = modelDir.toFile().listFiles((File f, String name) -> name.endsWith(".onnx"));
                        if (onnxModelFiles.length > 1) {
                            // TODO remove file "._resnet34v2.onnx" from test set
                            boolean validModelFound = false;
                            for (int i = 0; i < onnxModelFiles.length; i++) {
                                if (!onnxModelFiles[i].getName().equals("._resnet34v2.onnx")) {
                                    onnxModelFiles[0] = onnxModelFiles[i];
                                    validModelFound = true;
                                }
                            }

                            if (!validModelFound) {
                                String modelNamesList = Arrays.stream(onnxModelFiles).map(File::toString).collect(Collectors.joining(","));
                                throw new RuntimeException("Opset " + opset + ": Model " + modelDir + ". Can't determine model file name. Found these :" + modelNamesList);
                            }
                        }

                        onnxModelFileName = modelDir.resolve(onnxModelFiles[0].toPath()).toString();
                        try (OrtSession session = env.createSession(onnxModelFileName, options)) {
                            ValueInfo first = session.getInputInfo().get(0).getInfo();
                            long[] inputShape = ((TensorInfo) first).shape;
                            Path testRoot = modelDir.resolve("test_data");
                            Path inputDataPath = testRoot.resolve("input_0.pb");
                            Path outputDataPath = testRoot.resolve("output_0.pb");
                            float[] dataIn = loadTensorFromFilePb(inputDataPath);
                            float[] dataOut = loadTensorFromFilePb(outputDataPath);
                            List<OnnxTensor> nov = new ArrayList<>();
                            nov.add(OnnxTensor.createTensor(env,OrtUtil.reshape(dataIn, inputShape)));
                            OrtSession.Result res = session.run(nov);
                            float[] resultArray = TestHelpers.flattenFloat(res.values().iterator().next().getValue());
                            assertArrayEquals(dataOut, resultArray, 1e-6f);
                            OnnxValue.close(res);
                        }
                    } catch (Exception ex) {
                        String msg = "Opset " + opset + ": Model " + modelDir + ": ModelFile = " + onnxModelFileName + " error = " + ex.getMessage() + ".";
                        throw new RuntimeException(msg);
                    }
                } //model
            } //opset
        }
    }
     */

    @Test
    public void testModelInputFLOAT() throws OrtException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_FLOAT.pb").toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputFLOAT");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            String inputName = session.getInputNames().iterator().next();
            long[] shape = new long[]{1, 5};
            Map<String, OnnxTensor> container = new HashMap<>();
            float[] flatInput = new float[]{1.0f, 2.0f, -3.0f, Float.MIN_VALUE, Float.MAX_VALUE};
            Object tensorIn = OrtUtil.reshape(flatInput, shape);
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName, ov);
            float[] resultArray;
            try (OrtSession.Result res = session.run(container)) {
                resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray, 1e-6f);
                float[] resultBufferArray = new float[flatInput.length];
                ((OnnxTensor)res.get(0)).getFloatBuffer().get(resultBufferArray);
                assertArrayEquals(flatInput,resultBufferArray,1e-6f);
                OnnxValue.close(container);
            }
            container.clear();

            // Now test loading from buffer
            FloatBuffer buffer = FloatBuffer.wrap(flatInput);
            OnnxTensor newTensor = OnnxTensor.createTensor(env,buffer,shape);
            container.put(inputName,newTensor);
            try (OrtSession.Result res = session.run(container)) {
                resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray, 1e-6f);
                OnnxValue.close(container);
            }
        }
    }

    @Test
    public void testModelInputBOOL() throws OrtException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_BOOL.pb").toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputBOOL");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            String inputName = session.getInputNames().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            boolean[] flatInput = new boolean[] { true, false, true, false, true };
            Object tensorIn = OrtUtil.reshape(flatInput, new long[] { 1, 5 });
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName,ov);
            try (OrtSession.Result res = session.run(container)) {
                boolean[] resultArray = TestHelpers.flattenBoolean(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray);
            }
            OnnxValue.close(container);
        }
    }

    @Test
    public void testModelInputINT32() throws OrtException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT32.pb").toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT32");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            String inputName = session.getInputNames().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            int[] flatInput = new int[] { 1, -2, -3, Integer.MIN_VALUE, Integer.MAX_VALUE };
            Object tensorIn = OrtUtil.reshape(flatInput, new long[] { 1, 5 });
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName,ov);
            try (OrtSession.Result res = session.run(container)) {
                int[] resultArray = TestHelpers.flattenInteger(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray);
            }
            OnnxValue.close(container);
        }
    }

    @Test
    public void testModelInputDOUBLE() throws OrtException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_DOUBLE.pb").toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputDOUBLE");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            String inputName = session.getInputNames().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            double[] flatInput = new double[] { 1.0, 2.0, -3.0, 5, 5 };
            Object tensorIn = OrtUtil.reshape(flatInput, new long[] { 1, 5 });
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName,ov);
            try (OrtSession.Result res = session.run(container)) {
                double[] resultArray = TestHelpers.flattenDouble(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray, 1e-6f);
            }
            OnnxValue.close(container);
        }
    }

    @Test
    public void TestModelInputINT8() throws OrtException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT8.pb").toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT8");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            String inputName = session.getInputNames().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            byte[] flatInput = new byte[] { 1, 2, -3, Byte.MIN_VALUE, Byte.MAX_VALUE };
            Object tensorIn = OrtUtil.reshape(flatInput, new long[] { 1, 5 });
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName,ov);
            try (OrtSession.Result res = session.run(container)) {
                byte[] resultArray = TestHelpers.flattenByte(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray);
            }
            OnnxValue.close(container);
        }
    }

    @Test
    public void TestModelInputINT16() throws OrtException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT16.pb").toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT16");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            String inputName = session.getInputNames().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            short[] flatInput = new short[] { 1, 2, 3, Short.MIN_VALUE, Short.MAX_VALUE };
            Object tensorIn = OrtUtil.reshape(flatInput, new long[] { 1, 5 });
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName,ov);
            try (OrtSession.Result res = session.run(container)) {
                short[] resultArray = TestHelpers.flattenShort(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray);
            }
            OnnxValue.close(container);
        }
    }

    @Test
    public void TestModelInputINT64() throws OrtException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT64.pb").toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT64");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            String inputName = session.getInputNames().iterator().next();
            Map<String,OnnxTensor> container = new HashMap<>();
            long[] flatInput = new long[] { 1, 2, -3, Long.MIN_VALUE, Long.MAX_VALUE };
            Object tensorIn = OrtUtil.reshape(flatInput, new long[] { 1, 5 });
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName,ov);
            try (OrtSession.Result res = session.run(container)) {
                long[] resultArray = TestHelpers.flattenLong(res.get(0).getValue());
                assertArrayEquals(flatInput, resultArray);
            }
            OnnxValue.close(container);
        }
    }

    @Test
    public void testModelSequenceOfMapIntFloat() throws OrtException {
        // test model trained using lightgbm classifier
        // produces 2 named outputs
        //   "label" is a tensor,
        //   "probabilities" is a sequence<map<int64, float>>
        // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

        String modelPath = resourcePath.resolve("test_sequence_map_int_float.pb").toString();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelSequenceOfMapIntFloat");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {

            Map<String,NodeInfo> outputInfos = session.getOutputInfo();
            Iterator<NodeInfo> valuesItr = outputInfos.values().iterator();
            NodeInfo firstOutputInfo = valuesItr.next();
            NodeInfo secondOutputInfo = valuesItr.next();
            assertTrue(firstOutputInfo.getInfo() instanceof TensorInfo);
            assertTrue(secondOutputInfo.getInfo() instanceof SequenceInfo);
            assertEquals(OnnxJavaType.INT64,((TensorInfo)firstOutputInfo.getInfo()).type);

            Map<String,OnnxTensor> container = new HashMap<>();
            long[] shape = new long[] { 1, 2 };
            float[] flatInput = new float[] {5.8f, 2.8f};
            Object tensorIn = OrtUtil.reshape(flatInput,shape);
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(session.getInputNames().iterator().next(),ov);

            try (OrtSession.Result outputs = session.run(container)) {
                assertEquals(2, outputs.size());

                // first output is a tensor containing label
                OnnxValue firstOutput = outputs.get(firstOutputInfo.getName()).get();
                assertTrue(firstOutput instanceof OnnxTensor);

                // try-cast as a tensor
                long[] labelOutput = (long[]) firstOutput.getValue();

                // Label 1 should have highest probability
                assertEquals(1, labelOutput[0]);
                assertEquals(1, labelOutput.length);

                // second output is a sequence<map<int64, float>>
                // try-cast to an sequence of NOV
                OnnxValue secondOutput = outputs.get(secondOutputInfo.getName()).get();
                assertTrue(secondOutput instanceof OnnxSequence);
                SequenceInfo sequenceInfo = ((OnnxSequence) secondOutput).getInfo();
                assertTrue(sequenceInfo.sequenceOfMaps);
                assertEquals(OnnxJavaType.INT64, sequenceInfo.mapInfo.keyType);
                assertEquals(OnnxJavaType.FLOAT, sequenceInfo.mapInfo.valueType);

                // try-cast first element in sequence to map/dictionary type
                Map<Long, Float> map = (Map<Long, Float>) ((List<Object>) secondOutput.getValue()).get(0);
                assertEquals(0.25938290, map.get(0L), 1e-6);
                assertEquals(0.40904793, map.get(1L), 1e-6);
                assertEquals(0.33156919, map.get(2L), 1e-6);
            }
            ov.close();
        }
    }

    @Test
    public void testModelSequenceOfMapStringFloat() throws OrtException {
        // test model trained using lightgbm classifier
        // produces 2 named outputs
        //   "label" is a tensor,
        //   "probabilities" is a sequence<map<int64, float>>
        // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py
        String modelPath = resourcePath.resolve("test_sequence_map_string_float.pb").toString();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelSequenceOfMapStringFloat");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {

            Map<String,NodeInfo> outputInfos = session.getOutputInfo();
            Iterator<NodeInfo> valuesItr = outputInfos.values().iterator();
            NodeInfo firstOutputInfo = valuesItr.next();
            NodeInfo secondOutputInfo = valuesItr.next();
            assertTrue(firstOutputInfo.getInfo() instanceof TensorInfo);
            assertTrue(secondOutputInfo.getInfo() instanceof SequenceInfo);
            assertEquals(OnnxJavaType.STRING,((TensorInfo)firstOutputInfo.getInfo()).type);

            Map<String,OnnxTensor> container = new HashMap<>();
            long[] shape = new long[] { 1, 2 };
            float[] flatInput = new float[] {5.8f, 2.8f};
            Object tensorIn = OrtUtil.reshape(flatInput,shape);
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(session.getInputNames().iterator().next(),ov);

            try (OrtSession.Result outputs = session.run(container)) {
                assertEquals(2, outputs.size());

                // first output is a tensor containing label
                OnnxValue firstOutput = outputs.get(firstOutputInfo.getName()).get();
                assertTrue(firstOutput instanceof OnnxTensor);

                // try-cast as a tensor
                String[] labelOutput = (String[]) firstOutput.getValue();

                // Label 1 should have highest probability
                assertEquals("1", labelOutput[0]);
                assertEquals(1, labelOutput.length);

                // second output is a sequence<map<int64, float>>
                // try-cast to an sequence of NOV
                OnnxValue secondOutput = outputs.get(secondOutputInfo.getName()).get();
                assertTrue(secondOutput instanceof OnnxSequence);
                SequenceInfo sequenceInfo = ((OnnxSequence) secondOutput).getInfo();
                assertTrue(sequenceInfo.sequenceOfMaps);
                assertEquals(OnnxJavaType.STRING, sequenceInfo.mapInfo.keyType);
                assertEquals(OnnxJavaType.FLOAT, sequenceInfo.mapInfo.valueType);

                // try-cast first element in sequence to map/dictionary type
                Map<String, Float> map = (Map<String, Float>) ((List<Object>) secondOutput.getValue()).get(0);
                assertEquals(0.25938290, map.get("0"), 1e-6);
                assertEquals(0.40904793, map.get("1"), 1e-6);
                assertEquals(0.33156919, map.get("2"), 1e-6);
            }
            ov.close();
        }
    }

    @Test
    public void testStringIdentity() throws OrtException {
        String modelPath = otherTestPath.resolve("identity_string.onnx").toString();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment("testStringIdentity");
             SessionOptions options = new SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {

            Map<String,NodeInfo> outputInfos = session.getOutputInfo();
            ValueInfo firstOutputInfo = outputInfos.values().iterator().next().getInfo();
            assertTrue(firstOutputInfo instanceof TensorInfo);
            assertEquals(OnnxJavaType.STRING,((TensorInfo)firstOutputInfo).type);

            String inputName = session.getInputNames().iterator().next();

            Map<String,OnnxTensor> container = new HashMap<>();
            String[][] tensorIn = new String[][]{new String[] {"this", "is"}, new String[] {"identity", "test"}};
            OnnxTensor ov = OnnxTensor.createTensor(env,tensorIn);
            container.put(inputName,ov);

            try (OrtSession.Result outputs = session.run(container)) {
                assertEquals(1, outputs.size());

                OnnxValue firstOutput = outputs.get(0);
                assertTrue(firstOutput instanceof OnnxTensor);

                String[] labelOutput = (String[]) firstOutput.getValue();

                assertEquals("this", labelOutput[0]);
                assertEquals("is", labelOutput[1]);
                assertEquals("identity", labelOutput[2]);
                assertEquals("test", labelOutput[3]);
                assertEquals(4, labelOutput.length);

                OnnxValue.close(container);
                container.clear();
            }

            String[] tensorInFlatArr = new String[]{"this", "is", "identity", "test"};
            ov = OnnxTensor.createTensor(env,tensorInFlatArr, new long[]{2,2});
            container.put(inputName,ov);

            try (OrtSession.Result outputs = session.run(container)) {
                assertEquals(1, outputs.size());

                OnnxValue firstOutput = outputs.get(0);
                assertTrue(firstOutput instanceof OnnxTensor);

                String[] labelOutput = (String[]) firstOutput.getValue();

                assertEquals("this", labelOutput[0]);
                assertEquals("is", labelOutput[1]);
                assertEquals("identity", labelOutput[2]);
                assertEquals("test", labelOutput[3]);
                assertEquals(4, labelOutput.length);
            }
        }
    }

    /**
     * Carrier tuple for the squeeze net model.
     */
    private static class SqueezeNetTuple {
        public final OrtEnvironment env;
        public final OrtSession session;
        public final float[] inputData;
        public final float[] outputData;

        public SqueezeNetTuple(OrtEnvironment env, OrtSession session, float[] inputData, float[] outputData) {
            this.env = env;
            this.session = session;
            this.inputData = inputData;
            this.outputData = outputData;
        }
    }

    private static SqueezeNetTuple openSessionSqueezeNet() throws OrtException {
        return openSessionSqueezeNet(-1);
    }

    private static SqueezeNetTuple openSessionSqueezeNet(int cudaDeviceId) throws OrtException {
        Path squeezeNet = resourcePath.resolve("squeezenet.onnx");
        String modelPath = squeezeNet.toString();
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        SessionOptions options = new SessionOptions();
        if (cudaDeviceId != -1) {
            options.addCUDA(cudaDeviceId);
        }
        OrtSession session = env.createSession(modelPath,options);
        float[] inputData = loadTensorFromFile(resourcePath.resolve("bench.in"));
        float[] expectedOutput = loadTensorFromFile(resourcePath.resolve("bench.expected_out"));
        return new SqueezeNetTuple(env, session, inputData, expectedOutput);
    }

    private static float[] loadTensorFromFile(Path filename) {
        return loadTensorFromFile(filename,true);
    }

    private static float[] loadTensorFromFile(Path filename, boolean skipHeader) {
        // read data from file
        try (BufferedReader reader = new BufferedReader(new FileReader(filename.toFile()))) {
            if (skipHeader) {
                reader.readLine(); //skip the input name
            }
            String[] dataStr = LOAD_PATTERN.split(reader.readLine());
            List<Float> tensorData = new ArrayList<>();
            for (int i = 0; i < dataStr.length; i++) {
                if (!dataStr[i].isEmpty()) {
                    tensorData.add(Float.parseFloat(dataStr[i]));
                }
            }
            return TestHelpers.toPrimitiveFloat(tensorData);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static class TypeWidth {
        public final OnnxJavaType type;
        public final int width;
        public TypeWidth(OnnxJavaType type, int width) {
            this.type = type;
            this.width = width;
        }
    }

    /*
    private static OnnxTensor loadTensorFromFilePb(String filename, Map<String,NodeInfo> nodeMetaDict) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(filename));
        OnnxMl.TensorProto tensor = OnnxMl.TensorProto.parseFrom(is);
        is.close();

        TypeWidth tw = getTypeAndWidth(tensor.getDataType());
        int width = tw.width;
        OnnxJavaType tensorElemType = tw.type;
        int[] intDims = new int[tensor.getDimsCount()];
        for (int i = 0; i < tensor.getDimsCount(); i++) {
            intDims[i] = (int)tensor.getDims(i);
        }

        TensorInfo nodeMeta = null;
        String nodeName = "";
        if (nodeMetaDict.size() == 1) {
            for (Map.Entry<String,NodeInfo> e : nodeMetaDict.entrySet()) {
                nodeMeta = (TensorInfo) e.getValue().getInfo();
                nodeName = e.getKey(); // valid for single node input
            }
        } else if (nodeMetaDict.size() > 1) {
            if (!tensor.getName().isEmpty()) {
                nodeMeta = (TensorInfo) nodeMetaDict.get(tensor.getName()).getInfo();
                nodeName = tensor.getName();
            } else {
                boolean matchfound = false;
                // try to find from matching type and shape
                for (Map.Entry<String,NodeInfo> e : nodeMetaDict.entrySet()) {
                    if (e.getValue().getInfo() instanceof TensorInfo) {
                        TensorInfo meta = (TensorInfo) e.getValue().getInfo();
                        if (tensorElemType == meta.type && tensor.getDimsCount() == meta.shape.length) {
                            int i = 0;
                            for (; i < meta.shape.length; i++) {
                                if (meta.shape[i] != -1 && meta.shape[i] != intDims[i]) {
                                    break;
                                }
                            }
                            if (i >= meta.shape.length) {
                                matchfound = true;
                                nodeMeta = meta;
                                nodeName = e.getKey();
                                break;
                            }
                        }
                    }
                }
                if (!matchfound) {
                    // throw error
                    throw new IllegalStateException("No matching Tensor found in InputOutputMetadata corresponding to the serialized tensor loaded from " + filename);
                }
            }
        } else {
            // throw error
            throw new IllegalStateException("While reading the serialized tensor loaded from " + filename + ", metaDataDict has 0 elements");
        }

        Assertions.assertEquals(tensorElemType, nodeMeta.type);
        Assertions.assertEquals(nodeMeta.shape.length, tensor.getDimsCount());
        for (int i = 0; i < nodeMeta.shape.length; i++) {
            Assertions.assertTrue((nodeMeta.shape[i] == -1) || (nodeMeta.shape[i] == intDims[i]));
        }

        return createOnnxTensorFromRawData(nodeName, tensor.getRawData().toByteArray(), nodeMeta, intDims);
    }
     */

}
