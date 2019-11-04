/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.ONNXSession.SessionOptions;
import ai.onnxruntime.ONNXSession.SessionOptions.OptLevel;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * Tests for the onnx-runtime Java interface.
 */
public class InferenceTest {
    private static final Pattern LOAD_PATTERN = Pattern.compile("[,\\[\\] ]");
    private static final Path resourcePath = Paths.get("..","csharp","testdata");

    @Test
    public void createSessionFromPath() {
        String modelPath = resourcePath.resolve("squeezenet.onnx").toString();
        try (ONNXEnvironment env = new ONNXEnvironment("createSessionFromPath");
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession.SessionOptions options = new SessionOptions()) {
            try (ONNXSession session = env.createSession(modelPath,allocator,options)) {
                assertNotNull(session);
                assertEquals(1, session.getNumInputs()); // 1 input node
                List<NodeInfo> inputInfoList = session.getInputInfo();
                assertNotNull(inputInfoList);
                assertEquals(1,inputInfoList.size());
                NodeInfo input = inputInfoList.get(0);
                assertEquals("data_0",input.getName()); // input node name
                assertTrue(input.getInfo() instanceof TensorInfo);
                TensorInfo inputInfo = (TensorInfo) input.getInfo();
                assertEquals(ONNXJavaType.FLOAT, inputInfo.type);
                int[] expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                assertEquals(expectedInputDimensions.length, inputInfo.shape.length);
                for (int i = 0; i < expectedInputDimensions.length; i++) {
                    assertEquals(expectedInputDimensions[i], inputInfo.shape[i]);
                }

                assertEquals(1, session.getNumOutputs()); // 1 output node
                List<NodeInfo> outputInfoList = session.getOutputInfo();
                assertNotNull(outputInfoList);
                assertEquals(1,outputInfoList.size());
                NodeInfo output = outputInfoList.get(0);
                assertEquals("softmaxout_1",output.getName()); // output node name
                assertTrue(output.getInfo() instanceof TensorInfo);
                TensorInfo outputInfo = (TensorInfo) output.getInfo();
                assertEquals(ONNXJavaType.FLOAT, outputInfo.type);
                int[] expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                assertEquals(expectedOutputDimensions.length, outputInfo.shape.length);
                for (int i = 0; i < expectedOutputDimensions.length; i++) {
                    assertEquals(expectedOutputDimensions[i], outputInfo.shape[i]);
                }
            }
        } catch (ONNXException e) {
            fail("Exception thrown - " + e);
        }
    }
    @Test
    public void createSessionFromByteArray() throws IOException {
        Path modelPath = resourcePath.resolve("squeezenet.onnx");
        byte[] modelBytes = Files.readAllBytes(modelPath);
        try (ONNXEnvironment env = new ONNXEnvironment("createSessionFromByteArray");
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession.SessionOptions options = new SessionOptions()) {
            try (ONNXSession session = env.createSession(modelBytes,allocator,options)) {
                assertNotNull(session);
                assertEquals(1, session.getNumInputs()); // 1 input node
                List<NodeInfo> inputInfoList = session.getInputInfo();
                assertNotNull(inputInfoList);
                assertEquals(1,inputInfoList.size());
                NodeInfo input = inputInfoList.get(0);
                assertEquals("data_0",input.getName()); // input node name
                assertTrue(input.getInfo() instanceof TensorInfo);
                TensorInfo inputInfo = (TensorInfo) input.getInfo();
                assertEquals(ONNXJavaType.FLOAT, inputInfo.type);
                int[] expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                assertEquals(expectedInputDimensions.length, inputInfo.shape.length);
                for (int i = 0; i < expectedInputDimensions.length; i++) {
                    assertEquals(expectedInputDimensions[i], inputInfo.shape[i]);
                }

                assertEquals(1, session.getNumOutputs()); // 1 output node
                List<NodeInfo> outputInfoList = session.getOutputInfo();
                assertNotNull(outputInfoList);
                assertEquals(1,outputInfoList.size());
                NodeInfo output = outputInfoList.get(0);
                assertEquals("softmaxout_1",output.getName()); // output node name
                assertTrue(output.getInfo() instanceof TensorInfo);
                TensorInfo outputInfo = (TensorInfo) output.getInfo();
                assertEquals(ONNXJavaType.FLOAT, outputInfo.type);
                int[] expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                assertEquals(expectedOutputDimensions.length, outputInfo.shape.length);
                for (int i = 0; i < expectedOutputDimensions.length; i++) {
                    assertEquals(expectedOutputDimensions[i], outputInfo.shape[i]);
                }
            }
        } catch (ONNXException e) {
            fail("Exception thrown - " + e);
        }
    }

    @Test
    public void inferenceTest() {
        canRunInferenceOnAModel(OptLevel.NO_OPT,true);
        canRunInferenceOnAModel(OptLevel.NO_OPT,false);
        canRunInferenceOnAModel(OptLevel.ALL_OPT,true);
        canRunInferenceOnAModel(OptLevel.ALL_OPT,false);
    }

    private void canRunInferenceOnAModel(OptLevel graphOptimizationLevel, boolean disableSequentialExecution) {
        String modelPath = resourcePath.resolve("squeezenet.onnx").toString();

        // Set the graph optimization level for this session.
        try (ONNXEnvironment env = new ONNXEnvironment("canRunInferenceOnAModel");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator()) {
            options.setOptimisationLevel(graphOptimizationLevel);
            if (disableSequentialExecution) {
                options.setSequentialExecution(false);
            }

            try (ONNXSession session = env.createSession(modelPath, allocator, options)) {
                List<NodeInfo> inputMeta = session.getInputInfo();
                List<ONNXTensor> container = new ArrayList<>();

                float[] inputData = loadTensorFromFile(resourcePath.resolve("bench.in"));
                // this is the data for only one input tensor for this model
                Object tensorData = ONNXUtil.reshape(inputData,((TensorInfo) inputMeta.get(0).getInfo()).getShape());
                container.add(allocator.createTensor(tensorData));

                // Run the inference
                List<ONNXValue> results = session.score(container);
                assertEquals(1, results.size());

                float[] expectedOutput = loadTensorFromFile(resourcePath.resolve("bench.expected_out"));
                // validate the results
                // Only iterates once
                for (ONNXValue r : results) {
                    assertTrue(r instanceof ONNXTensor);
                    ONNXTensor resultTensor = (ONNXTensor) r;
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
                ONNXValue.close(container);
                ONNXValue.close(results);
            }
        } catch (ONNXException e) {
            fail("Exception thrown - " + e);
        }
    }

    @Test
    public void throwWrongInputType() throws ONNXException {
        SqueezeNetTuple tuple = openSessionSqueezeNet();
        try (ONNXEnvironment env = tuple.env;
             ONNXAllocator allocator = tuple.allocator;
             ONNXSession session = tuple.session) {

            float[] inputData = tuple.inputData;
            List<ONNXTensor> container = new ArrayList<>();
            long[] inputShape = ((TensorInfo)session.getInputInfo().get(0).getInfo()).shape;
            int[] inputDataInt = new int[inputData.length];
            for (int i = 0; i < inputData.length; i++) {
                inputDataInt[i] = (int) inputData[i];
            }
            Object tensor = ONNXUtil.reshape(inputDataInt,inputShape);
            container.add(allocator.createTensor(tensor));
            try {
                session.score(container);
                ONNXValue.close(container);
                fail("Should throw exception for incorrect type.");
            } catch (ONNXException e) {
                ONNXValue.close(container);
                String msg = e.getMessage();
                assertTrue(msg.contains("Unexpected input data type"));
            }
        }
    }

    @Test
    public void throwExtraInputs() throws ONNXException {
        SqueezeNetTuple tuple = openSessionSqueezeNet();
        try (ONNXEnvironment env = tuple.env;
             ONNXAllocator allocator = tuple.allocator;
             ONNXSession session = tuple.session) {

            float[] inputData = tuple.inputData;
            List<ONNXTensor> container = new ArrayList<>();
            long[] inputShape = ((TensorInfo) session.getInputInfo().get(0).getInfo()).shape;
            Object tensor = ONNXUtil.reshape(inputData, inputShape);
            container.add(allocator.createTensor(tensor));
            container.add(allocator.createTensor(tensor));
            try {
                session.score(container);
                ONNXValue.close(container);
                fail("Should throw exception for incorrect number of inputs.");
            } catch (ONNXException e) {
                ONNXValue.close(container);
                String msg = e.getMessage();
                assertTrue(msg.contains("Unexpected number of inputs"));
            }
        }
    }

    @Test
    public void testMultiThreads() throws ONNXException, InterruptedException {
        int numThreads = 10;
        int loop = 10;
        SqueezeNetTuple tuple = openSessionSqueezeNet();
        try (ONNXEnvironment env = tuple.env;
             ONNXAllocator allocator = tuple.allocator;
             ONNXSession session = tuple.session) {

            float[] inputData = tuple.inputData;
            float[] expectedOutput = tuple.outputData;
            List<ONNXTensor> container = new ArrayList<>();
            long[] inputShape = ((TensorInfo) session.getInputInfo().get(0).getInfo()).shape;
            Object tensor = ONNXUtil.reshape(inputData, inputShape);
            container.add(allocator.createTensor(tensor));
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            for (int i = 0; i < numThreads; i++) {
                executor.submit(() -> {
                    for (int j = 0; j < loop; j++) {
                        try (ONNXValue resultTensor = session.score(container).get(0)) {
                            float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
                            assertEquals(expectedOutput.length, resultArray.length);
                            assertArrayEquals(expectedOutput, resultArray, 1e-6f);
                        } catch (ONNXException e) {
                            throw new IllegalStateException("Failed to execute a scoring operation",e);
                        }
                    }
                });
            }
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
            ONNXValue.close(container);
            assertTrue(executor.isTerminated());
        }
    }

    /*
    @Disabled
    @Test
    public void testPreTrainedModelsOpset7And8() throws IOException, ONNXException {
        Set<String> skipModels = new HashSet<>(Arrays.asList(
                "mxnet_arcface",  // Model not supported by CPU execution provider
                "tf_inception_v2",  // TODO: Debug failing model, skipping for now
                "fp16_inception_v1",  // 16-bit float not supported type in C#.
                "fp16_shufflenet",  // 16-bit float not supported type in C#.
                "fp16_tiny_yolov2", // 16-bit float not supported type in C#.
                "test_tiny_yolov2"));

        String[] opsets = new String[]{"opset7", "opset8"};
        Path modelsDir = GetTestModelsDir();
        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputFLOAT");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator()) {
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
                        try (ONNXSession session = env.createSession(onnxModelFileName, allocator, options)) {
                            ValueInfo first = session.getInputInfo().get(0).getInfo();
                            long[] inputShape = ((TensorInfo) first).shape;
                            Path testRoot = modelDir.resolve("test_data");
                            Path inputDataPath = testRoot.resolve("input_0.pb");
                            Path outputDataPath = testRoot.resolve("output_0.pb");
                            float[] dataIn = LoadTensorFromFilePb(inputDataPath);
                            float[] dataOut = LoadTensorFromFilePb(outputDataPath);
                            List<ONNXTensor> nov = new ArrayList<>();
                            nov.add(allocator.createTensor(ONNXUtil.reshape(dataIn, inputShape)));
                            List<ONNXValue> res = session.score(nov);
                            float[] resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
                            assertArrayEquals(dataOut, resultArray, 1e-6f);
                            ONNXValue.close(res);
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
    public void testModelInputFLOAT() throws ONNXException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_FLOAT.pb").toString();

        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputFLOAT");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {
            long[] shape = new long[] { 1, 5 };
            List<ONNXTensor> container = new ArrayList<>();
            float[] flatInput = new float[] { 1.0f, 2.0f, -3.0f, Float.MIN_VALUE, Float.MAX_VALUE };
            Object tensorIn = ONNXUtil.reshape(flatInput,shape);
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);
            List<ONNXValue> res = session.score(container);
            float[] resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray,1e-6f);
            ONNXValue.close(res);
            ONNXValue.close(container);
            container.clear();

            // Now test loading from buffer
            FloatBuffer buffer = FloatBuffer.wrap(flatInput);
            ONNXTensor newTensor = allocator.createTensor(buffer,shape);
            Object array = newTensor.getValue();
            container.add(newTensor);
            res = session.score(container);
            resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray,1e-6f);
            ONNXValue.close(res);
            ONNXValue.close(container);
        }
    }

    @Disabled
    @Test
    public void testModelInputBOOL() throws ONNXException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_BOOL.pb").toString();

        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputBOOL");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {
            List<ONNXTensor> container = new ArrayList<>();
            boolean[] flatInput = new boolean[] { true, false, true, false, true };
            Object tensorIn = TestHelpers.reshape(flatInput, new long[] { 1, 5 });
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);
            List<ONNXValue> res = session.score(container);
            boolean[] resultArray = TestHelpers.flattenBoolean(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray);
            ONNXValue.close(res);
            ONNXValue.close(container);
        }
    }

    @Test
    public void testModelInputINT32() throws ONNXException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT32.pb").toString();

        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputINT32");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {
            List<ONNXTensor> container = new ArrayList<>();
            int[] flatInput = new int[] { 1, -2, -3, Integer.MIN_VALUE, Integer.MAX_VALUE };
            Object tensorIn = TestHelpers.reshape(flatInput, new long[] { 1, 5 });
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);
            List<ONNXValue> res = session.score(container);
            int[] resultArray = TestHelpers.flattenInteger(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray);
            ONNXValue.close(res);
            ONNXValue.close(container);
        }
    }

    @Test
    public void testModelInputDOUBLE() throws ONNXException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_DOUBLE.pb").toString();

        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputDOUBLE");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {
            List<ONNXTensor> container = new ArrayList<>();
            double[] flatInput = new double[] { 1.0, 2.0, -3.0, 5, 5 };
            Object tensorIn = TestHelpers.reshape(flatInput, new long[] { 1, 5 });
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);
            List<ONNXValue> res = session.score(container);
            double[] resultArray = TestHelpers.flattenDouble(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray,1e-6f);
            ONNXValue.close(res);
            ONNXValue.close(container);
        }
    }

    @Disabled // Model expects a 4d tensor, but the C# code is supplying a 2d tensor.
    @Test
    public void TestModelInputINT8() throws ONNXException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT8.pb").toString();

        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputINT8");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {
            List<ONNXTensor> container = new ArrayList<>();
            byte[] flatInput = new byte[] { 1, 2, -3, Byte.MIN_VALUE, Byte.MAX_VALUE };
            Object tensorIn = TestHelpers.reshape(flatInput, new long[] { 1, 5 });
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);
            List<ONNXValue> res = session.score(container);
            byte[] resultArray = TestHelpers.flattenByte(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray);
            ONNXValue.close(res);
            ONNXValue.close(container);
        }
    }

    @Test
    public void TestModelInputINT16() throws ONNXException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT16.pb").toString();

        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputINT16");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {
            List<ONNXTensor> container = new ArrayList<>();
            short[] flatInput = new short[] { 1, 2, 3, Short.MIN_VALUE, Short.MAX_VALUE };
            Object tensorIn = TestHelpers.reshape(flatInput, new long[] { 1, 5 });
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);
            List<ONNXValue> res = session.score(container);
            short[] resultArray = TestHelpers.flattenShort(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray);
            ONNXValue.close(res);
            ONNXValue.close(container);
        }
    }

    @Test
    public void TestModelInputINT64() throws ONNXException {
        // model takes 1x5 input of fixed type, echoes back
        String modelPath = resourcePath.resolve("test_types_INT64.pb").toString();

        try (ONNXEnvironment env = new ONNXEnvironment("testModelInputINT64");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {
            List<ONNXTensor> container = new ArrayList<>();
            long[] flatInput = new long[] { 1, 2, -3, Long.MIN_VALUE, Long.MAX_VALUE };
            Object tensorIn = TestHelpers.reshape(flatInput, new long[] { 1, 5 });
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);
            List<ONNXValue> res = session.score(container);
            long[] resultArray = TestHelpers.flattenLong(res.get(0).getValue());
            assertArrayEquals(flatInput,resultArray);
            ONNXValue.close(res);
            ONNXValue.close(container);
        }
    }

    @Test
    public void testModelSequenceOfMapIntFloat() throws ONNXException {
        // test model trained using lightgbm classifier
        // produces 2 named outputs
        //   "label" is a tensor,
        //   "probabilities" is a sequence<map<int64, float>>
        // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

        String modelPath = resourcePath.resolve("test_sequence_map_int_float.pb").toString();
        try (ONNXEnvironment env = new ONNXEnvironment("testModelSequenceOfMapIntFloat");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {

            List<NodeInfo> outputInfos = session.getOutputInfo();
            ValueInfo firstOutputInfo = outputInfos.get(0).getInfo();
            ValueInfo secondOutputInfo = outputInfos.get(1).getInfo();
            assertTrue(firstOutputInfo instanceof TensorInfo);
            assertTrue(secondOutputInfo instanceof SequenceInfo);
            assertEquals(ONNXJavaType.INT64,((TensorInfo)firstOutputInfo).type);

            List<ONNXTensor> container = new ArrayList<>();
            long[] shape = new long[] { 1, 2 };
            float[] flatInput = new float[] {5.8f, 2.8f};
            Object tensorIn = ONNXUtil.reshape(flatInput,shape);
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);

            List<ONNXValue> outputs = session.score(container);
            assertEquals(2,outputs.size());

            // first output is a tensor containing label
            ONNXValue firstOutput = outputs.get(0);
            assertTrue(firstOutput instanceof ONNXTensor);

            // try-cast as a tensor
            long[] labelOutput = (long[]) firstOutput.getValue();

            // Label 1 should have highest probability
            assertEquals(1, labelOutput[0]);
            assertEquals(1,labelOutput.length);

            // second output is a sequence<map<int64, float>>
            // try-cast to an sequence of NOV
            ONNXValue secondOutput = outputs.get(1);
            assertTrue(secondOutput instanceof ONNXSequence);
            SequenceInfo sequenceInfo = ((ONNXSequence)secondOutput).getInfo();
            assertTrue(sequenceInfo.sequenceOfMaps);
            assertEquals(ONNXJavaType.INT64,sequenceInfo.mapInfo.keyType);
            assertEquals(ONNXJavaType.FLOAT,sequenceInfo.mapInfo.valueType);

            // try-cast first element in sequence to map/dictionary type
            Map<Long,Float> map = (Map<Long,Float>) ((List<Object>)secondOutput.getValue()).get(0);
            assertEquals(0.25938290, map.get(0L), 1e-6);
            assertEquals(0.40904793, map.get(1L), 1e-6);
            assertEquals(0.33156919, map.get(2L), 1e-6);
        }
    }

    @Test
    public void testModelSequenceOfMapStringFloat() throws ONNXException {
        // test model trained using lightgbm classifier
        // produces 2 named outputs
        //   "label" is a tensor,
        //   "probabilities" is a sequence<map<int64, float>>
        // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py
        String modelPath = resourcePath.resolve("test_sequence_map_string_float.pb").toString();
        try (ONNXEnvironment env = new ONNXEnvironment("testModelSequenceOfMapStringFloat");
             SessionOptions options = new SessionOptions();
             ONNXAllocator allocator = new ONNXAllocator();
             ONNXSession session = env.createSession(modelPath, allocator, options)) {

            List<NodeInfo> outputInfos = session.getOutputInfo();
            ValueInfo firstOutputInfo = outputInfos.get(0).getInfo();
            ValueInfo secondOutputInfo = outputInfos.get(1).getInfo();
            assertTrue(firstOutputInfo instanceof TensorInfo);
            assertTrue(secondOutputInfo instanceof SequenceInfo);
            assertEquals(ONNXJavaType.STRING,((TensorInfo)firstOutputInfo).type);

            List<ONNXTensor> container = new ArrayList<>();
            long[] shape = new long[] { 1, 2 };
            float[] flatInput = new float[] {5.8f, 2.8f};
            Object tensorIn = ONNXUtil.reshape(flatInput,shape);
            ONNXTensor ov = allocator.createTensor(tensorIn);
            container.add(ov);

            List<ONNXValue> outputs = session.score(container);
            assertEquals(2,outputs.size());

            // first output is a tensor containing label
            ONNXValue firstOutput = outputs.get(0);
            assertTrue(firstOutput instanceof ONNXTensor);

            // try-cast as a tensor
            String[] labelOutput = (String[]) firstOutput.getValue();

            // Label 1 should have highest probability
            assertEquals("1", labelOutput[0]);
            assertEquals(1,labelOutput.length);

            // second output is a sequence<map<int64, float>>
            // try-cast to an sequence of NOV
            ONNXValue secondOutput = outputs.get(1);
            assertTrue(secondOutput instanceof ONNXSequence);
            SequenceInfo sequenceInfo = ((ONNXSequence)secondOutput).getInfo();
            assertTrue(sequenceInfo.sequenceOfMaps);
            assertEquals(ONNXJavaType.STRING,sequenceInfo.mapInfo.keyType);
            assertEquals(ONNXJavaType.FLOAT,sequenceInfo.mapInfo.valueType);

            // try-cast first element in sequence to map/dictionary type
            Map<String,Float> map = (Map<String,Float>) ((List<Object>)secondOutput.getValue()).get(0);
            assertEquals(0.25938290, map.get("0"), 1e-6);
            assertEquals(0.40904793, map.get("1"), 1e-6);
            assertEquals(0.33156919, map.get("2"), 1e-6);
        }
    }

    /**
     * Carrier tuple for the squeeze net model.
     */
    private static class SqueezeNetTuple {
        public final ONNXEnvironment env;
        public final ONNXAllocator allocator;
        public final ONNXSession session;
        public final float[] inputData;
        public final float[] outputData;

        public SqueezeNetTuple(ONNXEnvironment env, ONNXAllocator allocator, ONNXSession session, float[] inputData, float[] outputData) {
            this.env = env;
            this.allocator = allocator;
            this.session = session;
            this.inputData = inputData;
            this.outputData = outputData;
        }
    }

    private static SqueezeNetTuple openSessionSqueezeNet() throws ONNXException {
        return openSessionSqueezeNet(-1);
    }

    private static SqueezeNetTuple openSessionSqueezeNet(int cudaDeviceId) throws ONNXException {
        Path squeezeNet = resourcePath.resolve("squeezenet.onnx");
        String modelPath = squeezeNet.toString();
        ONNXEnvironment env = new ONNXEnvironment();
        ONNXAllocator allocator = new ONNXAllocator();
        SessionOptions options = new SessionOptions();
        if (cudaDeviceId != -1) {
            options.addCUDA(cudaDeviceId);
        }
        ONNXSession session = env.createSession(modelPath,allocator,options);
        float[] inputData = loadTensorFromFile(resourcePath.resolve("bench.in"));
        float[] expectedOutput = loadTensorFromFile(resourcePath.resolve("bench.expected_out"));
        return new SqueezeNetTuple(env, allocator, session, inputData, expectedOutput);
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

}
