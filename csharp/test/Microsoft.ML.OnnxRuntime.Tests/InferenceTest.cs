// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    public class InferenceTest
    {
        private const string module = "onnxruntime.dll";
        private const string propertiesFile = "Properties.txt";

        [Fact]
        public void TestSessionOptions()
        {
            using (SessionOptions opt = new SessionOptions())
            {
                Assert.NotNull(opt);

                // check default values of the properties
                Assert.True(opt.EnableSequentialExecution);
                Assert.True(opt.EnableMemoryPattern);
                Assert.False(opt.EnableProfiling);
                Assert.Equal("onnxruntime_profile_", opt.ProfileOutputPathPrefix);
                Assert.True(opt.EnableCpuMemArena);
                Assert.Equal("", opt.LogId);
                Assert.Equal(LogLevel.Verbose, opt.LogVerbosityLevel);
                Assert.Equal(0, opt.ThreadPoolSize);
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_BASIC, opt.GraphOptimizationLevel);

                // try setting options 
                opt.EnableSequentialExecution = false;
                Assert.False(opt.EnableSequentialExecution);

                opt.EnableMemoryPattern = false;
                Assert.False(opt.EnableMemoryPattern);

                opt.EnableProfiling = true;
                Assert.True(opt.EnableProfiling);
                Assert.Equal("onnxruntime_profile_", opt.ProfileOutputPathPrefix);

                opt.ProfileOutputPathPrefix = "Ort_P_";
                Assert.Equal("Ort_P_", opt.ProfileOutputPathPrefix);

                opt.EnableCpuMemArena = false;
                Assert.False(opt.EnableCpuMemArena);

                opt.LogId = "MyLogId";
                Assert.Equal("MyLogId", opt.LogId);

                opt.LogVerbosityLevel = LogLevel.Error;
                Assert.Equal(LogLevel.Error, opt.LogVerbosityLevel);

                opt.ThreadPoolSize = 4;
                Assert.Equal(4, opt.ThreadPoolSize);

                opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, opt.GraphOptimizationLevel);
                
                Assert.Throws<OnnxRuntimeException>(() => { opt.GraphOptimizationLevel = (GraphOptimizationLevel)10; });
            }
        }

        [Fact]
        public void TestRunOptions()
        {
            using (var opt = new RunOptions())
            {
                Assert.NotNull(opt);

                //verify default options
                Assert.False(opt.Terminate);
                Assert.Equal(LogLevel.Verbose, opt.LogVerbosityLevel);
                Assert.Equal("", opt.LogTag);

                // try setting options
                opt.Terminate = true;
                Assert.True(opt.Terminate);

                opt.LogVerbosityLevel = LogLevel.Error;
                Assert.Equal(LogLevel.Error, opt.LogVerbosityLevel);

                opt.LogTag = "MyLogTag";
                Assert.Equal("MyLogTag", opt.LogTag);
            }
        }

        [Fact]
        public void CanCreateAndDisposeSessionWithModelPath()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                Assert.NotNull(session);
                Assert.NotNull(session.InputMetadata);
                Assert.Equal(1, session.InputMetadata.Count); // 1 input node
                Assert.True(session.InputMetadata.ContainsKey("data_0")); // input node name
                Assert.Equal(typeof(float), session.InputMetadata["data_0"].ElementType);
                Assert.True(session.InputMetadata["data_0"].IsTensor);
                var expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                Assert.Equal(expectedInputDimensions.Length, session.InputMetadata["data_0"].Dimensions.Length);
                for (int i = 0; i < expectedInputDimensions.Length; i++)
                {
                    Assert.Equal(expectedInputDimensions[i], session.InputMetadata["data_0"].Dimensions[i]);
                }

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output node
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output node name
                Assert.Equal(typeof(float), session.OutputMetadata["softmaxout_1"].ElementType);
                Assert.True(session.OutputMetadata["softmaxout_1"].IsTensor);
                var expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                Assert.Equal(expectedOutputDimensions.Length, session.OutputMetadata["softmaxout_1"].Dimensions.Length);
                for (int i = 0; i < expectedOutputDimensions.Length; i++)
                {
                    Assert.Equal(expectedOutputDimensions[i], session.OutputMetadata["softmaxout_1"].Dimensions[i]);
                }
            }
        }

        [Theory]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, true)]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, false)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, true)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, false)]
        private void CanRunInferenceOnAModel(GraphOptimizationLevel graphOptimizationLevel, bool disableSequentialExecution)
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");

            // Set the graph optimization level for this session.
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = graphOptimizationLevel;
            if (disableSequentialExecution) options.EnableSequentialExecution = false;

            using (var session = new InferenceSession(modelPath, options))
            {
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();

                float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model

                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run the inference
                using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                {
                    validateRunResults(results);
                }

                // Run Inference with RunOptions
                using (var runOptions = new RunOptions())
                {
                    runOptions.LogTag = "CsharpTest";
                    runOptions.Terminate = true;
                    runOptions.LogVerbosityLevel = LogLevel.Error;
                    IReadOnlyCollection<string> outputNames = session.OutputMetadata.Keys.ToList();

                    using (var results = session.Run(container, outputNames, runOptions))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        validateRunResults(results);
                    }
                }
            }
        }

        private void validateRunResults(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
            // validate the results
            foreach (var r in results)
            {
                Assert.Equal(1, results.Count);
                Assert.Equal("softmaxout_1", r.Name);

                var resultTensor = r.AsTensor<float>();
                int[] expectedDimensions = { 1, 1000, 1, 1 };  // hardcoded for now for the test data
                Assert.Equal(expectedDimensions.Length, resultTensor.Rank);

                var resultDimensions = resultTensor.Dimensions;
                for (int i = 0; i < expectedDimensions.Length; i++)
                {
                    Assert.Equal(expectedDimensions[i], resultDimensions[i]);
                }

                var resultArray = r.AsTensor<float>().ToArray();
                Assert.Equal(expectedOutput.Length, resultArray.Length);
                Assert.Equal(expectedOutput, resultArray, new floatComparer());
            }
        }


        [Fact]
        private void ThrowWrongInputName()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("wrong_name", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.Contains("Invalid Feed Input", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowWrongInputType()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            int[] inputDataInt = inputData.Select(x => (int)x).ToArray();
            var tensor = new DenseTensor<int>(inputDataInt, inputMeta["data_0"].Dimensions);
            container.Add(NamedOnnxValue.CreateFromTensor<int>("data_0", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            var msg = ex.ToString().Substring(0, 101);
            // TODO: message is diff in LInux. Use substring match
            Assert.Equal("Microsoft.ML.OnnxRuntime.OnnxRuntimeException: [ErrorCode:InvalidArgument] Unexpected input data type", msg);
            session.Dispose();
        }

        [Fact]
        private void ThrowExtraInputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            var nov1 = NamedOnnxValue.CreateFromTensor<float>("data_0", tensor);
            var nov2 = NamedOnnxValue.CreateFromTensor<float>("extra", tensor);
            container.Add(nov1);
            container.Add(nov2);
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.StartsWith("[ErrorCode:InvalidArgument] Invalid Feed Input Name", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void TestMultiThreads()
        {
            var numThreads = 10;
            var loop = 10;
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var expectedOut = tuple.Item4;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
            var tasks = new Task[numThreads];
            for (int i = 0; i < numThreads; i++)
            {
                tasks[i] = Task.Factory.StartNew(() =>
                {
                    for (int j = 0; j < loop; j++)
                    {
                        var resnov = session.Run(container);
                        var res = resnov.ToArray()[0].AsTensor<float>().ToArray<float>();
                        Assert.Equal(res, expectedOut, new floatComparer());
                    }
                });
            };
            Task.WaitAll(tasks);
            session.Dispose();
        }

        [x64Fact]
        private void TestPreTrainedModelsOpset7And8()
        {
            var skipModels = new List<String>() {
                "mxnet_arcface",  // Model not supported by CPU execution provider
                "tf_inception_v2",  // TODO: Debug failing model, skipping for now
                "fp16_inception_v1",  // 16-bit float not supported type in C#.
                "fp16_shufflenet",  // 16-bit float not supported type in C#.
                "fp16_tiny_yolov2" };  // 16-bit float not supported type in C#.

            var disableContribOpsEnvVar = Environment.GetEnvironmentVariable("DisableContribOps");
            var isContribOpsDisabled = (disableContribOpsEnvVar != null) ? disableContribOpsEnvVar.Equals("ON") : false;
            if (isContribOpsDisabled)
            {
                skipModels.Add("test_tiny_yolov2");
            }

            var opsets = new[] { "opset7", "opset8" };
            var modelsDir = GetTestModelsDir();
            foreach (var opset in opsets)
            {
                var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opset));
                foreach (var modelDir in modelRoot.EnumerateDirectories())
                {
                    String onnxModelFileName = null;

                    if (skipModels.Contains(modelDir.Name))
                        continue;

                    try
                    {
                        var onnxModelNames = modelDir.GetFiles("*.onnx");
                        if (onnxModelNames.Length > 1)
                        {
                            // TODO remove file "._resnet34v2.onnx" from test set
                            bool validModelFound = false;
                            for (int i = 0; i < onnxModelNames.Length; i++)
                            {
                                if (onnxModelNames[i].Name != "._resnet34v2.onnx")
                                {
                                    onnxModelNames[0] = onnxModelNames[i];
                                    validModelFound = true;
                                }
                            }

                            if (!validModelFound)
                            {
                                var modelNamesList = string.Join(",", onnxModelNames.Select(x => x.ToString()));
                                throw new Exception($"Opset {opset}: Model {modelDir}. Can't determine model file name. Found these :{modelNamesList}");
                            }
                        }

                        onnxModelFileName = Path.Combine(modelsDir, opset, modelDir.Name, onnxModelNames[0].Name);
                        using (var session = new InferenceSession(onnxModelFileName))
                        {
                            var inMeta = session.InputMetadata;
                            var innodepair = inMeta.First();
                            var innodename = innodepair.Key;
                            var innodedims = innodepair.Value.Dimensions;
                            for (int i = 0; i < innodedims.Length; i++)
                            {
                                if (innodedims[i] < 0)
                                    innodedims[i] = -1 * innodedims[i];
                            }

                            var testRoot = new DirectoryInfo(Path.Combine(modelsDir, opset, modelDir.Name));
                            var testData = testRoot.EnumerateDirectories("test_data*").First();
                            var dataIn = LoadTensorFromFilePb(Path.Combine(modelsDir, opset, modelDir.Name, testData.ToString(), "input_0.pb"));
                            var dataOut = LoadTensorFromFilePb(Path.Combine(modelsDir, opset, modelDir.Name, testData.ToString(), "output_0.pb"));
                            var tensorIn = new DenseTensor<float>(dataIn, innodedims);
                            var nov = new List<NamedOnnxValue>();
                            nov.Add(NamedOnnxValue.CreateFromTensor<float>(innodename, tensorIn));
                            using (var resnov = session.Run(nov))
                            {
                                var res = resnov.ToArray()[0].AsTensor<float>().ToArray<float>();
                                Assert.Equal(res, dataOut, new floatComparer());
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        var msg = $"Opset {opset}: Model {modelDir}: ModelFile = {onnxModelFileName} error = {ex.Message}";
                        throw new Exception(msg);
                    }
                } //model
            } //opset
        }

        [Fact]
        private void TestModelInputFloat()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT.pb");

            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<float>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact(Skip = "Boolean tensor not supported yet")]
        private void TestModelInputBOOL()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_BOOL.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<bool>(new bool[] { true, false, true, false, true }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<bool>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT32.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<int>(new int[] { 1, -2, -3, int.MinValue, int.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<int>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputDOUBLE()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_DOUBLE.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<double>(new double[] { 1.0, 2.0, -3.0, 5, 5 }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<double>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }

        }

        [Fact(Skip = "String tensor not supported yet")]
        private void TestModelInputSTRING()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_STRING.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[] { "a", "c", "d", "z", "f" }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<string>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact(Skip = "Int8 not supported yet")]
        private void TestModelInputINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT8.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<sbyte>(new sbyte[] { 1, 2, -3, sbyte.MinValue, sbyte.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<sbyte>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT8.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<byte>(new byte[] { 1, 2, 3, byte.MinValue, byte.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<byte>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt16>(new UInt16[] { 1, 2, 3, UInt16.MinValue, UInt16.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Int16>(new Int16[] { 1, 2, 3, Int16.MinValue, Int16.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<Int16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT64.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Int64>(new Int64[] { 1, 2, -3, Int64.MinValue, Int64.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<Int64>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT32.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt32>(new UInt32[] { 1, 2, 3, UInt32.MinValue, UInt32.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt32>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }
        [Fact]
        private void TestModelInputUINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT64.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt64>(new UInt64[] { 1, 2, 3, UInt64.MinValue, UInt64.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt64>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact(Skip = "FLOAT16 not available in C#")]
        private void TestModelInputFLOAT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<float>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelSequenceOfMapIntFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_sequence_map_int_float.pb");
            using (var session = new InferenceSession(modelPath))
            {

                var outMeta = session.OutputMetadata;
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outMeta["label"].OnnxValueType);
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outMeta["probabilities"].OnnxValueType);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode1 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode1.Name);

                    // try-cast as a tensor
                    var outLabelTensor = outNode1.AsTensor<Int64>();

                    // Label 1 should have highest probaility
                    Assert.Equal(1, outLabelTensor[0]);

                    // second output is a sequence<map<int64, float>>
                    // try-cast to an sequence of NOV
                    var outNode2 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode2.Name);

                    // try-cast to an sequence of NOV
                    var seq = outNode2.AsEnumerable<NamedOnnxValue>();

                    // try-cast first element in sequence to map/dictionary type
                    if (System.Environment.Is64BitProcess)
                    {
                        var map = seq.First().AsDictionary<Int64, float>();
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                    else // 32-bit
                    {
                        var map = seq.First().AsDictionary<long, float>();
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                }
            }
        }

        [Fact]
        private void TestModelSequenceOfMapStringFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_sequence_map_string_float.pb");

            using (var session = new InferenceSession(modelPath))
            {
                var outMeta = session.OutputMetadata;
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outMeta["label"].OnnxValueType);
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outMeta["probabilities"].OnnxValueType);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode1 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode1.Name);

                    // try-cast as a tensor
                    var outLabelTensor = outNode1.AsTensor<string>();

                    // Label 1 should have highest probaility
                    Assert.Equal("1", outLabelTensor[0]);

                    // second output is a sequence<map<int64, float>>
                    // try-cast to an sequence of NOV
                    var outNode2 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode2.Name);

                    // try-cast to an sequence of NOV
                    var seq = outNode2.AsEnumerable<NamedOnnxValue>();

                    // try-cast first element in sequence to map/dictionary type
                    var map = seq.First().AsDictionary<string, float>();
                    //verify values are valid
                    Assert.Equal(0.25938290, map["0"], 6);
                    Assert.Equal(0.40904793, map["1"], 6);
                    Assert.Equal(0.33156919, map["2"], 6);
                }
            }
        }

        [Fact]
        private void TestModelSerialization()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            string modelOutputPath = Path.Combine(Directory.GetCurrentDirectory(), "optimized-squeezenet.onnx");
            // Set the optimized model file path to assert that no exception are thrown.
            SessionOptions options = new SessionOptions();
            options.OptimizedModelFilePath = modelOutputPath;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
            var session = new InferenceSession(modelPath, options);
            Assert.NotNull(session);
            Assert.True(File.Exists(modelOutputPath));
        }

        [GpuFact]
        private void TestGpu()
        {
            var gpu = Environment.GetEnvironmentVariable("TESTONGPU");
            var tuple = OpenSessionSqueezeNet(0); // run on deviceID 0
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");

            using (var session = tuple.Item1)
            {
                var inputData = tuple.Item2;
                var tensor = tuple.Item3;
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
                var res = session.Run(container);
                var resultArray = res.First().AsTensor<float>().ToArray();
                Assert.Equal(expectedOutput, resultArray, new floatComparer());
            }
        }


        [DllImport("kernel32", SetLastError = true)]
        static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32", CharSet = CharSet.Ansi)]
        static extern UIntPtr GetProcAddress(IntPtr hModule, string procName);

        [Fact]
        private void VerifyNativeMethodsExist()
        {
            // Check for  external API changes
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;
            var entryPointNames = new[]{
            "OrtCreateEnv","OrtReleaseEnv",
            "OrtGetErrorCode","OrtGetErrorMessage", "OrtReleaseStatus",
            "OrtCreateSession","OrtRun",
            "OrtSessionGetInputCount", "OrtSessionGetOutputCount","OrtSessionGetInputName","OrtSessionGetOutputName",
            "OrtSessionGetInputTypeInfo", "OrtSessionGetOutputTypeInfo","OrtReleaseSession",
            "OrtCreateSessionOptions","OrtCloneSessionOptions",
            "OrtEnableSequentialExecution","OrtDisableSequentialExecution","OrtEnableProfiling","OrtDisableProfiling",
            "OrtEnableMemPattern","OrtDisableMemPattern","OrtEnableCpuMemArena","OrtDisableCpuMemArena",
            "OrtSetSessionLogId","OrtSetSessionLogVerbosityLevel","OrtSetSessionThreadPoolSize","OrtSetSessionGraphOptimizationLevel",
            "OrtSetOptimizedModelFilePath", "OrtSessionOptionsAppendExecutionProvider_CPU",
            "OrtCreateRunOptions", "OrtReleaseRunOptions", "OrtRunOptionsSetRunLogVerbosityLevel", "OrtRunOptionsSetRunTag",
            "OrtRunOptionsGetRunLogVerbosityLevel", "OrtRunOptionsGetRunTag","OrtRunOptionsEnableTerminate", "OrtRunOptionsDisableTerminate",
            "OrtCreateAllocatorInfo","OrtCreateCpuAllocatorInfo",
            "OrtCreateDefaultAllocator","OrtAllocatorFree","OrtAllocatorGetInfo",
            "OrtCreateTensorWithDataAsOrtValue","OrtGetTensorMutableData", "OrtReleaseAllocatorInfo",
            "OrtCastTypeInfoToTensorInfo","OrtGetTensorTypeAndShape","OrtGetTensorElementType","OrtGetDimensionsCount",
            "OrtGetDimensions","OrtGetTensorShapeElementCount","OrtReleaseValue"};

            var hModule = LoadLibrary(module);
            foreach (var ep in entryPointNames)
            {
                var x = GetProcAddress(hModule, ep);
                Assert.False(x == UIntPtr.Zero, $"Entrypoint {ep} not found in module {module}");
            }
        }

        static string GetTestModelsDir()
        {
            // get build directory, append downloaded models location
            var cwd = Directory.GetCurrentDirectory();
            var props = File.ReadAllLines(Path.Combine(cwd, propertiesFile));
            var modelsRelDir = Path.Combine(props[0].Split('=')[1].Trim());
            var modelsDir = Path.Combine(cwd, @"../../..", modelsRelDir, "models");
            return modelsDir;
        }

        static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        static float[] LoadTensorFromFilePb(string filename)
        {
            var file = File.OpenRead(filename);
            var tensor = Onnx.TensorProto.Parser.ParseFrom(file);
            file.Close();
            var raw = tensor.RawData.ToArray();
            var floatArr = new float[raw.Length / sizeof(float)];
            Buffer.BlockCopy(raw, 0, floatArr, 0, raw.Length);
            return floatArr;
        }

        static Tuple<InferenceSession, float[], DenseTensor<float>, float[]> OpenSessionSqueezeNet(int? cudaDeviceId = null)
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            var session = (cudaDeviceId.HasValue)
                ? new InferenceSession(modelPath, SessionOptions.MakeSessionOptionWithCudaProvider(cudaDeviceId.Value))
                : new InferenceSession(modelPath);
            float[] inputData = LoadTensorFromFile(@"bench.in");
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
            var inputMeta = session.InputMetadata;
            var tensor = new DenseTensor<float>(inputData, inputMeta["data_0"].Dimensions);
            return new Tuple<InferenceSession, float[], DenseTensor<float>, float[]>(session, inputData, tensor, expectedOutput);
        }

        class floatComparer : IEqualityComparer<float>
        {
            private float atol = 1e-3f;
            private float rtol = 1.7e-2f;

            public bool Equals(float x, float y)
            {
                return Math.Abs(x - y) <= (atol + rtol * Math.Abs(y));
            }
            public int GetHashCode(float x)
            {
                return 0;
            }
        }

        private class GpuFact : FactAttribute
        {
            public GpuFact()
            {
                var testOnGpu = System.Environment.GetEnvironmentVariable("TESTONGPU");
                if (testOnGpu == null || !testOnGpu.Equals("ON"))
                {
                    Skip = "GPU testing not enabled";
                }
            }
        }

        private class x64Fact : FactAttribute
        {
            public x64Fact()
            {
                if (System.Environment.Is64BitProcess == false)
                {
                    Skip = "Not 64-bit process";
                }
            }
        }
    }
}
