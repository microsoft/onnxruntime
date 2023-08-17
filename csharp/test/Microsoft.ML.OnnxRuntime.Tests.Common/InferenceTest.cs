// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;

using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

// This runs in a separate package built from EndToEndTests
// and for this reason it can not refer to non-public members
// of Onnxruntime package
namespace Microsoft.ML.OnnxRuntime.Tests
{
    // This is to make sure it does not run in parallel with OrtEnvTests
    // or any other test class within the same collection
    [Collection("Ort Inference Tests")]
    public partial class InferenceTest
    {
        private readonly ITestOutputHelper output;

        public InferenceTest(ITestOutputHelper o)
        {
            this.output = o;
        }

        [Fact(DisplayName = "TestSessionOptions")]
        public void TestSessionOptions()
        {
            // get instance to setup logging
            var ortEnvInstance = OrtEnv.Instance();

            using (SessionOptions opt = new SessionOptions())
            {
                Assert.NotNull(opt);

                // check default values of the properties
                Assert.Equal(ExecutionMode.ORT_SEQUENTIAL, opt.ExecutionMode);
                Assert.True(opt.EnableMemoryPattern);
                Assert.False(opt.EnableProfiling);
                Assert.Equal("onnxruntime_profile_", opt.ProfileOutputPathPrefix);
                Assert.True(opt.EnableCpuMemArena);
                Assert.Equal("", opt.LogId);
                Assert.Equal(0, opt.LogVerbosityLevel);
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, opt.LogSeverityLevel);
                Assert.Equal(0, opt.IntraOpNumThreads);
                Assert.Equal(0, opt.InterOpNumThreads);
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_ALL, opt.GraphOptimizationLevel);

                // try setting options
                opt.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                Assert.Equal(ExecutionMode.ORT_PARALLEL, opt.ExecutionMode);

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

                opt.LogVerbosityLevel = 1;
                Assert.Equal(1, opt.LogVerbosityLevel);

                opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, opt.LogSeverityLevel);

                opt.IntraOpNumThreads = 4;
                Assert.Equal(4, opt.IntraOpNumThreads);

                opt.InterOpNumThreads = 4;
                Assert.Equal(4, opt.InterOpNumThreads);

                opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, opt.GraphOptimizationLevel);

                Assert.Throws<OnnxRuntimeException>(() => { opt.GraphOptimizationLevel = (GraphOptimizationLevel)10; });

                opt.AddSessionConfigEntry("key", "value");

                var ex = Assert.Throws<OnnxRuntimeException>(() => { opt.AddSessionConfigEntry("", "invalid key"); });
                Assert.Contains("[ErrorCode:InvalidArgument] Config key is empty", ex.Message);

                // SessionOptions.RegisterOrtExtensions can be manually tested by referencing the
                // Microsoft.ML.OnnxRuntime.Extensions nuget package. After that is done, this should not throw.                
                ex = Assert.Throws<OnnxRuntimeException>(() => { opt.RegisterOrtExtensions(); });
                Assert.Contains("Microsoft.ML.OnnxRuntime.Extensions NuGet package must be referenced", ex.Message);

#if USE_CUDA
                opt.AppendExecutionProvider_CUDA(0);
#endif

#if USE_DML
                // Explicitly set dll probe path so that the (potentially) stale system DirectML.dll
                // doesn't get loaded by the test process when it is eventually delay loaded by onnruntime.dll
                // The managed tests binary path already contains the right DirectML.dll, so use that

                var directml_dll_path = AppDomain.CurrentDomain.BaseDirectory;
                SetDllDirectory(directml_dll_path);
                
                try
                {
                    opt.AppendExecutionProvider_DML(0);
                }
                catch (OnnxRuntimeException ortException)
                {
                    // if we run on a CI machine with the incorrect hardware we might get an error due to that.
                    // allow that as the call made it through to the DML EP so the C# layer is working correctly. 
                    // any other exception type or error message is considered a failure.
                    Assert.Contains("The specified device interface or feature level is not supported on this system.",
                                    ortException.Message);
                }

                // Restore the default dll search order
                SetDllDirectory(null);
#endif

#if USE_DNNL
                opt.AppendExecutionProvider_Dnnl(0);
#endif

#if USE_MIGRAPHX
                opt.AppendExecutionProvider_MIGraphX(0);
#endif

#if USE_NNAPI
                opt.AppendExecutionProvider_Nnapi(0);
#endif

#if USE_TVM
                opt.AppendExecutionProvider_Tvm("Vulkan -device=amd_apu");
#endif

#if USE_OPENVINO
                opt.AppendExecutionProvider_OpenVINO();
#endif

#if USE_ROCM
                opt.AppendExecutionProvider_ROCm(0);
#endif

#if USE_TENSORRT
                opt.AppendExecutionProvider_Tensorrt(0);
#endif
#if USE_XNNPACK
                opt.AppendExecutionProvider("XNNPACK");
#else
                ex = Assert.Throws<OnnxRuntimeException>(() => { opt.AppendExecutionProvider("XNNPACK"); });
                Assert.Contains("XNNPACK execution provider is not supported in this build", ex.Message);
#endif
#if USE_SNPE
                opt.AppendExecutionProvider("SNPE");
#else
                ex = Assert.Throws<OnnxRuntimeException>(() => { opt.AppendExecutionProvider("SNPE"); });
                Assert.Contains("SNPE execution provider is not supported in this build", ex.Message);
#endif
#if USE_QNN
                opt.AppendExecutionProvider("QNN");
#else
                ex = Assert.Throws<OnnxRuntimeException>(() => { opt.AppendExecutionProvider("QNN"); });
                Assert.Contains("QNN execution provider is not supported in this build", ex.Message);
#endif

                opt.AppendExecutionProvider_CPU(1);
            }
        }

#if! __MOBILE__
        // Use to set dll probe path so that the right dll(s) is loaded by the test process
        // Invoke only to specify Windows specific EPs' dll locations explicitly
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]

        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);
#else
        static bool SetDllDirectory(string lpPathName)
        {
            throw new NotSupportedException();
        }
#endif

        [Fact(DisplayName = "TestRunOptions")]
        public void TestRunOptions()
        {
            using (var opt = new RunOptions())
            {
                Assert.NotNull(opt);

                //verify default options
                Assert.False(opt.Terminate);
                Assert.Equal(0, opt.LogVerbosityLevel);
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, opt.LogSeverityLevel);
                Assert.Equal("", opt.LogId);

                // try setting options
                opt.Terminate = true;
                Assert.True(opt.Terminate);

                opt.LogVerbosityLevel = 1;
                Assert.Equal(1, opt.LogVerbosityLevel);

                opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, opt.LogSeverityLevel);

                opt.LogId = "MyLogTag";
                Assert.Equal("MyLogTag", opt.LogId);

                opt.AddRunConfigEntry("key", "value");

                var ex = Assert.Throws<OnnxRuntimeException>(() => { opt.AddRunConfigEntry("", "missing key"); });
                Assert.Contains("[ErrorCode:InvalidArgument] Config key is empty", ex.Message);
            }
        }

        [Fact(DisplayName = "TestThreadingOptions")]
        public void TestThreadingOptions()
        {
            using (var opt = new OrtThreadingOptions())
            {
                Assert.NotNull(opt);

                //verify default options
                opt.GlobalSpinControl = false;
                opt.GlobalInterOpNumThreads = 1;
                opt.GlobalIntraOpNumThreads = 1;
                opt.SetGlobalDenormalAsZero();
            }
        }

        [Fact(DisplayName = "CanCreateAndDisposeSessionWithModel")]
        public void CanCreateAndDisposeSessionWithModel()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            using (var session = new InferenceSession(model))
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

        [Theory(DisplayName = "CanRunInferenceOnAModel")]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, true)]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, false)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, true)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, false)]
        private void CanRunInferenceOnAModel(GraphOptimizationLevel graphOptimizationLevel, bool enableParallelExecution)
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                // Set the graph optimization level for this session.
                SessionOptions options = new SessionOptions();
                cleanUp.Add(options);
                options.GraphOptimizationLevel = graphOptimizationLevel;
                if (enableParallelExecution) options.ExecutionMode = ExecutionMode.ORT_PARALLEL;

                var session = new InferenceSession(model, options);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                var outputMeta = session.OutputMetadata;
                var container = new List<NamedOnnxValue>();

                float[] expectedOutput = TestDataLoader.LoadTensorFromEmbeddedResource("bench.expected_out");
                int[] expectedDimensions = { 1, 1000, 1, 1 };  // hardcoded for now for the test data
                ReadOnlySpan<int> expectedOutputDimensions = expectedDimensions;
                string[] expectedOutputNames = new string[] { "softmaxout_1" };

                float[] inputData = TestDataLoader.LoadTensorFromEmbeddedResource("bench.in"); // this is the data for only one input tensor for this model

                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run inference with named inputs and outputs created with in Run()
                using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                {
                    ValidateRunResults(results);
                }

                // Run inference with named inputs, outputs created with in Run() and RunOptions
                using (var runOptions = new RunOptions())
                {
                    runOptions.LogId = "CsharpTest";
                    runOptions.Terminate = false;  // TODO: Test terminate = true, it currently crashes
                    runOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
                    IReadOnlyCollection<string> outputNames = session.OutputMetadata.Keys.ToList();

                    using (var results = session.Run(container, outputNames, runOptions))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        ValidateRunResults(results);
                    }
                }

                // Run inference with pinned inputs and outputs created with in Run()
                using (var pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var inputNames = container.Select(i => i.Name).ToArray();
                    pinnedInputs.AddRange(container.Select(i => FixedBufferOnnxValue.CreateFromTensor(i.AsTensor<float>())));

                    // output names not specified
                    using (var results = session.Run(inputNames, pinnedInputs))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        ValidateRunResults(results);
                    }

                    // output names specified explicitly
                    using (var results = session.Run(inputNames, pinnedInputs, expectedOutputNames))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        ValidateRunResults(results);
                    }
                }

                // Run inference with outputs pinned from buffers
                using (var pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>())
                using (var pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var memInfo = OrtMemoryInfo.DefaultInstance; // CPU

                    // Create inputs
                    Assert.Single(inputMeta.Keys);
                    var inputNames = inputMeta.Keys.ToArray();
                    var inputName = inputNames[0];
                    Assert.Equal(typeof(float), inputMeta[inputName].ElementType);
                    Assert.True(inputMeta[inputName].IsTensor);
                    var longShape = Array.ConvertAll<int, long>(inputMeta[inputName].Dimensions, Convert.ToInt64);
                    var byteSize = ShapeUtils.GetSizeForShape(longShape);
                    pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, inputData,
                        TensorElementType.Float, longShape, byteSize));


                    // Prepare output buffer
                    Assert.Single(outputMeta.Keys);
                    var outputNames = outputMeta.Keys.ToArray();
                    var outputName = outputNames[0];
                    Assert.Equal(typeof(float), outputMeta[outputName].ElementType);
                    Assert.True(outputMeta[outputName].IsTensor);
                    longShape = Array.ConvertAll<int, long>(outputMeta[outputName].Dimensions, Convert.ToInt64);
                    byteSize = ShapeUtils.GetSizeForShape(longShape);
                    float[] outputBuffer = new float[expectedOutput.Length];
                    pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, outputBuffer,
                        TensorElementType.Float, longShape, byteSize));

                    session.Run(inputNames, pinnedInputs, outputNames, pinnedOutputs);
                    Assert.Equal(expectedOutput, outputBuffer, new FloatComparer());
                }

                // Run inference with named inputs and named outputs
                {
                    // correct pre-allocated outputs
                    var expectedOutputValues = new List<NamedOnnxValue>()
                    {
                        NamedOnnxValue.CreateFromTensor("softmaxout_1", new DenseTensor<float>(expectedOutputDimensions))
                    };
                    session.Run(container, expectedOutputValues);
                    ValidateRunResultData(expectedOutputValues[0].AsTensor<float>(), expectedOutput, expectedDimensions);
                }

                // Run inference with pinned inputs and named outputs
                using (var pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var inputNames = container.Select(i => i.Name).ToArray();
                    pinnedInputs.AddRange(container.Select(i => FixedBufferOnnxValue.CreateFromTensor(i.AsTensor<float>())));

                    // expected inputs and outputs
                    var expectedOutputValues = new List<NamedOnnxValue>()
                    {
                        NamedOnnxValue.CreateFromTensor("softmaxout_1", new DenseTensor<float>(expectedOutputDimensions))
                    };
                    session.Run(inputNames, pinnedInputs, expectedOutputValues);
                    ValidateRunResultData(expectedOutputValues[0].AsTensor<float>(), expectedOutput, expectedDimensions);
                }

                // Run inference with named inputs and pinned outputs
                {
                    // correct pre-allocated outputs
                    using (var pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                    {
                        var outputTensor = new DenseTensor<float>(expectedOutputDimensions);
                        pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromTensor(outputTensor));
                        session.Run(container, expectedOutputNames, pinnedOutputs);
                        ValidateRunResultData(outputTensor, expectedOutput, expectedDimensions);
                    }
                }

                // Run inference with pinned inputs and pinned outputs
                using (DisposableListTest<FixedBufferOnnxValue> pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>(),
                                                            pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var inputNames = container.Select(i => i.Name).ToArray();
                    pinnedInputs.AddRange(container.Select(i => FixedBufferOnnxValue.CreateFromTensor(i.AsTensor<float>())));

                    var outputTensor = new DenseTensor<float>(expectedOutputDimensions);
                    pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromTensor(outputTensor));

                    session.Run(inputNames, pinnedInputs, expectedOutputNames, pinnedOutputs);
                    ValidateRunResultData(outputTensor, expectedOutput, expectedDimensions);
                }
            }
        }

        [Fact(DisplayName = "RunInferenceUsingPreAllocatedOutputsAndDictionary")]
        public void RunInferenceUsingPreAllocatedOutputsAndDictionary()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var runOptions = new RunOptions();
                cleanUp.Add(runOptions);
                var session = new InferenceSession(model);
                cleanUp.Add(session);

                var inputMeta = session.InputMetadata;
                Assert.Single(inputMeta.Keys);
                var inputNames = inputMeta.Keys.ToList().AsReadOnly();
                Assert.Equal(TensorElementType.Float, inputMeta[inputNames[0]].ElementDataType);
                Assert.True(inputMeta[inputNames[0]].IsTensor);
                var inputShape = Array.ConvertAll<int, long>(inputMeta[inputNames[0]].Dimensions, Convert.ToInt64);


                var outputMeta = session.OutputMetadata;
                var expectedOutputNames = new List<string> { "softmaxout_1" }.AsReadOnly();
                Assert.Contains(expectedOutputNames[0], outputMeta.Keys);
                long[] expectedShape = { 1, 1000, 1, 1 };  // hardcoded for the test data

                // this is the data for only one input tensor for this model
                float[] inputData = TestDataLoader.LoadTensorFromEmbeddedResource("bench.in");
                float[] expectedOutput = TestDataLoader.LoadTensorFromEmbeddedResource("bench.expected_out");

                // Allocate input OrtValue on top of the inputData
                // Input should stay pinned for the entire duration of the inference
                var inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(inputData, inputShape);
                cleanUp.Add(inputOrtValue);

                // Create OrtValue and pre-allocate output buffer using the expected output shape
                using (var outputOrtValue = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance,
                    TensorElementType.Float, expectedShape))
                {
                    // Run inference
                    var inputValues = new List<OrtValue> { inputOrtValue }.AsReadOnly();
                    var outputValues = new List<OrtValue> { outputOrtValue }.AsReadOnly();
                    session.Run(runOptions, inputNames, inputValues,
                        expectedOutputNames, outputValues);
                    ValidateRunResult(outputOrtValue, expectedOutput, expectedShape);
                }

                //Let's run this again with an interface that takes a Dictionary of name/OrtValue
                var inputDict = new Dictionary<string, OrtValue>();
                inputDict.Add(inputNames[0], inputOrtValue);
                using (var results = session.Run(runOptions, inputDict, expectedOutputNames))
                {
                    Assert.Single(results);
                    var outputOrtValue = results[0];
                    ValidateRunResult(outputOrtValue, expectedOutput, expectedShape);
                }
            }
        }

        [Fact(DisplayName = "InferenceSessionDisposed")]
        public void InferenceSessionDisposed()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            // Set the graph optimization level for this session.
            using (SessionOptions options = new SessionOptions())
            {
                options.ProfileOutputPathPrefix = "Ort_P_";
                options.EnableProfiling = true;
                using (var session = new InferenceSession(model, options))
                {
                    var inputMeta = session.InputMetadata;
                    var container = new List<NamedOnnxValue>();

                    float[] inputData = TestDataLoader.LoadTensorFromEmbeddedResource("bench.in"); // this is the data for only one input tensor for this model

                    foreach (var name in inputMeta.Keys)
                    {
                        Assert.Equal(typeof(float), inputMeta[name].ElementType);
                        Assert.True(inputMeta[name].IsTensor);
                        var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }

                    // Run inference with named inputs and outputs created with in Run()
                    using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        ValidateRunResults(results);
                    }

                    string profile_file = session.EndProfiling();

                    // Profile file should have the output path prefix in it
                    Assert.Contains("Ort_P_", profile_file);
                }
            }
        }

        [Fact(DisplayName = "InferenceSessionGetProfilingStartTimeNs")]
        public void InferenceSessionGetProfilingStartTimeNs()
        {
            ulong getSingleSessionProfilingStartTime()
            {
                ulong startTime = 0;
                using (SessionOptions options = new SessionOptions())
                {
                    options.EnableProfiling = true;

                    var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

                    using (var session = new InferenceSession(model, options))
                    {
                        startTime = session.ProfilingStartTimeNs;
                    }
                }
                return startTime;
            }

            // Get profiling's start time
            ulong ProfilingStartTime = getSingleSessionProfilingStartTime();

            // Check the profiling's start time has been updated
            Assert.True(ProfilingStartTime != 0);
        }

        [Fact(DisplayName = "SessionOptionsFreeDimensionOverrides")]
        public void SessionOptionsFreeDimensionOverrides()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("abs_free_dimensions.onnx");

            // By Name
            using (SessionOptions options = new SessionOptions())
            {
                options.AddFreeDimensionOverrideByName("Dim1", 4);
                options.AddFreeDimensionOverrideByName("Dim2", 6);

                using (var session = new InferenceSession(model, options))
                {
                    var inputMetadata = session.InputMetadata;
                    var dims = inputMetadata["x"].Dimensions;
                    Assert.Equal(3, dims.Length);
                    Assert.Equal(4, dims[0]);
                    Assert.Equal(6, dims[1]);
                    Assert.Equal(5, dims[2]);
                }
            }

            // By Denotation
            using (SessionOptions options = new SessionOptions())
            {
                options.AddFreeDimensionOverride("DATA_BATCH", 3);
                options.AddFreeDimensionOverride("DATA_CHANNEL", 5);

                using (var session = new InferenceSession(model, options))
                {
                    var inputMetadata = session.InputMetadata;
                    var dims = inputMetadata["x"].Dimensions;
                    Assert.Equal(3, dims.Length);
                    Assert.Equal(3, dims[0]);
                    Assert.Equal(5, dims[1]);
                    Assert.Equal(5, dims[2]);
                }
            }
        }

        private void ValidateRunResults(IReadOnlyCollection<NamedOnnxValue> results)
        {
            // validate the results
            foreach (var r in results)
            {
                Assert.Equal(1, results.Count);
                Assert.Equal("softmaxout_1", r.Name);

                float[] expectedOutput = TestDataLoader.LoadTensorFromEmbeddedResource("bench.expected_out");
                int[] expectedDimensions = { 1, 1000, 1, 1 };  // hardcoded for now for the test data
                ValidateRunResultData(r.AsTensor<float>(), expectedOutput, expectedDimensions);
            }
        }

        private void ValidateRunResultData(Tensor<float> resultTensor, float[] expectedOutput, int[] expectedDimensions)
        {
            Assert.Equal(expectedDimensions.Length, resultTensor.Rank);

            var resultDimensions = resultTensor.Dimensions;
            for (int i = 0; i < expectedDimensions.Length; i++)
            {
                Assert.Equal(expectedDimensions[i], resultDimensions[i]);
            }

            var resultArray = resultTensor.ToArray();
            Assert.Equal(expectedOutput.Length, resultArray.Length);
            Assert.Equal(expectedOutput, resultArray, new FloatComparer());
        }

        private static void ValidateRunResult(OrtValue resultTensor, ReadOnlySpan<float> expectedOutput, long[] expectedShape)
        {
            Assert.True(resultTensor.IsTensor);

            var typeShape = resultTensor.GetTensorTypeAndShape();
            Assert.Equal(TensorElementType.Float, typeShape.ElementDataType);

            Assert.Equal(typeShape.Shape, expectedShape);
            var resultSpan = resultTensor.GetTensorDataAsSpan<float>().ToArray();
            var expectedSpan = expectedOutput.ToArray();
            Assert.Equal(expectedSpan, resultSpan, new FloatComparer());
        }

        [Fact(DisplayName = "ThrowWrongInputName")]
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
            Assert.Contains("Input name: 'wrong_name' is not in the metadata", ex.Message);
            session.Dispose();
        }

        [Fact(DisplayName = "ThrowWrongInputType")]
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
            var msg = ex.ToString();
            Assert.Contains("Tensor element data type discovered", msg);
            session.Dispose();
        }

        [Fact(DisplayName = "ThrowExtraInputs")]
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
            Assert.Contains("Input name: 'extra' is not in the metadata", ex.Message);
            session.Dispose();
        }

        [Fact(DisplayName = "ThrowInconsistentPinnedInputs")]
        private void ThrowInconsistentPinnedInputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;

            using (var inputs = new DisposableListTest<FixedBufferOnnxValue>())
            {
                inputs.Add(FixedBufferOnnxValue.CreateFromTensor(tensor));
                var ex = Assert.Throws<ArgumentException>(() => session.Run(new string[0], inputs));
                Assert.StartsWith("Length of inputNames (0) must match that of inputValues (1).", ex.Message);
            }
        }

        [Fact(DisplayName = "ThrowWrongOutputName")]
        private void ThrowWrongOutputName()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 2 });
            // var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("bad_output_name", outputTensor) };
            var bad_names = new string[] { "bad_output_name" };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, bad_names));
            Assert.Contains("Output name: 'bad_output_name' is not in the metadata", ex.Message);
            session.Dispose();
        }

        [Fact(DisplayName = "ThrowWrongOutputType")]
        private void ThrowWrongOutputType()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<int>((ReadOnlySpan<int>)new[] { 1, 1000, 1, 1 });
            var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("softmaxout_1", outputTensor) };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, outputs));
            // TODO: check exception message
            // InferenceSession::ValidateOutputs() does not check type so far. Currently this will finally trigger an error in Softmax.
            session.Dispose();
        }

        [Fact(DisplayName = "ThrowWrongOutputDimension")]
        private void ThrowWrongOutputDimension()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 1001, 1, 1 });
            var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("softmaxout_1", outputTensor) };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, outputs));
            // TODO: check exception message
            // InferenceSession::ValidateOutputs() does not check dims so far. Currently this will finally trigger an error in Softmax.
            session.Dispose();
        }

        [Fact(DisplayName = "ThrowNoOutput")]
        private void ThrowNoOutput()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 1000, 1, 1 });
            var outputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("softmaxout_1", outputTensor) };
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(inputs, new NamedOnnxValue[0]));
            Assert.Contains("[ErrorCode:InvalidArgument] At least one output should be requested.", ex.Message);
            session.Dispose();
        }

        [Fact(DisplayName = "ThrowInconsistentPinnedOutputs")]
        private void ThrowInconsistentPinnedOutputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputTensor = tuple.Item3;
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("data_0", inputTensor) };
            var outputTensor = new DenseTensor<float>((ReadOnlySpan<int>)new[] { 1, 1000, 1, 1 });

            using (var outputs = new DisposableListTest<FixedBufferOnnxValue>())
            {
                var ex = Assert.Throws<ArgumentException>(() => session.Run(inputs, new string[] { "softmaxout_1" }, outputs));
                Assert.StartsWith("Length of outputNames (1) must match that of outputValues (0).", ex.Message);
            }
        }

        [Fact(DisplayName = "TestMultiThreads")]
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
                tasks[i] = Task.Factory.StartNew((Action)(() =>
                {
                    for (int j = 0; j < loop; j++)
                    {
                        var resnov = session.Run(container);
                        var res = resnov.ToArray()[0].AsTensor<float>().ToArray();

                        Assert.Equal(res, expectedOut, (IEqualityComparer<float>)new FloatComparer());
                    }
                }));
            };
            Task.WaitAll(tasks);
            session.Dispose();
        }

        [Fact(DisplayName = "TestOverridableInitializerMetadata")]
        private void TestOverridableInitializerMetadata()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("overridable_initializer.onnx");
            using (var session = new InferenceSession(model))
            {
                Assert.Equal(2, session.InputMetadata.Count);
                Assert.True(session.InputMetadata.ContainsKey("Label"));
                Assert.True(session.InputMetadata.ContainsKey("F2"));

                Assert.Equal(1, session.OverridableInitializerMetadata.Count);
                Assert.True(session.OverridableInitializerMetadata.ContainsKey("F1"));
                Assert.True(session.OverridableInitializerMetadata["F1"].IsTensor);
                Assert.Equal(typeof(float), session.OverridableInitializerMetadata["F1"].ElementType);
                Assert.Equal(2, session.OverridableInitializerMetadata["F1"].Dimensions.Length);
                Assert.Equal(1, session.OverridableInitializerMetadata["F1"].Dimensions[0]);
                Assert.Equal(1, session.OverridableInitializerMetadata["F1"].Dimensions[1]);

                var container = new List<NamedOnnxValue>();
                var Label_input = new DenseTensor<bool>(new bool[] { true }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("Label", Label_input));

                var F2_input = new DenseTensor<string>(new string[] { "f2_string" }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("F2", F2_input));

                var F1_initializer = new DenseTensor<float>(new float[] { 2.0f }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("F1", F1_initializer));

                using (var result = session.Run(container))
                {
                    var resultMap = new Dictionary<string, NamedOnnxValue>();

                    foreach (var output in result)
                    {
                        resultMap[output.Name] = output;
                    }

                    Assert.True(resultMap.ContainsKey("Label0"));
                    Assert.True(resultMap.ContainsKey("F20"));
                    Assert.True(resultMap.ContainsKey("F11"));

                    var overriddenInitializer = resultMap["F11"].AsTensor<float>();
                    Assert.NotNull(overriddenInitializer);
                    Assert.True(overriddenInitializer.SequenceEqual(F1_initializer));
                }
            }
        }

        [Fact(DisplayName = "TestSymbolicDimsMetadata")]
        private void TestSymbolicDimsMetadata()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("capi_symbolic_dims.onnx");
            using (var session = new InferenceSession(model))
            {
                var inputs = session.InputMetadata;
                var outputs = session.OutputMetadata;

                Assert.Equal(2, inputs.Count);
                Assert.Equal(1, session.OutputMetadata.Count);
                Assert.True(inputs.ContainsKey("A"));
                Assert.True(inputs.ContainsKey("B"));
                Assert.True(outputs.ContainsKey("C"));

                var inputA = inputs["A"];
                var inputB = inputs["B"];
                var outputC = outputs["C"];

                // dimension values and any symbolic dimension info should have the same length
                Assert.Equal(inputA.Dimensions.Length, inputA.SymbolicDimensions.Length);
                Assert.Equal(inputB.Dimensions.Length, inputB.SymbolicDimensions.Length);
                Assert.Equal(outputC.Dimensions.Length, outputC.SymbolicDimensions.Length);

                Assert.Equal(inputA.Dimensions, new int[] { -1, 2 });
                Assert.Equal(inputA.SymbolicDimensions, new string[] { "n", "" });
                Assert.Equal(inputB.Dimensions, new int[] { -1 });
                Assert.Equal(inputB.SymbolicDimensions, new string[] { "m" });
                Assert.Equal(outputC.Dimensions, new int[] { -1 });
                Assert.Equal(outputC.SymbolicDimensions, new string[] { "" }); // unnamed symbolic dim
            }
        }

        [Fact(DisplayName = "TestModelInputFloat")]
        private void TestModelInputFloat()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_FLOAT.pb");

            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputBOOL")]
        private void TestModelInputBOOL()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_BOOL.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestReusingRunOutputNonStringType")]
        private void TestReusingRunOutputNonStringType()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_BOOL.pb");
            using (var session = new InferenceSession(model))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<bool>(new bool[] { true, false, true, false, true }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                var res1 = session.Run(container);

                // change the name of the DisposableNamedOnnxValue
                res1.First().Name = "input";

                // Run inferencing 2 times using the output of the first Run()
                for (int i = 0; i < 2; ++i)
                {
                    using (var res2 = session.Run(res1))
                    {
                        var tensorOut = res2.First().AsTensor<bool>();
                        Assert.True(tensorOut.SequenceEqual(tensorIn));
                    }
                }
            }
        }

        [Fact(DisplayName = "TestReusingRunOutputStringType")]
        private void TestReusingRunOutputStringType()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_STRING.pb");
            using (var session = new InferenceSession(model))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[] { "a", "b", "c", "d", "e" }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                var res1 = session.Run(container);

                // change the name of the DisposableNamedOnnxValue
                res1.First().Name = "input";

                // Run inferencing 2 times using the output of the first Run()
                for (int i = 0; i < 2; ++i)
                {
                    using (var res2 = session.Run(res1))
                    {
                        var tensorOut = res2.First().AsTensor<string>();
                        Assert.True(tensorOut.SequenceEqual(tensorIn));
                    }
                }
            }
        }

        [Fact(DisplayName = "TestCreateFixedBufferOnnxValueFromStringTensor")]
        public void TestCreateFixedBufferOnnxValueFromStringTensor()
        {
            var tensor = new DenseTensor<string>(new string[] { "a", "b" }, new int[] { 1, 2 });
            using (var value = FixedBufferOnnxValue.CreateFromTensor(tensor)) { }
        }

        [Fact(DisplayName = "TestReusingStringFixedBufferOnnxValue")]
        public void TestReusingStringFixedBufferOnnxValue()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_STRING.pb");
            using (var session = new InferenceSession(model))
            {
                var tensorA = new DenseTensor<string>(new string[] { "a", "b", "c", "d", "e" }, new int[] { 1, 5 });
                var tensorB = new DenseTensor<string>(new string[] { "v", "w", "x", "y", "z" }, new int[] { 1, 5 });
                var tensorC = new DenseTensor<string>(new string[] { "i", "j", "k", "l", "m" }, new int[] { 1, 5 });
                var tensorD = new DenseTensor<string>(new string[] { "i", "j", "k", "l", "m" }, new int[] { 1, 5 });
                using (FixedBufferOnnxValue a = FixedBufferOnnxValue.CreateFromTensor(tensorA),
                                            b = FixedBufferOnnxValue.CreateFromTensor(tensorB),
                                            c = FixedBufferOnnxValue.CreateFromTensor(tensorC),
                                            d = FixedBufferOnnxValue.CreateFromTensor(tensorD))
                {
                    // OK to use string type FixedBufferOnnxValue only in input
                    session.Run(new[] { "input" }, new[] { a });

                    // Cannot use string type FixedBufferOnnxValue in output
                    Assert.Throws<NotSupportedException>(() =>
                    {
                        // NamedOnnxValue inputs
                        session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", tensorB) }, new[] { "output" }, new[] { b });
                    });
                    Assert.Throws<NotSupportedException>(() =>
                    {
                        // both FixedBufferOnnxValue for inputs and outputs
                        session.Run(new[] { "input" }, new[] { c }, new[] { "output" }, new[] { d });
                    });
                }
            }
        }

        [Fact(DisplayName = "TestReusingFixedBufferOnnxValue")]
        private void TestReusingFixedBufferOnnxValue()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_BOOL.pb");
            using (var session = new InferenceSession(model))
            {
                var bufferA = new bool[] { true, false, true, false, true };
                var bufferB = new bool[bufferA.Length];
                var bufferC = new bool[bufferA.Length];
                var tensorA = new DenseTensor<bool>(bufferA, new int[] { 1, 5 });
                var tensorB = new DenseTensor<bool>(bufferB, new int[] { 1, 5 });
                var tensorC = new DenseTensor<bool>(bufferC, new int[] { 1, 5 });
                using (FixedBufferOnnxValue a = FixedBufferOnnxValue.CreateFromTensor(tensorA),
                                            b = FixedBufferOnnxValue.CreateFromTensor(tensorB),
                                            c = FixedBufferOnnxValue.CreateFromTensor(tensorC))
                {
                    session.Run(new[] { "input" }, new[] { a }, new[] { "output" }, new[] { b });
                    session.Run(new[] { "input" }, new[] { b }, new[] { "output" }, new[] { c });
                }

                Assert.True(tensorC.SequenceEqual(tensorA));
            }
        }

        [Fact(DisplayName = "TestReusingFixedBufferOnnxValueMultiInferences")]
        private void TestReusingFixedBufferOnnxValueMultiInferences()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_INT32.pb");
            using (var session = new InferenceSession(model))
            {
                var bufferInput = new int[5];
                var bufferOutput = new int[5];
                var tensorInput = new DenseTensor<int>(bufferInput, new int[] { 1, 5 });
                var tensorOutput = new DenseTensor<int>(bufferOutput, new int[] { 1, 5 });

                using (FixedBufferOnnxValue valueInput = FixedBufferOnnxValue.CreateFromTensor(tensorInput),
                                            valueOutput = FixedBufferOnnxValue.CreateFromTensor(tensorOutput))
                {
                    var inputNames = new[] { "input" };
                    var outputNames = new[] { "output" };
                    var inputValues = new[] { valueInput };
                    var outputValues = new[] { valueOutput };

                    var rand = new Random();

                    // run the model for multiple times
                    for (var i = 0; i < 1000; i++)
                    {
                        // feed inputs ( 5 random integers )
                        var inputs = Enumerable.Range(0, 5).Select(x => rand.Next()).ToArray();
                        inputs.CopyTo(bufferInput, 0);

                        // run inference
                        session.Run(inputNames, inputValues, outputNames, outputValues);

                        // use outputs
                        Assert.Equal(inputs, bufferOutput);
                    }
                }
            }
        }

        [Fact(DisplayName = "TestModelInputINT32")]
        private void TestModelInputINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_INT32.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputDOUBLE")]
        private void TestModelInputDOUBLE()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_DOUBLE.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputSTRING")]
        private void TestModelInputSTRING()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_STRING.pb");
            using (var session = new InferenceSession(model))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[] {
                 "hello",
                 "École élémentaire",
                 "mit freundlichen grüßen",
                 "Понедельник",
                 "最好的问候,"+
                 "नमस्ते," +
                 "こんにちは," +
                 "안녕하세요"
                }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<string>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact(DisplayName = "TestModelInputSTRING_ShouldFailWithNullInput")]
        private void TestModelInputSTRING_ShouldFailWithNullInput()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_STRING.pb");
            using (var session = new InferenceSession(model))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[5], // null
                                                       new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                Assert.Throws<ArgumentNullException>(() => { session.Run(container); });
            }
        }

        [Fact(DisplayName = "TestModelInputINT8")]
        private void TestModelInputINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_INT8.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputUINT8")]
        private void TestModelInputUINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_UINT8.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputUINT16")]
        private void TestModelInputUINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_UINT16.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputINT16")]
        private void TestModelInputINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_INT16.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputINT64")]
        private void TestModelInputINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_INT64.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputUINT32")]
        private void TestModelInputUINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_UINT32.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputUINT64")]
        private void TestModelInputUINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_UINT64.pb");
            using (var session = new InferenceSession(model))
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

        [Fact(DisplayName = "TestModelInputFLOAT16")]
        private void TestModelInputFLOAT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            Float16[] modelInput = { new Float16(15360), new Float16(16384), new Float16(16896), new Float16(17408), new Float16(17664) };
            int[] inputShape = { 1, 5 };
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_FLOAT16.onnx");
            using (var session = new InferenceSession(model))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Float16>(modelInput, inputShape);
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var valueOut = res.First();
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, valueOut.ValueType);
                    Assert.Equal(Tensors.TensorElementType.Float16, valueOut.ElementType);
                    var tensorOut = res.First().AsTensor<Float16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact(DisplayName = "TestModelInputBFLOAT16")]
        private void TestModelInputBFLOAT16()
        {
            BFloat16[] modelInput = { new BFloat16(16256), new BFloat16(16384),
                new BFloat16(16448), new BFloat16(16512), new BFloat16(16544) };
            int[] inputShape = { 1, 5 };
            // model takes 1x5 input of fixed type, echoes back
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_BFLOAT16.onnx");
            using (var session = new InferenceSession(model))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<BFloat16>(modelInput, inputShape);
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var valueOut = res.First();
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, valueOut.ValueType);
                    Assert.Equal(Tensors.TensorElementType.BFloat16, valueOut.ElementType);
                    var tensorOut = res.First().AsTensor<BFloat16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        private class IgnoreWhenMlOpsDisabledFact : FactAttribute
        {
            public IgnoreWhenMlOpsDisabledFact()
            {
                var disableMlOpsEnvVar = Environment.GetEnvironmentVariable("DisableMlOps");
                var isMlOpsDisabled = (disableMlOpsEnvVar != null) ? disableMlOpsEnvVar.Equals("ON") : false;
                if (isMlOpsDisabled)
                {
                    Skip = "Skipping this test since Ml Ops are disabled.";
                }
            }
        }

        [IgnoreWhenMlOpsDisabledFact(DisplayName = "TestModelSequenceOfMapIntFloat")]
        private void TestModelSequenceOfMapIntFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_sequence_map_int_float.pb");
            using (var session = new InferenceSession(model))
            {

                var outMeta = session.OutputMetadata;
                var label_meta = outMeta["label"];
                Assert.True(label_meta.IsTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, label_meta.OnnxValueType);
                Assert.Equal(TensorElementType.Int64, label_meta.ElementDataType);
                Assert.NotEmpty(label_meta.Dimensions);

                // sequence<map<int64, float>>
                var probabilities_meta = outMeta["probabilities"];
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, probabilities_meta.OnnxValueType);
                var seqElementMetata = probabilities_meta.AsSequenceMetadata().ElementMeta;
                Assert.Equal(OnnxValueType.ONNX_TYPE_MAP, seqElementMetata.OnnxValueType);
                var mapMetadata = seqElementMetata.AsMapMetadata();
                // Map<int64, float tensor>
                Assert.Equal(Tensors.TensorElementType.Int64, mapMetadata.KeyDataType);
                var valueTensorMeta = mapMetadata.ValueMetadata;
                Assert.True(valueTensorMeta.IsTensor);
                Assert.Equal(Tensors.TensorElementType.Float, valueTensorMeta.ElementDataType);

                // tensor<float>
                var inputMeta = session.InputMetadata["input"];
                Assert.True(inputMeta.IsTensor);
                Assert.Equal(Tensors.TensorElementType.Float, inputMeta.ElementDataType);
                Assert.Equal(2, inputMeta.Dimensions.Length);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode0 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode0.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outNode0.ValueType);
                    Assert.Equal(Tensors.TensorElementType.Int64, outNode0.ElementType);

                    // try-cast as a tensor
                    var outLabelTensor = outNode0.AsTensor<long>();
                    Assert.NotNull(outLabelTensor);

                    // Label 1 should have highest probability
                    Assert.Equal(1, outLabelTensor[0]);

                    // second output is a sequence<map<int64, float>>
                    // try-cast to an sequence of NOV
                    var outNode1 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode1.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outNode1.ValueType);

                    // try-cast to an sequence of NOV
                    var seq = outNode1.AsEnumerable<NamedOnnxValue>();
                    Assert.NotNull(seq);
                    // Try-cast into DisposableNov so we can control and check the process


                    // try-cast first element in sequence to map/dictionary type
                    if (System.Environment.Is64BitProcess)
                    {
                        var map = seq.First().AsDictionary<Int64, float>();
                        Assert.NotNull(map);
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                    else // 32-bit
                    {
                        var map = seq.First().AsDictionary<long, float>();
                        Assert.NotNull(map);
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                }
            }
        }

        [IgnoreWhenMlOpsDisabledFact(DisplayName = "TestModelSequenceOfMapStringFloat")]
        private void TestModelSequenceOfMapStringFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_sequence_map_string_float.pb");

            using (var session = new InferenceSession(model))
            {
                var outMeta = session.OutputMetadata;
                var label_meta = outMeta["label"];
                Assert.True(label_meta.IsTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, label_meta.OnnxValueType);
                Assert.True(label_meta.IsString);
                Assert.Equal(TensorElementType.String, label_meta.ElementDataType);
                Assert.NotEmpty(label_meta.Dimensions);

                // sequence<map<string, float>>
                var probabilities_meta = outMeta["probabilities"];
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, probabilities_meta.OnnxValueType);
                var seqElementMetata = probabilities_meta.AsSequenceMetadata().ElementMeta;
                Assert.Equal(OnnxValueType.ONNX_TYPE_MAP, seqElementMetata.OnnxValueType);
                var mapMetadata = seqElementMetata.AsMapMetadata();
                Assert.Equal(Tensors.TensorElementType.String, mapMetadata.KeyDataType);
                var valueTensorMeta = mapMetadata.ValueMetadata;
                Assert.True(valueTensorMeta.IsTensor);
                Assert.Equal(Tensors.TensorElementType.Float, valueTensorMeta.ElementDataType);


                // tensor<float>
                var inputMeta = session.InputMetadata["input"];
                Assert.True(inputMeta.IsTensor);
                Assert.False(inputMeta.IsString);
                Assert.Equal(Tensors.TensorElementType.Float, inputMeta.ElementDataType);
                Assert.Equal(2, inputMeta.Dimensions.Length);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode0 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode0.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outNode0.ValueType);
                    Assert.Equal(TensorElementType.String, outNode0.ElementType);

                    // try-cast as a tensor
                    var outLabelTensor = outNode0.AsTensor<string>();
                    Assert.NotNull(outLabelTensor);

                    // Label 1 should have highest probability
                    Assert.Equal("1", outLabelTensor[0]);

                    // second output is a sequence<map<string, float>>
                    // try-cast to an sequence of NOV
                    var outNode1 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode1.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outNode1.ValueType);

                    // try-cast to an sequence of NOV
                    var seq = outNode1.AsEnumerable<NamedOnnxValue>();

                    // try-cast first element in sequence to map/dictionary type
                    var map = seq.First().AsDictionary<string, float>();
                    Assert.NotNull(map);
                    //verify values are valid
                    Assert.Equal(0.25938290, map["0"], 6);
                    Assert.Equal(0.40904793, map["1"], 6);
                    Assert.Equal(0.33156919, map["2"], 6);
                }
            }
        }

        [Fact(DisplayName = "TestModelSequenceOfTensors")]
        private void TestModelSequenceOfTensors()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_sequence_tensors.onnx");

            using (var session = new InferenceSession(model))
            {
                var outMeta = session.OutputMetadata;
                var output_seq = outMeta["output_sequence"];
                Assert.False(output_seq.IsTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, output_seq.OnnxValueType);
                var elemMeta = output_seq.AsSequenceMetadata().ElementMeta;
                Assert.True(elemMeta.IsTensor);
                Assert.Equal(Tensors.TensorElementType.Int64, elemMeta.ElementDataType);

                // Inputs
                var tensor1Meta = session.InputMetadata["tensor1"];
                Assert.True(tensor1Meta.IsTensor);
                Assert.Equal(Tensors.TensorElementType.Int64, tensor1Meta.ElementDataType);
                Assert.Equal(2, tensor1Meta.Dimensions.Length);

                var container = new List<NamedOnnxValue>();
                var firstInputTensor = new DenseTensor<Int64>(new Int64[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
                var secondInputTensor = new DenseTensor<Int64>(new Int64[] { 7, 8, 9, 10, 11, 12 }, new int[] { 2, 3 });

                var firstNov = NamedOnnxValue.CreateFromTensor("tensor1", firstInputTensor);
                var secondNov = NamedOnnxValue.CreateFromTensor("tensor2", secondInputTensor);

                container.Add(firstNov);
                container.Add(secondNov);

                using (var outputs = session.Run(container))
                {
                    // output is a sequence<tensors>
                    // try-cast to an sequence of NOV
                    var outNode = outputs.ElementAtOrDefault(0);
                    Assert.Equal("output_sequence", outNode.Name);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outNode.ValueType);

                    // try-cast to an sequence of NOV
                    var seq = outNode.AsEnumerable<NamedOnnxValue>();

                    // make sure that the sequence holds only 2 elements (tensors)
                    Assert.True(seq.Count() == 2);

                    // try-cast the elements in sequence to tensor type
                    var firstTensorInOuputSequence = seq.First().AsTensor<Int64>();
                    var secondTensorInOuputSequence = seq.Last().AsTensor<Int64>();
                    Assert.NotNull(firstTensorInOuputSequence);
                    Assert.NotNull(secondTensorInOuputSequence);

                    // make sure the tensors in the output sequence hold the correct values
                    Assert.True(firstTensorInOuputSequence.GetValue(0) == 1);
                    Assert.True(firstTensorInOuputSequence.GetValue(1) == 2);
                    Assert.True(firstTensorInOuputSequence.GetValue(2) == 3);
                    Assert.True(firstTensorInOuputSequence.GetValue(3) == 4);
                    Assert.True(firstTensorInOuputSequence.GetValue(4) == 5);
                    Assert.True(firstTensorInOuputSequence.GetValue(5) == 6);

                    Assert.True(secondTensorInOuputSequence.GetValue(0) == 7);
                    Assert.True(secondTensorInOuputSequence.GetValue(1) == 8);
                    Assert.True(secondTensorInOuputSequence.GetValue(2) == 9);
                    Assert.True(secondTensorInOuputSequence.GetValue(3) == 10);
                    Assert.True(secondTensorInOuputSequence.GetValue(4) == 11);
                    Assert.True(secondTensorInOuputSequence.GetValue(5) == 12);
                }
            }
        }

        [Fact(DisplayName = "TestModelMetadata")]
        private void TestModelMetadata()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("model_with_valid_ort_config_json.onnx");

            using (var session = new InferenceSession(model))
            {
                var modelMetadata = session.ModelMetadata;

                Assert.Equal(1, modelMetadata.Version);

                Assert.Equal("Hari", modelMetadata.ProducerName);

                Assert.Equal("matmul test", modelMetadata.GraphName);

                Assert.Equal("", modelMetadata.Domain);

                Assert.Equal("This is a test model with a valid ORT config Json", modelMetadata.Description);

                Assert.Equal("graph description", modelMetadata.GraphDescription);

                Assert.Equal(2, modelMetadata.CustomMetadataMap.Keys.Count);
                Assert.Equal("dummy_value", modelMetadata.CustomMetadataMap["dummy_key"]);
                Assert.Equal("{\"session_options\": {\"inter_op_num_threads\": 5, \"intra_op_num_threads\": 2, \"graph_optimization_level\": 99, \"enable_profiling\": 1}}",
                              modelMetadata.CustomMetadataMap["ort_config"]);
            }
        }

        [Fact(DisplayName = "TestInferenceSessionWithByteArray")]
        private void TestInferenceSessionWithByteArray()
        {
            // model takes 1x5 input of fixed type, echoes back
            var modelData = TestDataLoader.LoadModelFromEmbeddedResource("test_types_FLOAT.pb");

            using (var session = new InferenceSession(modelData))
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

        void TestCPUAllocatorInternal(InferenceSession session)
        {
            int device_id = 0;
            using (var info_cpu = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.ArenaAllocator, device_id, OrtMemType.Default))
            {
                Assert.Equal("Cpu", info_cpu.Name);
                Assert.Equal(device_id, info_cpu.Id);
                Assert.Equal(OrtAllocatorType.ArenaAllocator, info_cpu.GetAllocatorType());
                Assert.Equal(OrtMemType.Default, info_cpu.GetMemoryType());

                using (var allocator = new OrtAllocator(session, info_cpu))
                {
                    var alloc_info = allocator.Info;
                    // Allocator type returned may be different on x86 so we don't compare.
                    Assert.Equal(info_cpu.Name, alloc_info.Name);
                    Assert.Equal(info_cpu.GetMemoryType(), alloc_info.GetMemoryType());
                    Assert.Equal(info_cpu.Id, alloc_info.Id);

                    uint size = 1024;
                    OrtMemoryAllocation chunk = allocator.Allocate(size);
                    Assert.Equal(chunk.Size, size);
                    var chunk_info = chunk.Info;
                    // Allocator type returned may be different on x86 so we don't compare.
                    Assert.Equal(chunk_info.Name, alloc_info.Name);
                    Assert.Equal(chunk_info.GetMemoryType(), alloc_info.GetMemoryType());
                    Assert.Equal(chunk_info.Id, alloc_info.Id);
                    chunk.Dispose();
                    alloc_info.Dispose();
                }
            }
        }

#if USE_CUDA
        void TestCUDAAllocatorInternal(InferenceSession session)
        {
            int device_id = 0;
            using (var info_cuda = new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.ArenaAllocator, device_id, OrtMemType.Default))
            {
                Assert.Equal("Cuda", info_cuda.Name);
                Assert.Equal(device_id, info_cuda.Id);
                Assert.Equal(OrtAllocatorType.ArenaAllocator, info_cuda.GetAllocatorType());
                Assert.Equal(OrtMemType.Default, info_cuda.GetMemoryType());

                using (var allocator = new OrtAllocator(session, info_cuda))
                {
                    var alloc_info = allocator.Info;
                    Assert.True(info_cuda.Equals(alloc_info));

                    uint size = 1024;
                    OrtMemoryAllocation chunk = allocator.Allocate(size);
                    Assert.Equal(chunk.Size, size);
                    Assert.True(chunk.Info.Equals(alloc_info));
                    chunk.Dispose();
                    alloc_info.Dispose();
                }
            }
        }
#endif

#if USE_ROCM
        void TestROCMAllocatorInternal(InferenceSession session)
        {
            int device_id = 0;
            using (var info_rocm = new OrtMemoryInfo(OrtMemoryInfo.allocatorHIP, OrtAllocatorType.ArenaAllocator, device_id, OrtMemType.Default))
            {
                Assert.Equal("Hip", info_rocm.Name);
                Assert.Equal(device_id, info_rocm.Id);
                Assert.Equal(OrtAllocatorType.ArenaAllocator, info_rocm.GetAllocatorType());
                Assert.Equal(OrtMemType.Default, info_rocm.GetMemoryType());

                using (var allocator = new OrtAllocator(session, info_rocm))
                {
                    var alloc_info = allocator.Info;
                    Assert.True(info_rocm.Equals(alloc_info));

                    uint size = 1024;
                    OrtMemoryAllocation chunk = allocator.Allocate(size);
                    Assert.Equal(chunk.Size, size);
                    Assert.True(chunk.Info.Equals(alloc_info));
                    chunk.Dispose();
                    alloc_info.Dispose();
                }
            }
        }
#endif

        [Fact(DisplayName = "TestAllocator")]
        private void TestAllocator()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
            using (SessionOptions options = new SessionOptions())
            {
                options.AppendExecutionProvider_CPU(1);
#if USE_CUDA
                options.AppendExecutionProvider_CUDA(0);
#endif

#if USE_ROCM
                options.AppendExecutionProvider_ROCm(0);
#endif

                using (var session = new InferenceSession(model, options))
                {
                    TestCPUAllocatorInternal(session);
#if USE_CUDA
                    TestCUDAAllocatorInternal(session);
#endif
#if USE_ROCM
                    TestROCMAllocatorInternal(session);
#endif

                }
            }
        }

        [Fact(DisplayName = "TestSharingOfInitializerAndItsPrepackedVersion")]
        private void TestSharingOfInitializerAndItsPrepackedVersion()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("matmul_1.onnx");

            // create initializer to share
            var ortCpuMemInfo = OrtMemoryInfo.DefaultInstance;
            var dims = new long[] { 2, 1 };
            var dataBuffer = new float[] { 2.0F, 1.0F };
            var dataHandle = GCHandle.Alloc(dataBuffer, GCHandleType.Pinned);

            try
            {
                unsafe
                {
                    float* p = (float*)dataHandle.AddrOfPinnedObject();
                    for (int i = 0; i < dataBuffer.Length; ++i)
                    {
                        *p++ = dataBuffer[i];
                    }
                }
                var dataBufferNumBytes = (uint)dataBuffer.Length * sizeof(float);

                using (var sharedInitializer = OrtValue.CreateTensorValueWithData(ortCpuMemInfo, Tensors.TensorElementType.Float,
                                        dims, dataHandle.AddrOfPinnedObject(), dataBufferNumBytes))
                {

                    using (var prepackedWeightsContainer = new PrePackedWeightsContainer())
                    {
                        using (var options = new SessionOptions())
                        {
                            // We want to share initializers between the two sessions
                            options.AddInitializer("W", sharedInitializer);

                            float[] expectedOutput = { 4.0F, 10.0F, 16.0F };
                            int[] expectedDimensions = { 3, 1 };

                            // We want the pre-packed weights of the shared initializer to be shared between sessions (memory savings)
                            // and hence we pass in the 'prepackedWeightsContainer' at session creation time
                            byte[] modelData = model;

                            // Test both InferenceSession ctors that take PrePackedWeightsContainer instances
                            using (var session = new InferenceSession(model, options, prepackedWeightsContainer))
                            using (var session2 = new InferenceSession(modelData, options, prepackedWeightsContainer))
                            {
                                var inputMeta = session.InputMetadata;
                                var container = new List<NamedOnnxValue>();

                                foreach (var name in inputMeta.Keys)
                                {
                                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                                    Assert.True(inputMeta[name].IsTensor);
                                    var tensor = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, inputMeta[name].Dimensions);
                                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                                }

                                ReadOnlySpan<int> expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                                string[] expectedOutputNames = new string[] { "Y" };

                                // Run inference with named inputs and outputs created with in Run()
                                using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                                {
                                    foreach (var r in results)
                                    {
                                        ValidateRunResultData(r.AsTensor<float>(), expectedOutput, expectedDimensions);
                                    }
                                }

                                // Run inference with named inputs and outputs created with in Run()
                                using (var results2 = session2.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                                {
                                    foreach (var r in results2)
                                    {
                                        ValidateRunResultData(r.AsTensor<float>(), expectedOutput, expectedDimensions);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            finally
            {
                dataHandle.Free();
            }
        }

        [Fact(DisplayName = "TestSharedAllocatorUsingCreateAndRegisterAllocator")]
        private void TestSharedAllocatorUsingCreateAndRegisterAllocator()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("mul_1.onnx");

            using (var memInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU,
                                                   OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default))
            using (var arenaCfg = new OrtArenaCfg(0, -1, -1, -1))
            {
                var env = OrtEnv.Instance();
                // Create and register the arena based allocator
                env.CreateAndRegisterAllocator(memInfo, arenaCfg);

                using (var sessionOptions = new SessionOptions())
                {
                    // Key must match kOrtSessionOptionsConfigUseEnvAllocators in onnxruntime_session_options_config_keys.h
                    sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1");

                    // Create two sessions to share the allocator
                    // Create a thrid session that DOES NOT use the allocator in the environment
                    using (var session1 = new InferenceSession(model, sessionOptions))
                    using (var session2 = new InferenceSession(model, sessionOptions))
                    using (var session3 = new InferenceSession(model)) // Use the default SessionOptions instance
                    {
                        // Input data
                        var inputDims = new long[] { 3, 2 };
                        var input = new float[] { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };

                        // Output data
                        int[] outputDims = { 3, 2 };
                        float[] output = { 1.0F, 4.0F, 9.0F, 16.0F, 25.0F, 36.0F };

                        // Run inference on all three models
                        var inputMeta = session1.InputMetadata;
                        var container = new List<NamedOnnxValue>();

                        foreach (var name in inputMeta.Keys)
                        {
                            Assert.Equal(typeof(float), inputMeta[name].ElementType);
                            Assert.True(inputMeta[name].IsTensor);
                            var tensor = new DenseTensor<float>(input, inputMeta[name].Dimensions);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                        }

                        // Run inference with named inputs and outputs created with in Run()
                        using (var results = session1.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                        {
                            foreach (var r in results)
                            {
                                ValidateRunResultData(r.AsTensor<float>(), output, outputDims);
                            }
                        }

                        // Run inference with named inputs and outputs created with in Run()
                        using (var results = session2.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                        {
                            foreach (var r in results)
                            {
                                ValidateRunResultData(r.AsTensor<float>(), output, outputDims);
                            }
                        }

                        // Run inference with named inputs and outputs created with in Run()
                        using (var results = session3.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                        {
                            foreach (var r in results)
                            {
                                ValidateRunResultData(r.AsTensor<float>(), output, outputDims);
                            }
                        }
                    }
                }
            }
        }

        internal static Tuple<InferenceSession, float[], DenseTensor<float>, float[]> OpenSessionSqueezeNet(int? deviceId = null)
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
#if USE_DML
            // Explicitly set dll probe path so that the (potentially) stale system DirectML.dll
            // doesn't get loaded by the test process when it is eventually delay loaded by onnruntime.dll
            // The managed tests binary path already contains the right DirectML.dll, so use that

            var directml_dll_path = AppDomain.CurrentDomain.BaseDirectory;
            SetDllDirectory(directml_dll_path);

            using (var option = new SessionOptions())
            {
                if (!deviceId.HasValue)
                {
                    option.AppendExecutionProvider_CPU(1);
                }

                else
                {
                    option.AppendExecutionProvider_DML(deviceId.Value);
                }

                 // Restore the default dll search order
                SetDllDirectory(null);
#elif USE_CUDA
            using (var option = (deviceId.HasValue) ?
                SessionOptions.MakeSessionOptionWithCudaProvider(deviceId.Value) :
                new SessionOptions())
            {
                if(!deviceId.HasValue)
                {
                    option.AppendExecutionProvider_CPU(1);
                }
#elif USE_ROCM
            using (var option = (deviceId.HasValue) ?
                SessionOptions.MakeSessionOptionWithRocmProvider(deviceId.Value) :
                new SessionOptions())
            {
                if(!deviceId.HasValue)
                {
                    option.AppendExecutionProvider_CPU(1);
                }
#else
            using (var option = new SessionOptions())
            {
                option.AppendExecutionProvider_CPU(1);
#endif
                var session = (deviceId.HasValue)
                    ? new InferenceSession(model, option)
                                  : new InferenceSession(model);
                float[] inputData = TestDataLoader.LoadTensorFromEmbeddedResource("bench.in");
                float[] expectedOutput = TestDataLoader.LoadTensorFromEmbeddedResource("bench.expected_out");
                var inputMeta = session.InputMetadata;
                var tensor = new DenseTensor<float>(inputData, inputMeta["data_0"].Dimensions);
                return new Tuple<InferenceSession, float[], DenseTensor<float>, float[]>(session, inputData, tensor, expectedOutput);
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

        private class SkipNonPackageTests : FactAttribute
        {
            public SkipNonPackageTests()
            {
                var skipNonPackageTests = System.Environment.GetEnvironmentVariable("SKIPNONPACKAGETESTS");
                if (skipNonPackageTests != null && skipNonPackageTests.Equals("ON"))
                {
                    Skip = "Test skipped while testing the package as it is not within the scope";
                }
            }
        }

        [Fact(DisplayName = "TestModelRunAsyncTask")]
        private async void TestModelRunAsyncTask()
        {
            Float16[] inputData = { new Float16(15360), new Float16(16384), new Float16(16896), new Float16(17408), new Float16(17664) };
            long[] shape = { 1, 5 };

            var inputNames = new List<string> { "input" };
            var inputValues = new List<OrtValue> { OrtValue.CreateTensorValueFromMemory(inputData, shape) };

            var outputNames = new List<string> { "output" };
            var outputValues = new List<OrtValue> { OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance,
                    TensorElementType.Float16, shape) };

            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_FLOAT16.onnx");
            using (SessionOptions opt = new SessionOptions())
            {
                opt.IntraOpNumThreads = 2;
                using (var session = new InferenceSession(model, opt))
                {
                    try
                    {
                        var task = session.RunAsync(null, inputNames, inputValues, outputNames, outputValues);
                        var outputs = await task;
                        var valueOut = outputs.ElementAt<OrtValue>(0);
                        var float16s = valueOut.GetTensorDataAsSpan<Float16>().ToArray();
                        Assert.Equal(new Float16(16896), float16s[2]);
                    }
                    catch
                    {
                        Assert.True(false);
                    }
                }
            }
        }

        [Fact(DisplayName = "TestModelRunAsyncTaskFail")]
        private async void TestModelRunAsyncTaskFail()
        {
            Float16[] inputData = { new Float16(15360), new Float16(16384), new Float16(16896), new Float16(17408), new Float16(17664) };
            long[] shape = { 1, 5 };

            var inputNames = new List<string> { "input" };
            var inputValues = new List<OrtValue> { OrtValue.CreateTensorValueFromMemory(inputData, shape) };

            var outputNames = new List<string> { "output" };
            var outputValues = new List<OrtValue> { OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance,
                    TensorElementType.Float16, shape) };

            var model = TestDataLoader.LoadModelFromEmbeddedResource("test_types_FLOAT16.onnx");
            using (SessionOptions opt = new SessionOptions())
            {
                opt.IntraOpNumThreads = 1;  // this will make RunAsync fail
                string err = "";
                using (var session = new InferenceSession(model, opt))
                {
                    try
                    {
                        var task = session.RunAsync(null, inputNames, inputValues, outputNames, outputValues);
                        var outputs = await task;
                    }
                    catch (Exception ex)
                    {
                        err = ex.Message;
                    }
                    finally
                    {
                        Assert.Contains("intra op thread pool must have at least one thread for RunAsync", err);
                    }
                }
            }
        }

#if USE_AZURE
        [Fact(DisplayName = "TestLoadAzureEP")]
        private void TestLoadAzureEP()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("mul_1.onnx");

            using (var memInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU,
                                                   OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default))
            using (var arenaCfg = new OrtArenaCfg(0, -1, -1, -1))
            {
                using (var sessionOptions = new SessionOptions())
                {
                    sessionOptions.AppendExecutionProvider("AZURE");
                    try {
                        using (var session1 = new InferenceSession(model, sessionOptions))
                        {

                        }
                    }
                    catch (Exception) {
                        Assert.True(false);
                    } 
                }
            }
        }
#endif
    }
}
