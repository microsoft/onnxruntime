// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    /// <summary>
    /// Collection of OrtEnv tests that must be ran sequentially
    /// </summary>
    [Collection("Ort Inference Tests")]
    public class OrtEnvCollectionTest
    {
        [Fact(DisplayName = "EnablingAndDisablingTelemetryEventCollection")]
        public void EnablingAndDisablingTelemetryEventCollection()
        {
            var ortEnvInstance = OrtEnv.Instance();
            ortEnvInstance.DisableTelemetryEvents();

            // no-op on non-Windows builds
            // may be no-op on certain Windows builds based on build configuration

            ortEnvInstance.EnableTelemetryEvents();
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvGetVersion
    {
        [Fact(DisplayName = "GetVersionString")]
        public void GetVersionString()
        {
            var ortEnvInstance = OrtEnv.Instance();
            string versionString = ortEnvInstance.GetVersionString();
            Assert.False(versionString.Length == 0);
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvGetAvailableProviders
    {

        [Fact(DisplayName = "GetAvailableProviders")]
        public void GetAvailableProviders()
        {
            var ortEnvInstance = OrtEnv.Instance();
            string[] providers = ortEnvInstance.GetAvailableProviders();

            Assert.True(providers.Length > 0);
            Assert.Equal("CPUExecutionProvider", providers[providers.Length - 1]);

#if USE_CUDA
            Assert.True(Array.Exists(providers, provider => provider == "CUDAExecutionProvider"));
#endif
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithCustomLogLevel
    {

        [Fact(DisplayName = "TestUpdatingEnvWithCustomLogLevel")]
        public void TestUpdatingEnvWithCustomLogLevel()
        {
            var ortEnvInstance = OrtEnv.Instance();
            Assert.True(OrtEnv.IsCreated);
            ortEnvInstance.Dispose();
            Assert.False(OrtEnv.IsCreated);

            // Must be default level of warning
            ortEnvInstance = OrtEnv.Instance();
            ortEnvInstance.Dispose();
            Assert.False(OrtEnv.IsCreated);

            var envOptions = new EnvironmentCreationOptions
            {
                // Everything else is unpopulated
                logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL
            };

            ortEnvInstance = OrtEnv.CreateInstanceWithOptions(ref envOptions);
            Assert.True(OrtEnv.IsCreated);
            Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, ortEnvInstance.EnvLogLevel);

            ortEnvInstance.Dispose();
            Assert.False(OrtEnv.IsCreated);
            envOptions = new EnvironmentCreationOptions
            {
                // Everything else is unpopulated
                logId = "CSharpOnnxRuntimeTestLogid"
            };

            ortEnvInstance = OrtEnv.CreateInstanceWithOptions(ref envOptions);
            Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, ortEnvInstance.EnvLogLevel);

            // Change and see if this takes effect
            ortEnvInstance.EnvLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, ortEnvInstance.EnvLogLevel);
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithThreadingOptions
    {
        [Fact(DisplayName = "TestUpdatingEnvWithThreadingOptions")]
        public void TestUpdatingEnvWithThreadingOptions()
        {
            OrtEnv.Instance().Dispose();
            Assert.False(OrtEnv.IsCreated);

            using (var opt = new OrtThreadingOptions())
            {
                var envOptions = new EnvironmentCreationOptions
                {
                    threadOptions = opt
                };

                // Make sure we start anew
                var env = OrtEnv.CreateInstanceWithOptions(ref envOptions);
                Assert.True(OrtEnv.IsCreated);
            }
        }
    }

    public class CustomLoggingFunctionTestBase
    {
        // Custom logging constants
        protected static readonly string TestLogId = "CSharpTestLogId";
        protected static readonly IntPtr TestLogParam = (IntPtr)5;
        protected static int LoggingInvokes = 0;

        protected static void CustomLoggingFunction(IntPtr param,
                                                  OrtLoggingLevel severity,
                                                  string category,
                                                  string logId,
                                                  string codeLocation,
                                                  string message)
        {
            Assert.Equal(TestLogParam, param); // Passing test param
            Assert.False(string.IsNullOrEmpty(codeLocation));
            Assert.False(string.IsNullOrEmpty(message));
            LoggingInvokes++;
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithCustomLogger : CustomLoggingFunctionTestBase
    {
        [Fact(DisplayName = "TesEnvWithCustomLogger")]
        public void TesEnvWithCustomLogger()
        {
            // Make sure we start anew
            OrtEnv.Instance().Dispose();
            Assert.False(OrtEnv.IsCreated);
            var envOptions = new EnvironmentCreationOptions
            {
                logId = TestLogId,
                logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                loggingFunction = CustomLoggingFunction,
                loggingParam = TestLogParam
            };

            LoggingInvokes = 0;

            var env = OrtEnv.CreateInstanceWithOptions(ref envOptions);
            Assert.True(OrtEnv.IsCreated);

            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
            // Trigger some logging
            // Empty stmt intentional
            using (var session = new InferenceSession(model))
            { }

            Assert.True(LoggingInvokes > 0);
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithCustomLoggerAndThreadindOptions : CustomLoggingFunctionTestBase
    {
        [Fact(DisplayName = "TestEnvWithCustomLoggerAndThredingOptions")]
        public void TestEnvWithCustomLoggerAndThredingOptions()
        {
            OrtEnv.Instance().Dispose();
            Assert.False(OrtEnv.IsCreated);

            using (var opt = new OrtThreadingOptions())
            {
                var envOptions = new EnvironmentCreationOptions
                {
                    logId = TestLogId,
                    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                    threadOptions = opt,
                    loggingFunction = CustomLoggingFunction,
                    loggingParam = TestLogParam
                };

                LoggingInvokes = 0;

                var env = OrtEnv.CreateInstanceWithOptions(ref envOptions);
                Assert.True(OrtEnv.IsCreated);

                var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
                // Trigger some logging
                // Empty stmt intentional
                using (var session = new InferenceSession(model))
                { }

                Assert.True(LoggingInvokes > 0);
            }
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvSharedAllocatorsTests
    {
        private void ValidateRunResultData(Tensor<float> resultTensor, float[] expectedOutput, int[] expectedDimensions)
        {
            Assert.Equal(expectedDimensions.Length, resultTensor.Rank);

            var resultDimensions = resultTensor.Dimensions;
            for (int i = 0; i < expectedDimensions.Length; i++)
            {
                Assert.Equal(expectedDimensions[i], resultDimensions[i]);
            }

            var resultArray = new float[resultTensor.Length];
            for (int i = 0; i < resultTensor.Length; i++)
            {
                resultArray[i] = resultTensor.GetValue(i);
            }
            Assert.Equal(expectedOutput.Length, resultArray.Length);
            Assert.Equal(expectedOutput, resultArray, new FloatComparer());
        }

        [Fact(DisplayName = "TestSharedAllocatorUsingCreateAndRegisterAllocator")]
        private void TestSharedAllocatorUsingCreateAndRegisterAllocator()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("mul_1.onnx");

            using var memInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU,
                                                   OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
            using var arenaCfg = new OrtArenaCfg(0, -1, -1, -1);
            var env = OrtEnv.Instance();
            // Create and register the arena based allocator
            env.CreateAndRegisterAllocator(memInfo, arenaCfg);
            try
            {
                using var sessionOptions = new SessionOptions();
                // Key must match kOrtSessionOptionsConfigUseEnvAllocators in onnxruntime_session_options_config_keys.h
                sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1");

                // Create two sessions to share the allocator
                // Create a third session that DOES NOT use the allocator in the environment
                using var session1 = new InferenceSession(model, sessionOptions);
                using var session2 = new InferenceSession(model, sessionOptions);
                using var session3 = new InferenceSession(model); // Use the default SessionOptions instance
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
                using var results = session1.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container
                foreach (var r in results)
                {
                    ValidateRunResultData(r.AsTensor<float>(), output, outputDims);
                }

                // Run inference with named inputs and outputs created with in Run()
                using var results2 = session2.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container
                foreach (var r in results2)
                {
                    ValidateRunResultData(r.AsTensor<float>(), output, outputDims);
                }

                // Run inference with named inputs and outputs created with in Run()
                using var results3 = session3.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container
                foreach (var r in results3)
                {
                    ValidateRunResultData(r.AsTensor<float>(), output, outputDims);
                }
            }
            finally
            {
                // Unregister the allocator
                env.UnregisterAllocator(memInfo);
            }
        }

        [Fact(DisplayName = "TestSharedAllocatorUsingCreateAndRegisterAllocatorV2")]
        private void TestSharedAllocatorUsingCreateAndRegisterAllocatorV2()
        {
            var model = TestDataLoader.LoadModelFromEmbeddedResource("mul_1.onnx");

            using var memInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU,
                                                   OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
            using var arenaCfg = new OrtArenaCfg(0, -1, -1, -1);
            var env = OrtEnv.Instance();

            // Fill in with two arbitrary key-value pairs
            var options = new Dictionary<string, string>() {
                { "key1", "value1" },
                { "key2", "value2"  }
            };

            // Simply execute CreateAndRegisterAllocatorV2 to verify that C# API works as expected
            env.CreateAndRegisterAllocator("CPUExecutionProvider", memInfo, arenaCfg, options);
            try
            {
                using var sessionOptions = new SessionOptions();
                // Key must match kOrtSessionOptionsConfigUseEnvAllocators in onnxruntime_session_options_config_keys.h
                sessionOptions.AddSessionConfigEntry("session.use_env_allocators", "1");
                using var session = new InferenceSession(model, sessionOptions);
            }
            finally
            {
                // Unregister the allocator
                env.UnregisterAllocator(memInfo);
            }
        }
        [Fact(DisplayName = "TestCreateGetReleaseSharedAllocator")]
        private void TestCreateGetReleaseSharedAllocator()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var env = OrtEnv.Instance();
                string libFullPath = Path.Combine(Directory.GetCurrentDirectory(), "example_plugin_ep.dll");
                Assert.True(File.Exists(libFullPath), $"Expected lib {libFullPath} does not exist.");

                // example plugin ep uses the registration name as the ep name
                const string epName = "csharp_ep";

                env.RegisterExecutionProviderLibrary(epName, libFullPath);
                try
                {
                    // Find OrtEpDevice for the example EP
                    OrtEpDevice epDevice = null;
                    var epDevices = env.GetEpDevices();
                    foreach (var d in epDevices)
                    {
                        if (string.Equals(epName, d.EpName, StringComparison.OrdinalIgnoreCase))
                        {
                            epDevice = d;
                        }
                    }
                    Assert.NotNull(epDevice);

                    using var epMemoryInfo = epDevice.GetMemoryInfo(OrtDeviceMemoryType.DEFAULT);

                    var options = new Dictionary<string, string>() {
                        { "arena.initial_chunk_size_bytes", "25600" },
                    };

                    // Strictly speaking the allocator is owned by the env
                    // but we want to dispose the C# object anyway
                    using var sharedAllocator = env.CreateSharedAllocator(epDevice,
                                                                           OrtDeviceMemoryType.DEFAULT,
                                                                           OrtAllocatorType.DeviceAllocator,
                                                                           options);

                    try
                    {
                        using var getAllocator = env.GetSharedAllocator(epMemoryInfo);
                        Assert.NotNull(getAllocator);
                    }
                    finally
                    {
                        // ReleaseSharedAllocator is a no-op if the allocator was created with CreateAndRegisterAllocator
                        env.ReleaseSharedAllocator(epDevice, OrtDeviceMemoryType.DEFAULT);
                    }
                }
                finally
                {
                    env.UnregisterExecutionProviderLibrary(epName);
                }
            }
        }

        [Fact(DisplayName = "TestCopyTensors")]
        void TestCopyTensors()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var env = OrtEnv.Instance();
                string libFullPath = Path.Combine(Directory.GetCurrentDirectory(), "example_plugin_ep.dll");
                Assert.True(File.Exists(libFullPath), $"Expected lib {libFullPath} does not exist.");

                // example plugin ep uses the registration name as the ep name
                const string epName = "csharp_ep";

                env.RegisterExecutionProviderLibrary(epName, libFullPath);
                try
                {
                    // Find the example device
                    OrtEpDevice epDevice = null;
                    var epDevices = env.GetEpDevices();
                    foreach (var d in epDevices)
                    {
                        if (string.Equals(epName, d.EpName, StringComparison.OrdinalIgnoreCase))
                        {
                            epDevice = d;
                        }
                    }
                    Assert.NotNull(epDevice);

                    using var syncStream = epDevice.CreateSyncStream(null);
                    Assert.NotNull(syncStream);
                    // This returned Zero for example EP
                    // therefore do not assert for zero.
                    var streamHandle = syncStream.GetHandle();
                    // Assert.NotEqual(IntPtr.Zero, streamHandle);

                    var inputDims = new long[] { 3, 2 };
                    float[] inputData1 = [1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F];
                    long[] inputData2 = [1, 2, 3, 4, 5, 6];

                    // Create source OrtValues on CPU on top of inputData
                    using var inputList = new DisposableListTest<OrtValue>(2)
                    {
                        OrtValue.CreateTensorValueFromMemory(inputData1, inputDims),
                        OrtValue.CreateTensorValueFromMemory(inputData2, inputDims)
                    };

                    using var epMemoryInfo = epDevice.GetMemoryInfo(OrtDeviceMemoryType.DEFAULT);
                    var options = new Dictionary<string, string>() {
                        { "arena.initial_chunk_size_bytes", "25600" },
                    };

                    // Strictly speaking the allocator is owned by the env
                    // but we want to dispose the C# object anyway
                    using var sharedAllocator = env.CreateSharedAllocator(epDevice,
                                                                           OrtDeviceMemoryType.DEFAULT,
                                                                           OrtAllocatorType.DeviceAllocator,
                                                                           options);
                    try
                    {
                        // Create destination empty OrtValues on the example EP device
                        using var outputList = new DisposableListTest<OrtValue>(2)
                        {
                            OrtValue.CreateAllocatedTensorValue(sharedAllocator,
                            TensorElementType.Float, inputDims),
                            OrtValue.CreateAllocatedTensorValue(sharedAllocator,
                            TensorElementType.Int64, inputDims)
                        };

                        env.CopyTensors(inputList, outputList, syncStream);

                        // Assert.Equal data on inputList and outputList
                        Assert.Equal(inputList[0].GetTensorDataAsSpan<float>(),
                                     outputList[0].GetTensorDataAsSpan<float>());
                        Assert.Equal(inputList[1].GetTensorDataAsSpan<long>(),
                                        outputList[1].GetTensorDataAsSpan<long>());
                    }
                    finally
                    {
                        // Unregister from the env
                        env.ReleaseSharedAllocator(epDevice, OrtDeviceMemoryType.DEFAULT);
                    }
                }
                finally
                {
                    env.UnregisterExecutionProviderLibrary(epName);
                }
            }
        }
    }
}
