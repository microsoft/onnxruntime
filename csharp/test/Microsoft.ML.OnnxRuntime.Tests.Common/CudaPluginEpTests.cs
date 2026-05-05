// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// not supported on mobile platforms
#if !(ANDROID || IOS)

namespace Microsoft.ML.OnnxRuntime.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

/// <summary>
/// Tests for the CUDA Plugin Execution Provider registration and functionality.
/// These tests are skipped if the CUDA Plugin EP library is not available.
/// </summary>
public class CudaPluginEpTests
{
    private readonly OrtEnv ortEnvInstance = OrtEnv.Instance();

    private const string CudaPluginEpName = "CudaPluginExecutionProvider";

    private static string GetCudaPluginLibraryPath()
    {
        string libName;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            libName = "onnxruntime_providers_cuda_plugin.dll";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            libName = "libonnxruntime_providers_cuda_plugin.so";
        }
        else
        {
            return null;
        }

        string fullPath = Path.Combine(Directory.GetCurrentDirectory(), libName);
        return fullPath;
    }

    private static bool IsCudaPluginEpAvailable()
    {
        string libPath = GetCudaPluginLibraryPath();
        return libPath != null && File.Exists(libPath);
    }

    [SkippableFact]
    public void RegisterCudaPluginEp()
    {
        Skip.IfNot(IsCudaPluginEpAvailable(), "CUDA Plugin EP library not available.");

        string libPath = GetCudaPluginLibraryPath();

        // Register the CUDA Plugin EP library
        ortEnvInstance.RegisterExecutionProviderLibrary(CudaPluginEpName, libPath);
        try
        {
            // Verify the CUDA Plugin EP device is now available
            var epDevices = ortEnvInstance.GetEpDevices();
            var cudaPluginDevice = epDevices.FirstOrDefault(
                d => d.EpName == CudaPluginEpName);
            Assert.NotNull(cudaPluginDevice);
        }
        finally
        {
            ortEnvInstance.UnregisterExecutionProviderLibrary(CudaPluginEpName);
        }
    }

    [SkippableFact]
    public void CudaPluginEp_CreateSessionAndRunInference()
    {
        Skip.IfNot(IsCudaPluginEpAvailable(), "CUDA Plugin EP library not available.");

        string libPath = GetCudaPluginLibraryPath();

        ortEnvInstance.RegisterExecutionProviderLibrary(CudaPluginEpName, libPath);
        try
        {
            var epDevices = ortEnvInstance.GetEpDevices();
            var cudaPluginDevices = epDevices
                .Where(d => d.EpName == CudaPluginEpName)
                .ToList();

            Assert.NotEmpty(cudaPluginDevices);

            using var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider(ortEnvInstance, cudaPluginDevices, null);

            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            using var session = new InferenceSession(model, sessionOptions);
            Assert.NotNull(session);

            // Run inference with the CUDA Plugin EP using OrtValue API
            float[] inputData = TestDataLoader.LoadTensorFromEmbeddedResource("bench.in");
            float[] expectedOutput = TestDataLoader.LoadTensorFromEmbeddedResource("bench.expected_out");
            long[] expectedShape = { 1, 1000, 1, 1 };

            var inputMeta = session.InputMetadata;
            var inputName = inputMeta.Keys.First();
            var inputShape = Array.ConvertAll(inputMeta[inputName].Dimensions, d => (long)d);

            using var runOptions = new RunOptions();

            // Run multiple times to verify no memory corruption across runs
            for (int i = 0; i < 3; i++)
            {
                using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, inputShape);
                using var results = session.Run(
                    runOptions,
                    new[] { inputName },
                    new[] { inputOrtValue },
                    session.OutputNames);

                Assert.Single(results);

                var output = results[0];
                Assert.True(output.IsTensor);

                var typeShape = output.GetTensorTypeAndShape();
                Assert.Equal(TensorElementType.Float, typeShape.ElementDataType);
                Assert.Equal(expectedShape, typeShape.Shape);

                var resultArray = output.GetTensorDataAsSpan<float>().ToArray();
                Assert.Equal(expectedOutput.Length, resultArray.Length);
                Assert.Equal(expectedOutput, resultArray, new FloatComparer());
            }
        }
        finally
        {
            ortEnvInstance.UnregisterExecutionProviderLibrary(CudaPluginEpName);
        }
    }

    [SkippableFact]
    public void CudaPluginEp_DeviceProperties()
    {
        Skip.IfNot(IsCudaPluginEpAvailable(), "CUDA Plugin EP library not available.");

        string libPath = GetCudaPluginLibraryPath();

        ortEnvInstance.RegisterExecutionProviderLibrary(CudaPluginEpName, libPath);
        try
        {
            var epDevices = ortEnvInstance.GetEpDevices();
            var cudaPluginDevice = epDevices.First(
                d => d.EpName == CudaPluginEpName);

            // Validate device properties
            Assert.NotEmpty(cudaPluginDevice.EpName);
            Assert.NotEmpty(cudaPluginDevice.EpVendor);

            var hwDevice = cudaPluginDevice.HardwareDevice;
            Assert.Equal(OrtHardwareDeviceType.GPU, hwDevice.Type);

            var metadata = cudaPluginDevice.EpMetadata;
            Assert.NotNull(metadata);

            var options = cudaPluginDevice.EpOptions;
            Assert.NotNull(options);
        }
        finally
        {
            ortEnvInstance.UnregisterExecutionProviderLibrary(CudaPluginEpName);
        }
    }

    [SkippableFact]
    public void CudaPluginEp_WithProviderOptions()
    {
        Skip.IfNot(IsCudaPluginEpAvailable(), "CUDA Plugin EP library not available.");

        string libPath = GetCudaPluginLibraryPath();

        ortEnvInstance.RegisterExecutionProviderLibrary(CudaPluginEpName, libPath);
        try
        {
            var epDevices = ortEnvInstance.GetEpDevices();
            var cudaPluginDevices = epDevices
                .Where(d => d.EpName == CudaPluginEpName)
                .ToList();

            Assert.NotEmpty(cudaPluginDevices);

            using var sessionOptions = new SessionOptions();

            // Pass provider options (e.g., device_id)
            var epOptions = new Dictionary<string, string>
            {
                { "device_id", "0" }
            };

            sessionOptions.AppendExecutionProvider(ortEnvInstance, cudaPluginDevices, epOptions);

            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            using var session = new InferenceSession(model, sessionOptions);
            Assert.NotNull(session);
        }
        finally
        {
            ortEnvInstance.UnregisterExecutionProviderLibrary(CudaPluginEpName);
        }
    }

    [SkippableFact]
    public void CudaPluginEp_AutoEpSelection()
    {
        Skip.IfNot(IsCudaPluginEpAvailable(), "CUDA Plugin EP library not available.");

        string libPath = GetCudaPluginLibraryPath();

        ortEnvInstance.RegisterExecutionProviderLibrary(CudaPluginEpName, libPath);
        try
        {
            using var sessionOptions = new SessionOptions();

            // Use automatic EP selection which should pick up the registered CUDA Plugin EP
            sessionOptions.SetEpSelectionPolicy(ExecutionProviderDevicePolicy.PREFER_GPU);

            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            using var session = new InferenceSession(model, sessionOptions);
            Assert.NotNull(session);
        }
        finally
        {
            ortEnvInstance.UnregisterExecutionProviderLibrary(CudaPluginEpName);
        }
    }

    [SkippableFact]
    public void CudaPluginEp_RunWithIoBinding()
    {
        Skip.IfNot(IsCudaPluginEpAvailable(), "CUDA Plugin EP library not available.");

        string libPath = GetCudaPluginLibraryPath();

        ortEnvInstance.RegisterExecutionProviderLibrary(CudaPluginEpName, libPath);
        try
        {
            var epDevices = ortEnvInstance.GetEpDevices();
            var cudaPluginDevices = epDevices
                .Where(d => d.EpName == CudaPluginEpName)
                .ToList();

            Assert.NotEmpty(cudaPluginDevices);

            using var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider(ortEnvInstance, cudaPluginDevices, null);

            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            using var session = new InferenceSession(model, sessionOptions);

            float[] inputData = TestDataLoader.LoadTensorFromEmbeddedResource("bench.in");
            float[] expectedOutput = TestDataLoader.LoadTensorFromEmbeddedResource("bench.expected_out");
            long[] expectedShape = { 1, 1000, 1, 1 };

            var inputMeta = session.InputMetadata;
            var inputName = inputMeta.Keys.First();
            var inputShape = Array.ConvertAll(inputMeta[inputName].Dimensions, d => (long)d);
            var outputName = session.OutputNames[0];

            using var runOptions = new RunOptions();
            using var ioBinding = session.CreateIoBinding();

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, inputShape);
            ioBinding.BindInput(inputName, inputOrtValue);
            ioBinding.BindOutputToDevice(outputName, OrtMemoryInfo.DefaultInstance);

            ioBinding.SynchronizeBoundInputs();

            using var results = session.RunWithBoundResults(runOptions, ioBinding);
            ioBinding.SynchronizeBoundOutputs();

            Assert.Single(results);

            var output = results.First();
            Assert.True(output.IsTensor);

            var typeShape = output.GetTensorTypeAndShape();
            Assert.Equal(TensorElementType.Float, typeShape.ElementDataType);
            Assert.Equal(expectedShape, typeShape.Shape);

            var resultArray = output.GetTensorDataAsSpan<float>().ToArray();
            Assert.Equal(expectedOutput.Length, resultArray.Length);
            Assert.Equal(expectedOutput, resultArray, new FloatComparer());
        }
        finally
        {
            ortEnvInstance.UnregisterExecutionProviderLibrary(CudaPluginEpName);
        }
    }
}

#endif