// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// not supported on mobile platforms
#if !(ANDROID || IOS)

namespace Microsoft.ML.OnnxRuntime.Tests;

using System;
using System.Linq;
using Xunit;
using System.Collections.Generic;
using Google.Protobuf;
using Onnx;

public class EpCompatibilityTests
{
    private readonly OrtEnv ortEnvInstance = OrtEnv.Instance();

    private IReadOnlyList<OrtEpDevice> GetDevices()
    {
        var epDevices = ortEnvInstance.GetEpDevices();
        Assert.NotNull(epDevices);
        Assert.NotEmpty(epDevices);
        return epDevices;
    }

    /// <summary>
    /// Creates a minimal valid ONNX ModelProto with optional compatibility metadata.
    /// </summary>
    private static byte[] CreateModelWithCompatibilityMetadata(
        Dictionary<string, string> epCompatibilityInfo = null)
    {
        var modelProto = new ModelProto();
        modelProto.IrVersion = (long)Onnx.Version.IrVersion;
        modelProto.Graph = new GraphProto { Name = "test_graph" };

        var opset = new OperatorSetIdProto();
        opset.Domain = "";
        opset.Version = 13;
        modelProto.OpsetImport.Add(opset);

        if (epCompatibilityInfo != null)
        {
            foreach (var kvp in epCompatibilityInfo)
            {
                var prop = new StringStringEntryProto();
                prop.Key = "ep_compatibility_info." + kvp.Key;
                prop.Value = kvp.Value;
                modelProto.MetadataProps.Add(prop);
            }
        }

        return modelProto.ToByteArray();
    }

    [Fact]
    public void GetEpCompatibility_InvalidArgs()
    {
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetModelCompatibilityForEpDevices(null, "info"));
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetModelCompatibilityForEpDevices(new List<OrtEpDevice>(), "info"));
    }

    [Fact]
    public void GetEpCompatibility_SingleDeviceCpuProvider()
    {
        var devices = GetDevices();
        var someInfo = "arbitrary-compat-string";

        // Use CPU device 
        var cpu = devices.First(d => d.EpName == "CPUExecutionProvider");
        Assert.NotNull(cpu);
        var selected = new List<OrtEpDevice> { cpu };
        var status = ortEnvInstance.GetModelCompatibilityForEpDevices(selected, someInfo);

        // CPU defaults to not applicable in this scenario
        Assert.Equal(OrtCompiledModelCompatibility.EP_NOT_APPLICABLE, status);
    }

    [Fact]
    public void GetCompatibilityInfoFromModel_InvalidArgs()
    {
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModel(null, "TestEP"));
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModel("", "TestEP"));
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModel("model.onnx", null));
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModel("model.onnx", ""));
    }

    [Fact]
    public void GetCompatibilityInfoFromModel_FileNotFound()
    {
        Assert.Throws<OnnxRuntimeException>(
            () => ortEnvInstance.GetCompatibilityInfoFromModel("nonexistent_model_path.onnx", "TestEP"));
    }

    [Fact]
    public void GetCompatibilityInfoFromModelBytes_InvalidArgs()
    {
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModelBytes(null, "TestEP"));
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModelBytes(new byte[0], "TestEP"));
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModelBytes(new byte[] { 1, 2, 3 }, null));
        Assert.Throws<ArgumentException>(() => ortEnvInstance.GetCompatibilityInfoFromModelBytes(new byte[] { 1, 2, 3 }, ""));
    }

    [Fact]
    public void GetCompatibilityInfoFromModel_WithMetadata()
    {
        const string epType = "TestCompatEP";
        const string expectedCompatInfo = "test_compat_v1.0_driver_123";

        byte[] modelData = CreateModelWithCompatibilityMetadata(
            new Dictionary<string, string> { { epType, expectedCompatInfo } });

        string tempModelPath = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            System.IO.Path.GetRandomFileName() + ".onnx");

        System.IO.File.WriteAllBytes(tempModelPath, modelData);

        try
        {
            string result = ortEnvInstance.GetCompatibilityInfoFromModel(tempModelPath, epType);
            Assert.NotNull(result);
            Assert.Equal(expectedCompatInfo, result);
        }
        finally
        {
            if (System.IO.File.Exists(tempModelPath))
            {
                System.IO.File.Delete(tempModelPath);
            }
        }
    }

    [Fact]
    public void GetNumHardwareDevices_ReturnsPositive()
    {
        var numDevices = ortEnvInstance.GetNumHardwareDevices();
        // Should return at least one device (CPU)
        Assert.True(numDevices > 0, "Expected at least one hardware device");
    }

    [Fact]
    public void GetHardwareDevices_ReturnsDevices()
    {
        var devices = ortEnvInstance.GetHardwareDevices();
        Assert.NotNull(devices);
        Assert.NotEmpty(devices);

        // Each device should have valid properties
        foreach (var device in devices)
        {
            Assert.NotNull(device);
            // Device type should be valid (CPU, GPU, or NPU)
            var deviceType = device.Type;
            Assert.True(
                deviceType == OrtHardwareDeviceType.CPU ||
                deviceType == OrtHardwareDeviceType.GPU ||
                deviceType == OrtHardwareDeviceType.NPU,
                $"Unexpected device type: {deviceType}");

            // Vendor should not be null
            Assert.NotNull(device.Vendor);
        }
    }

    [Fact]
    public void GetCompatibilityInfoFromModelBytes_InvalidModelData()
    {
        byte[] invalidData = System.Text.Encoding.UTF8.GetBytes("this is not a valid ONNX model");
        Assert.Throws<OnnxRuntimeException>(
            () => ortEnvInstance.GetCompatibilityInfoFromModelBytes(invalidData, "TestEP"));
    }

    [Fact]
    public void GetCompatibilityInfoFromModelBytes_WithMetadata()
    {
        const string epType = "TestCompatEP";
        const string expectedCompatInfo = "test_compat_v1.0_driver_123";

        byte[] modelData = CreateModelWithCompatibilityMetadata(
            new Dictionary<string, string> { { epType, expectedCompatInfo } });

        string result = ortEnvInstance.GetCompatibilityInfoFromModelBytes(modelData, epType);
        Assert.NotNull(result);
        Assert.Equal(expectedCompatInfo, result);
    }

    [Fact]
    public void GetCompatibilityInfoFromModelBytes_NotFound()
    {
        // Create model with metadata for a different EP
        byte[] modelData = CreateModelWithCompatibilityMetadata(
            new Dictionary<string, string> { { "DifferentEP", "some_value" } });

        string result = ortEnvInstance.GetCompatibilityInfoFromModelBytes(modelData, "NonExistentEP");
        Assert.Null(result);
    }

    [Fact]
    public void GetCompatibilityInfoFromModelBytes_NoMetadata()
    {
        // Create model without any compatibility metadata
        byte[] modelData = CreateModelWithCompatibilityMetadata();

        string result = ortEnvInstance.GetCompatibilityInfoFromModelBytes(modelData, "AnyEP");
        Assert.Null(result);
    }

    [Fact]
    public void GetHardwareDeviceEpIncompatibilityDetails_CpuEp()
    {
        var devices = ortEnvInstance.GetHardwareDevices();
        Assert.NotNull(devices);
        Assert.NotEmpty(devices);

        // Find CPU device
        var cpuDevice = devices.FirstOrDefault(d => d.Type == OrtHardwareDeviceType.CPU);
        Assert.NotNull(cpuDevice);

        // Get incompatibility details for CPU EP with CPU device
        using (var details = ortEnvInstance.GetHardwareDeviceEpIncompatibilityDetails("CPUExecutionProvider", cpuDevice))
        {
            // CPU EP should be compatible with CPU device (no incompatibility reasons)
            Assert.Equal(OrtDeviceEpIncompatibilityReason.None, details.ReasonsBitmask);
            Assert.Equal(0, details.ErrorCode);
        }
    }

    [Fact]
    public void GetHardwareDeviceEpIncompatibilityDetails_InvalidEpName()
    {
        var devices = ortEnvInstance.GetHardwareDevices();
        Assert.NotNull(devices);
        Assert.NotEmpty(devices);

        var firstDevice = devices[0];
        Assert.NotNull(firstDevice);

        // Invalid EP name should throw
        Assert.Throws<ArgumentException>(() =>
            ortEnvInstance.GetHardwareDeviceEpIncompatibilityDetails("", firstDevice));
        Assert.Throws<ArgumentNullException>(() =>
            ortEnvInstance.GetHardwareDeviceEpIncompatibilityDetails(null, firstDevice));
    }
}
#endif
