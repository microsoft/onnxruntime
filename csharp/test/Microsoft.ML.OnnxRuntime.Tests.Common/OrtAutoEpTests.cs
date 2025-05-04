﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// not supported on mobile platforms
#if !(ANDROID || IOS)

namespace Microsoft.ML.OnnxRuntime.Tests;

using System;
using System.Linq;
using System.IO;
using System.Runtime.InteropServices;
using Xunit;
using System.Collections.Generic;

/// <summary>
/// Tests for auto ep selection/registration.
/// Includes testing of OrtHardwareDevice and OrtEpDevice as those only come from auto ep related code and we only
/// get read-only access to them (i.e. we can't directly create instances of them to test).
/// </summary>
public class OrtAutoEpTests
{
    private OrtEnv ortEnvInstance = OrtEnv.Instance();

    private void ReadHardwareDeviceValues(OrtHardwareDevice device)
    {
        Assert.True(device.Type == OrtHardwareDeviceType.CPU ||
                    device.Type == OrtHardwareDeviceType.GPU ||
                    device.Type == OrtHardwareDeviceType.NPU);
        if (device.Type == OrtHardwareDeviceType.CPU)
        {
            Assert.NotEmpty(device.Vendor);
        }
        else
        {
            Assert.True(device.VendorId != 0);
            Assert.True(device.DeviceId != 0);
        }

        var metadata = device.Metadata;
        Assert.NotNull(metadata);
        foreach (var kvp in metadata.Entries)
        {
            Assert.NotEmpty(kvp.Key);
            // Assert.NotEmpty(kvp.Value); this is allowed
        }
    }

    [Fact]
    public void GetEpDevices()
    {
        var epDevices = ortEnvInstance.GetEpDevices();
        Assert.NotNull(epDevices);
        Assert.NotEmpty(epDevices);
        foreach (var ep_device in epDevices)
        {
            Assert.NotEmpty(ep_device.EpName);
            Assert.NotEmpty(ep_device.EpVendor);
            var metadata = ep_device.EpMetadata;
            Assert.NotNull(metadata);
            var options = ep_device.EpOptions;
            Assert.NotNull(options);
            ReadHardwareDeviceValues(ep_device.HardwareDevice);
        }
    }

    [Fact]
    public void RegisterUnregisterLibrary()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            string libFullPath = Path.Combine(Directory.GetCurrentDirectory(), "example_plugin_ep.dll");
            Assert.True(File.Exists(libFullPath), $"Expected lib {libFullPath} does not exist.");

            // example plugin ep uses the registration name as the ep name
            const string epName = "csharp_ep";

            // register. shouldn't throw
            ortEnvInstance.RegisterExecutionProviderLibrary(epName, libFullPath);

            // check OrtEpDevice was found
            var epDevices = ortEnvInstance.GetEpDevices();
            var found = epDevices.Any(d => string.Equals(epName, d.EpName, StringComparison.OrdinalIgnoreCase));
            Assert.True(found);

            // unregister
            ortEnvInstance.UnregisterExecutionProviderLibrary(epName);
        }
    }

    [Fact]
    public void AppendToSessionOptionsV2()
    {
        var runTest = (Func<Dictionary<string, string>> getEpOptions) =>
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;

            var epDevices = ortEnvInstance.GetEpDevices();

            // cpu ep ignores the provider options so we can use any value in epOptions and it won't break.
            List<OrtEpDevice> selectedEpDevices = epDevices.Where(d => d.EpName == "CPUExecutionProvider").ToList();

            Dictionary<string, string> epOptions = getEpOptions();
            sessionOptions.AppendExecutionProvider(ortEnvInstance, selectedEpDevices, epOptions);

            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

            // session should load successfully
            using (var session = new InferenceSession(model))
            {
                Assert.NotNull(session);
            }
        };

        runTest(() =>
        {
            // null options
            return null;
        });

        runTest(() =>
        {
            // empty options
            return new Dictionary<string, string>();
        });

        runTest(() =>
        {
            // dummy options
            return new Dictionary<string, string>
            {
                { "random_key", "value" },
            };
        });
    }

    [Fact]
    public void SetEpSelectionPolicy()
    {
        SessionOptions sessionOptions = new SessionOptions();
        sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;

        var epDevices = ortEnvInstance.GetEpDevices();
        Assert.NotEmpty(epDevices);

        // doesn't matter what the value is. should fallback to ORT CPU EP
        sessionOptions.SetEpSelectionPolicy(ExecutionProviderDevicePolicy.PREFER_GPU);

        var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");

        // session should load successfully
        using (var session = new InferenceSession(model))
        {
            Assert.NotNull(session);
        }
    }
}
#endif