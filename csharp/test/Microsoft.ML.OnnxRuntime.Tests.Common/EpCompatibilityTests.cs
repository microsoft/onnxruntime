// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// not supported on mobile platforms
#if !(ANDROID || IOS)

namespace Microsoft.ML.OnnxRuntime.Tests;

using System;
using System.Linq;
using Xunit;
using System.Collections.Generic;

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
}
#endif
