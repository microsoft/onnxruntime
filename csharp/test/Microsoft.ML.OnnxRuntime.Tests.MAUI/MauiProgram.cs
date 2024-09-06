// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Based on https://github.com/mattleibow/DeviceRunners/tree/main/sample/SampleMauiApp

using DeviceRunners.VisualRunners;
using Microsoft.Extensions.Logging;

#if MODE_XHARNESS
using DeviceRunners.XHarness;
#endif

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder
            // .ConfigureUITesting()
#if MODE_XHARNESS
            .UseXHarnessTestRunner(conf => conf
                .AddTestAssembly(typeof(MauiProgram).Assembly)
                .AddXunit())
#endif
            .UseVisualTestRunner(conf => conf
//#if MODE_NON_INTERACTIVE_VISUAL
//                .EnableAutoStart(true)
//                .AddTcpResultChannel(new TcpResultChannelOptions
//                {
//                    HostNames = ["localhost", "10.0.2.2"],
//                    Port = 16384,
//                    Formatter = new TextResultChannelFormatter(),
//                    Required = false
//                })
//#endif
                .AddConsoleResultChannel()
                .AddTestAssembly(typeof(MauiProgram).Assembly)
                .AddXunit());

#if DEBUG
        builder.Logging.AddDebug();
#endif

        return builder.Build();
    }
}