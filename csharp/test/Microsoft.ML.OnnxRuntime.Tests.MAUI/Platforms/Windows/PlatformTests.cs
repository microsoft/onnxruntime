// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI
{
    public class PlatformTests
    {
        // Semi-random test to check we can add an EP
        [Fact(DisplayName = "CPU_EP_NoArena")]
        public void TestEnableCpuEPWithNoArena()
        {
            var opt = new SessionOptions();
            int useArena = 0;
            opt.AppendExecutionProvider_CPU(useArena);
        }
    }
}

