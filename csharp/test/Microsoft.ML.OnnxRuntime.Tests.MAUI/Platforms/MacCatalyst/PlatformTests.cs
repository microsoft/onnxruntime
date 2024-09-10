// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI
{
    public class PlatformTests
    {
        [Fact(DisplayName = "EnableCoreML_NeuralNetwork")]
        public void TestEnableCoreML_NN()
        {
            var opt = new SessionOptions();
            opt.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
        }

        [Fact(DisplayName = "EnableCoreML_MLProgram")]
        public void TestEnableCoreML_MLProgram()
        {
            var opt = new SessionOptions();
            opt.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM);
        }
    }
}
