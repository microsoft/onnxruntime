// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI
{
    public class PlatformTests
    {
        // All the 'AppendExecutionProvider' calls will throw if unsuccessful
        [Fact(DisplayName = "EnableNNAPI")]
        public void TestEnableNNAPI()
        {
            var opt = new SessionOptions();
            opt.AppendExecutionProvider_Nnapi();
        }
    } 
}
