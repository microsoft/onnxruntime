using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    public partial class InferenceTest
    {
        [Fact(DisplayName = "TestPlatformSessionOptions")]
        public void TestPlatformSessionOptions()
        {
            var opt = new SessionOptions();
            opt.AppendExecutionProvider_Nnapi();
        }
    }
}