using System.Reflection;
using System.Threading.Tasks;
using Foundation;
using Microsoft.ML.OnnxRuntime.Tests.Devices;
using UIKit;
using Xunit.Runner;
using Xunit.Runners;
using Xunit.Sdk;

namespace Microsoft.ML.OnnxRuntime.Tests.iOS
{
    [Register("AppDelegate")]
    public partial class AppDelegate : RunnerAppDelegate
    {
        public override bool FinishedLaunching(UIApplication app, NSDictionary options)
        {
            AddExecutionAssembly(typeof(ExtensibilityPointFactory).Assembly);

#if __NATIVE_DEPENDENCIES_EXIST__
            AddTestAssembly(Assembly.GetExecutingAssembly());
            ResultChannel = new OnnxRuntimeResultChannel();
#endif

            return base.FinishedLaunching(app, options);
        }
    }

#if __NATIVE_DEPENDENCIES_EXIST__
    public class OnnxRuntimeResultChannel : ITestListener, IResultChannel
    {
        TestResultProcessor _resultProcessor;

        public Task CloseChannel()
        {
            // Serialize result data and push results to a pre-defined endpoint/webhook
            System.Console.WriteLine(_resultProcessor.GetSerializedResults());

            return Task.CompletedTask;
        }

        public Task<bool> OpenChannel(string message = null)
        {
            _resultProcessor = new TestResultProcessor();
            return Task.FromResult(true);
        }

        public void RecordResult(TestResultViewModel result)
            => _resultProcessor?.RecordResult(result.TestResultMessage, result.TestCase.TestCase, GetTestOutcomeFromTestState(result.TestCase.Result));

        TestOutcome GetTestOutcomeFromTestState(TestState state)
        {
            switch (state)
            {
                case TestState.Failed:
                    return TestOutcome.Failed;
                case TestState.NotRun:
                    return TestOutcome.NotRun;
                case TestState.Passed:
                    return TestOutcome.Passed;
                case TestState.Skipped:
                    return TestOutcome.Skipped;
                default:
                    throw new System.NotImplementedException();
            }
        }
    }
#endif
}