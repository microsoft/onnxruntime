using System;
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
#if __NATIVE_DEPENDENCIES_EXIST__
        OnnxRuntimeResultChannel _resultChannel = new OnnxRuntimeResultChannel();
#endif

        public override bool FinishedLaunching(UIApplication app, NSDictionary options)
        {
            Xamarin.Calabash.Start();

            AddExecutionAssembly(typeof(ExtensibilityPointFactory).Assembly);

#if __NATIVE_DEPENDENCIES_EXIST__
            AddTestAssembly(Assembly.GetExecutingAssembly());
            ResultChannel = _resultChannel;
#endif

            return base.FinishedLaunching(app, options);
        }

        [Export("getTestResults")]
        public NSString GetTestResults()
        {
            NSString results = null;

            try
            {
#if __NATIVE_DEPENDENCIES_EXIST__
                var serializedResults = _resultChannel.GetResults();
                if (serializedResults != null) results = new NSString(serializedResults);
#endif
            }
            catch (Exception ex)
            {
                CoreFoundation.OSLog.Default.Log(CoreFoundation.OSLogLevel.Error, ex.Message);
            }

            return results;
        }
    }

#if __NATIVE_DEPENDENCIES_EXIST__
    public class OnnxRuntimeResultChannel : ITestListener, IResultChannel
    {
        TestResultProcessor _resultProcessor;

        public string GetResults()
            => _resultProcessor.GetSerializedResults();

        public Task CloseChannel()
            => Task.CompletedTask;

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