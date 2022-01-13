using System;
using System.Reflection;
using System.Threading.Tasks;
using Android.App;
using Android.OS;
using Java.Interop;
using Microsoft.ML.OnnxRuntime.Tests.Devices;
using Xunit.Runners;
using Xunit.Runners.UI;
using Xunit.Sdk;

namespace Microsoft.ML.OnnxRuntime.Tests.Droid
{
    [Activity(Label = "@string/app_name", Theme = "@style/AppTheme", MainLauncher = true)]
    public class MainActivity : RunnerActivity
    {
        OnnxRuntimeResultChannel _resultChannel = new OnnxRuntimeResultChannel();

        protected override void OnCreate(Bundle bundle)
        {
            AddExecutionAssembly(typeof(ExtensibilityPointFactory).Assembly);
            AddTestAssembly(Assembly.GetExecutingAssembly());
            ResultChannel = _resultChannel;

            base.OnCreate(bundle);
        }

        [Export("GetTestResults")]
        public Java.Lang.String GetTestResults()
        {
            Java.Lang.String results = null;

            try
            {
                var serializedResults = _resultChannel.GetResults();
                results = new Java.Lang.String(serializedResults);
            }
            catch (Exception ex)
            {
                Android.Util.Log.Error(nameof(MainActivity), ex.Message);
            }

            return results;
        }
    }

    public class OnnxRuntimeResultChannel : ITestListener, IResultChannel
    {
        TestResultProcessor _resultProcessor = new TestResultProcessor();

        public string GetResults()
            => _resultProcessor?.GetSerializedResults();

        public Task CloseChannel()
            => Task.CompletedTask;

        public Task<bool> OpenChannel(string message = null)
        {
            if (_resultProcessor?.Results.Count > 0)
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
                    throw new NotImplementedException();
            }
        }
    }
}