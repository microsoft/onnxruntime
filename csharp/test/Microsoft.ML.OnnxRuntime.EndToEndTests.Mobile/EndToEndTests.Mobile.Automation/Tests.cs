using System;
using System.Threading;
using Newtonsoft.Json;
using NUnit.Framework;
using Xamarin.UITest;

namespace EndToEndTests.Mobile.Automation
{
    [TestFixture(Platform.Android)]
    [TestFixture(Platform.iOS)]
    public class Tests
    {
        IApp _app;
        Platform _platform;
        string _getResultsBackdoorMethodName;
        string _activityIndicatorClassName;

        public Tests(Platform platform)
        {
            _platform = platform;
            _activityIndicatorClassName = platform == Platform.Android ? "ActivityIndicatorRenderer" : "Xamarin_Forms_Platform_iOS_ActivityIndicatorRenderer";
            _getResultsBackdoorMethodName = platform == Platform.Android ? "GetTestResults" : "getTestResults";
        }

        [SetUp]
        public void BeforeEachTest()
            => _app = AppInitializer.StartApp(_platform);

        [Test]
        public void RunPlatformUnitTest()
        {
            _app.Screenshot("Pre-testing");
            _app.WaitForElement(i => i.Marked("Run Everything"));
            _app.Tap(i => i.Marked("Run Everything"));
            _app.WaitForElement(i => i.Class(_activityIndicatorClassName));
            _app.WaitForNoElement(i => i.Class(_activityIndicatorClassName).Child(0), timeout: TimeSpan.FromSeconds(30));
            Thread.Sleep(1000); // Gives the test app sufficient time to prepare the results

            var serializedResultSummary = _app.Invoke(_getResultsBackdoorMethodName)?.ToString();
            Assert.IsNotEmpty(serializedResultSummary, "Test results were not returned");

            var testOutcome = JsonConvert.DeserializeObject<TestOutcome>(serializedResultSummary);
            Assert.AreEqual(testOutcome.Failed, 0, $"{testOutcome.Failed} tests failed");

            _app.Screenshot("Post-testing");
        }
    }

    public class TestOutcome
    {
        public int TestCount { get; set; }
        public int Succeeded { get; set; }
        public int Skipped { get; set; }
        public int Failed { get; set; }
        public int NotRun { get; set; }
    }
}