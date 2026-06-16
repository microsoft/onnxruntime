using System;
using System.Threading;
using Microsoft.ML.OnnxRuntime.Tests.Devices;
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

            // Fix security issue (overflow with too much nesting): GHSA-5crp-9r3c-p9vr
            JsonConvert.DefaultSettings = () => new JsonSerializerSettings { MaxDepth = 128 };
            var testSummary = JsonConvert.DeserializeObject<TestResultSummary>(serializedResultSummary);
            Assert.AreEqual(testSummary.Failed, 0, $"{testSummary.Failed} tests failed");

            _app.Screenshot("Post-testing");
        }
    }
}
