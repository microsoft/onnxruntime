using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Linq;
using Newtonsoft.Json;
using Xunit.Abstractions;

namespace Microsoft.ML.OnnxRuntime.Tests.Devices
{
    public class TestResultProcessor
    {
        ConcurrentBag<TestResult> _results = new ConcurrentBag<TestResult>();

        public ConcurrentBag<TestResult> Results
        {
            get => _results == null ? (_results = new ConcurrentBag<TestResult>()) : _results;
            private set => _results = value;
        }

        internal void RecordResult(TestResult test)
            => Results.Add(test);

        public void RecordResult(ITestResultMessage testResult, ITestCase testCase, TestOutcome outcome)
        {
            try
            {
                RecordResult(new TestResult
                {
                    TestId = testCase.UniqueID,
                    TestName = testCase.DisplayName,
                    Output = testResult.Output,
                    TestOutcome = outcome,
                    Duration = TimeSpan.FromSeconds((double)testResult.ExecutionTime).ToString("c", CultureInfo.InvariantCulture)
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceError(ex.Message);
            }
        }

        public TestResultSummary GetResults()
            => new TestResultSummary(Results.ToList());

        public string GetSerializedResults()
        {
            var resultSummary = GetResults();
	    JsonConvert.DefaultSettings = () => new JsonSerializerSettings { MaxDepth = 128 };
            var serializedResultSummary = JsonConvert.SerializeObject(resultSummary, Formatting.Indented);
            return serializedResultSummary;
        }
    }
}
