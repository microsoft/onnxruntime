using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using Xunit.Abstractions;

namespace Microsoft.ML.OnnxRuntime.Tests.Devices
{
    public enum TestOutcome
    {
        Passed,
        Failed,
        Skipped,
        NotRun
    }

    public class TestResultSummary
    {
        public int TestCount { get; private set; }
        public int Succeeded { get; private set; }
        public int Skipped { get; private set; }
        public int Failed { get; private set; }
        public int NotRun { get; private set; }
        public IList<TestResult> TestResults { get; private set; }

        internal TestResultSummary(IList<TestResult> results)
        {
            TestResults = results == null ? new List<TestResult>() : results;
            TestCount = TestResults.Count;
            Succeeded = TestResults.Count(i => i.TestOutcome == TestOutcome.Passed);
            Skipped = TestResults.Count(i => i.TestOutcome == TestOutcome.Skipped);
            Failed = TestResults.Count(i => i.TestOutcome == TestOutcome.Failed);
            NotRun = TestResults.Count(i => i.TestOutcome == TestOutcome.NotRun);
        }
    }

    public class TestResult
    {
        internal TestOutcome TestOutcome { get; set; } = TestOutcome.NotRun;
        public string TestId { get; set; }
        public string TestName { get; set; }
        public string Duration { get; set; }
        public string Outcome => TestOutcome.ToString().ToUpper();
        public string Output { get; set; }
    }

    public class TestResultProcessor
    {
        List<TestResult> _results;
        JsonSerializerOptions _serializerOptions = new JsonSerializerOptions { WriteIndented = true };

        List<TestResult> Results
        {
            get => _results == null ? (_results = new List<TestResult>()) : _results;
            set => _results = value;
        }

        internal void RecordResult(TestResult test)
            => Results.Add(test);

        public void RecordResult(ITestResultMessage testResult, ITestCase testCase, TestOutcome outcome)
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

        public TestResultSummary GetResults()
            => new TestResultSummary(Results);

        public string GetSerializedResults()
        {
            var resultSummary = GetResults();
            var serializedResultSummary = JsonSerializer.Serialize(resultSummary, _serializerOptions);
            return serializedResultSummary;
        }
    }
}