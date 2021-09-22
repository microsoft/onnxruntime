using System.Collections.Generic;
using System.Linq;

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

        public TestResultSummary() {}

        public TestResultSummary(IList<TestResult> results)
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
        public TestOutcome TestOutcome { get; set; } = TestOutcome.NotRun;
        public string TestId { get; set; }
        public string TestName { get; set; }
        public string Duration { get; set; }
        public string Outcome => TestOutcome.ToString().ToUpper();
        public string Output { get; set; }
    }
}