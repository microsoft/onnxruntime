// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Xunit.Abstractions;
using Xunit.Sdk;

internal static class CSharpTestRunner
{
    private sealed class Options : ITestFrameworkDiscoveryOptions, ITestFrameworkExecutionOptions
    {
        private readonly Dictionary<string, object> values = new Dictionary<string, object>
        {
            ["xunit.execution.DisableParallelization"] = true,
        };

        public T GetValue<T>(string name) =>
            values.TryGetValue(name, out object value) ? (T)value : default(T);

        public void SetValue<T>(string name, T value) => values[name] = value;
    }

    private sealed class Sink : IMessageSink, IDisposable
    {
        internal readonly List<ITestCase> Cases = new List<ITestCase>();
        internal readonly ManualResetEventSlim Done = new ManualResetEventSlim();
        internal int Executed;
        internal int Failed;
        internal int Skipped;
        internal int Errors;

        public bool OnMessage(IMessageSinkMessage message)
        {
            if (message is ITestCaseDiscoveryMessage discovered)
                Cases.Add(discovered.TestCase);
            if (message is ITestPassed passed)
                Console.WriteLine("PASS " + passed.Test.DisplayName);
            if (message is ITestFailed failed)
            {
                Console.Error.WriteLine("FAIL " + failed.Test.DisplayName);
                Console.Error.WriteLine(string.Join(Environment.NewLine, failed.Messages));
                Console.Error.WriteLine(string.Join(Environment.NewLine, failed.StackTraces));
            }
            if (message is ITestSkipped skipped)
                Console.WriteLine("SKIP " + skipped.Test.DisplayName + ": " + skipped.Reason);
            if (message is IFailureInformation error && !(message is ITestFailed))
            {
                ++Errors;
                Console.Error.WriteLine(string.Join(Environment.NewLine, error.Messages));
                Console.Error.WriteLine(string.Join(Environment.NewLine, error.StackTraces));
            }
            if (message is IDiscoveryCompleteMessage)
                Done.Set();
            if (message is ITestAssemblyFinished finished)
            {
                Executed = finished.TestsRun;
                Failed = finished.TestsFailed;
                Skipped = finished.TestsSkipped;
                Done.Set();
            }
            return true;
        }

        public void Dispose() => Done.Dispose();
    }

    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("Expected the runtime library path. Select tests with --test_filter=Namespace.Class[.Method] (exact names).");
            return 2;
        }
        string filter = Environment.GetEnvironmentVariable("TESTBRIDGE_TEST_ONLY");
        string libraryPath = Path.GetFullPath(args[0]);
        string[] fixtures = args.Skip(1).Select(Path.GetFullPath).ToArray();
        string workingDirectory = Environment.GetEnvironmentVariable("TEST_TMPDIR");
        if (string.IsNullOrEmpty(workingDirectory))
            throw new InvalidOperationException("TEST_TMPDIR is missing. Run this executable through bazel test to provide an isolated writable fixture directory.");
        Directory.CreateDirectory(workingDirectory);
        foreach (string fixture in fixtures)
            File.Copy(fixture, Path.Combine(workingDirectory, Path.GetFileName(fixture)), true);
        Directory.SetCurrentDirectory(workingDirectory);
        NativeLibrary.SetDllImportResolver(
            typeof(InferenceSession).Assembly,
            (name, assembly, searchPath) =>
                name == "onnxruntime" ? NativeLibrary.Load(libraryPath) : IntPtr.Zero);
        Assembly tests = Assembly.GetExecutingAssembly();
        using var diagnostics = new Sink();
        using var framework = new XunitTestFramework(diagnostics);
        using var discovery = new Sink();
        var options = new Options();
        using var discoverer = framework.GetDiscoverer(new ReflectionAssemblyInfo(tests));
        discoverer.Find(false, discovery, options);
        discovery.Done.Wait();
        var selected = discovery.Cases.Where(test =>
            string.IsNullOrEmpty(filter) ||
            test.TestMethod.TestClass.Class.Name == filter ||
            test.TestMethod.TestClass.Class.Name + "." + test.TestMethod.Method.Name == filter).ToList();
        if (discovery.Errors != 0 || diagnostics.Errors != 0 || selected.Count == 0)
        {
            Console.Error.WriteLine($"Discovery failed or no tests matched '{filter}'. Use an exact fully qualified class or class.method name.");
            return 2;
        }
        Console.WriteLine($"Discovered {discovery.Cases.Count}; selected {selected.Count} xUnit test cases.");
        using var execution = new Sink();
        using var executor = framework.GetExecutor(tests.GetName());
        executor.RunTests(selected, execution, options);
        execution.Done.Wait();
        int errors = execution.Errors + diagnostics.Errors;
        Console.WriteLine($"Tests run: {execution.Executed}, failed: {execution.Failed}, skipped: {execution.Skipped}, errors: {errors}");
        if (execution.Executed <= execution.Skipped)
            Console.Error.WriteLine("No non-skipped tests executed. Select a supported desktop CPU test.");
        return execution.Failed == 0 && errors == 0 && execution.Executed > execution.Skipped ? 0 : 1;
    }
}
