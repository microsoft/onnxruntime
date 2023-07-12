using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;

namespace MauiModelTester
{
    public enum ExecutionProviders
    {
        CPU,    // CPU execution provider is always available by default
        NNAPI,  // NNAPI is available on Android
        CoreML, // CoreML is available on iOS/macOS
        XNNPACK // XNNPACK is available on ARM/ARM64 platforms and benefits 32-bit float models
    }

    // An inference session runs an ONNX model
    internal class OrtInferenceSession
    {
        public OrtInferenceSession(ExecutionProviders provider = ExecutionProviders.CPU)
        {
            _sessionOptions = new SessionOptions();
            switch (_executionProvider)
            {
                case ExecutionProviders.CPU:
                    break;
                case ExecutionProviders.NNAPI:
                    _sessionOptions.AppendExecutionProvider_Nnapi();
                    break;
                case ExecutionProviders.CoreML:
                    _sessionOptions.AppendExecutionProvider_CoreML();
                    break;
                case ExecutionProviders.XNNPACK:
                    _sessionOptions.AppendExecutionProvider("XNNPACK");
                    break;
            }

            // enable pre/post processing custom operators from onnxruntime-extensions
            _sessionOptions.RegisterOrtExtensions();

            _perfStats = new PerfStats();
        }

        // async task to create the inference session which is an expensive operation.
        public async Task Create()
        {
            // create the InferenceSession. this is an expensive operation so only do this when necessary.
            // the InferenceSession supports multiple calls to Run, including concurrent calls.
            var modelBytes = await Utils.LoadResource("test_data/model.onnx");

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            _inferenceSession = new InferenceSession(modelBytes, _sessionOptions);
            stopwatch.Stop();
            _perfStats.LoadTime = stopwatch.Elapsed;

            (_inputs, _expectedOutputs) = await Utils.LoadTestData();

            // warmup
            Run(1, true);
        }

        public void Run(int iterations = 1, bool isWarmup = false)
        {
            // do all setup outside of the timing
            var runOptions = new RunOptions();
            var outputNames = _inferenceSession.OutputNames;

            _perfStats.ClearRunTimes();

            // var stopwatch = new Stopwatch();

            for (var i = 0; i < iterations; i++)
            {
                // stopwatch.Restart();
                var stopwatch = new Stopwatch();
                stopwatch.Start();

                using IDisposableReadOnlyCollection<OrtValue> results =
                    _inferenceSession.Run(runOptions, _inputs, outputNames);

                stopwatch.Stop();

                if (isWarmup)
                {
                    _perfStats.WarmupTime = stopwatch.Elapsed;

                    // validate the expected output on the first Run only
                    if (_expectedOutputs.Count > 0)
                    {
                        // create dictionary of output name to results
                        var actual = outputNames.Zip(results).ToDictionary(x => x.First, x => x.Second);

                        foreach (var expectedOutput in _expectedOutputs)
                        {
                            var outputName = expectedOutput.Key;
                            Utils.TensorComparer.VerifyTensorResults(outputName, expectedOutput.Value,
                                                                     actual[outputName]);
                        }
                    }
                }
                else
                {
                    _perfStats.AddRunTime(stopwatch.Elapsed);
                }
            }
        }

        public PerfStats PerfStats => _perfStats;

        private SessionOptions _sessionOptions;
        private InferenceSession _inferenceSession;
        private ExecutionProviders _executionProvider = ExecutionProviders.CPU;
        private Dictionary<string, OrtValue> _inputs;
        private Dictionary<string, OrtValue> _expectedOutputs;
        private PerfStats _perfStats;
    }
}
