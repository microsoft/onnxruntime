using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.EP.WebGpu;

class Program
{
    static int Main()
    {
        string epLibPath = WebGpuEp.GetLibraryPath();
        string epRegistrationName = "webgpu_ep_registration";
        string epName = WebGpuEp.GetEpName();

        Console.WriteLine($"WebGPU EP library path: {epLibPath}");

        var env = OrtEnv.Instance();
        env.RegisterExecutionProviderLibrary(epRegistrationName, epLibPath);
        Console.WriteLine($"Registered EP library: {epLibPath}");

        try
        {
            // Find the OrtEpDevice for the WebGPU EP
            OrtEpDevice? epDevice = null;
            foreach (var d in env.GetEpDevices())
            {
                if (string.Equals(epName, d.EpName, StringComparison.Ordinal))
                {
                    epDevice = d;
                }
            }

            if (epDevice == null)
            {
                Console.Error.WriteLine($"ERROR: Unable to find OrtEpDevice with name '{epName}'");
                return 1;
            }
            Console.WriteLine($"Found OrtEpDevice for EP: {epName}");

            // Create session with WebGPU EP
            using var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider(env, new[] { epDevice }, new Dictionary<string, string>());
            sessionOptions.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");

            string inputModelPath = Path.Combine(AppContext.BaseDirectory, "mul.onnx");
            Console.WriteLine($"Loading model: {inputModelPath}");

            using var session = new InferenceSession(inputModelPath, sessionOptions);

            // Run model: mul(x, y) = x * y
            float[] inputData = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(inputData, new long[] { 2, 3 });
            var inputValues = new List<OrtValue> { inputOrtValue, inputOrtValue }.AsReadOnly();
            var inputNames = new List<string> { "x", "y" }.AsReadOnly();
            using var runOptions = new RunOptions();

            using var outputs = session.Run(runOptions, inputNames, inputValues, session.OutputNames);

            float[] expected = { 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f };
            var actual = outputs[0].GetTensorDataAsSpan<float>().ToArray();

            Console.WriteLine($"Input:    {string.Join(", ", inputData)}");
            Console.WriteLine($"Output:   {string.Join(", ", actual)}");
            Console.WriteLine($"Expected: {string.Join(", ", expected)}");

            // Validate output
            for (int i = 0; i < expected.Length; i++)
            {
                if (Math.Abs(actual[i] - expected[i]) > 1e-5f)
                {
                    Console.Error.WriteLine($"ERROR: Output mismatch at index {i}: expected {expected[i]}, got {actual[i]}");
                    return 1;
                }
            }

            Console.WriteLine("PASSED: All outputs match expected values.");
            return 0;
        }
        finally
        {
            env.UnregisterExecutionProviderLibrary(epRegistrationName);
        }
    }
}
