// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Program.cs
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Qualcomm.ML.OnnxRuntime.QNN;

namespace QnnEpNuGetTest
{
    class Program
    {
        private static string InputModelPath = "cmake\\external\\onnx\\onnx\\backend\\test\\data\\node\\test_averagepool_2d_default\\model.onnx";
        private const int Batch = 1;
        private const int Channel = 3;
        private const int Height = 32;
        private const int Width = 32;

        static void Main(string[] args)
        {
            // Create or get the global ONNX Runtime environment
            using var env = OrtEnv.Instance();

            // Get EP library path
            string epLibPath = QnnEpHelper.GetLibraryPath();
            Console.WriteLine($"EP Library: {epLibPath}");
            // Get QNN HTP path
            string htpLibPath = QnnEpHelper.GetQnnHtpLibraryPath();
            Console.WriteLine($"HTP Library: {htpLibPath}");
            // Get QNN CPU path
            string cpuLibPath = QnnEpHelper.GetQnnCpuLibraryPath();
            Console.WriteLine($"CPU Library: {cpuLibPath}");
            // Get all EP names
            var epNames = QnnEpHelper.GetEpNames();
            foreach (var availableEpName in epNames)
            {
                Console.WriteLine($"Available EP Name: {availableEpName}");
            }
            // Get EP name
            string epName = QnnEpHelper.GetEpName();
            Console.WriteLine($"EP Name: {epName}");

            // Register the QNN execution provider library
            env.RegisterExecutionProviderLibrary(
                "QnnExecutionProvider",
                epLibPath);

            var epDevices = env.GetEpDevices();
            var selectedEpDevices = epDevices.Where(epDevice => epDevice.EpName == epName).ToList();
            if (selectedEpDevices.Count == 0)
            {
                Console.WriteLine("No matching devices found");
            }
            else
            {
                Console.WriteLine($"Found {selectedEpDevices.Count} matching devices");

                // Build SessionOptions and enable QNN EP
                using var so = new SessionOptions();
                var qnnProviderOptions = new Dictionary<string, string>
                {
                    { "backend_path", htpLibPath },
                };
                so.AppendExecutionProvider(env, selectedEpDevices, qnnProviderOptions);
                so.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");

                Console.WriteLine($"Input Model Path: {InputModelPath}");
                if (!File.Exists(InputModelPath))
                {
                    throw new FileNotFoundException($"Model file not found: {InputModelPath}");
                }
                using var session = new InferenceSession(InputModelPath, so);
                var inputInfo = session.InputMetadata;
                foreach (var kv in inputInfo)
                {
                    var name = kv.Key;
                    var dims = kv.Value.Dimensions;
                    var shape = new int[] { Batch, Channel, Height, Width };

                    // Create a 1D array to hold all elements (Batch * Channel * Height * Width)
                    var data = new float[Batch * Channel * Height * Width];
                    var inputTensor = new DenseTensor<float>(data, shape);

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(name, inputTensor)
                    };

                    // Run inference
                    using var results = session.Run(inputs);

                    // Read outputs
                    foreach (var output in results)
                    {
                        Console.WriteLine($"Output: {output.Name}");
                        if (output.Value is DenseTensor<float> t)
                        {
                            Console.WriteLine($"  Shape: [{string.Join(",", t.Dimensions.ToArray())}]");
                        }
                        else
                        {
                            Console.WriteLine($"  Type: {output.Value?.GetType().FullName}");
                        }
                    }
                }
            }
            env.UnregisterExecutionProviderLibrary("QnnExecutionProvider");
        }
    }
}
