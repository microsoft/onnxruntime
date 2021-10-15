using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Microsoft.ML.OnnxRuntime.InferenceSample
{
    public static class InferenceSampleApi
    {
        public static void Execute(SessionOptions options = null)
        {
            var model = LoadModelFromEmbeddedResource("TestData.squeezenet.onnx");

            // Optional : Create session options and set any relevant values.
            // If an additional execution provider is needed it should be added to the SessionOptions prior to
            // creating the InferenceSession. The CPU Execution Provider is always added by default.
            if (options == null)
            {
                options = new SessionOptions { LogId = "Sample" };
            }

            using (var session = new InferenceSession(model, options))
            {
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();

                // this is the data for only one input tensor for this model
                float[] inputData = LoadTensorFromEmbeddedResource("TestData.bench.in");

                foreach (var name in inputMeta.Keys)
                {
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run the inference
                // 'results' is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                using (var results = session.Run(container))
                {
                    // dump the results
                    foreach (var r in results)
                    {
                        Console.WriteLine("Output for {0}", r.Name);
                        Console.WriteLine(r.AsTensor<float>().GetArrayString());
                    }
                }
            }
        }

        static float[] LoadTensorFromEmbeddedResource(string path)
        {
            var tensorData = new List<float>();
            var assembly = typeof(InferenceSampleApi).Assembly;

            using (StreamReader inputFile =
                       new StreamReader(assembly.GetManifestResourceStream($"{assembly.GetName().Name}.{path}")))
            {
                inputFile.ReadLine(); // skip the input name
                string[] dataStr =
                    inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        static byte[] LoadModelFromEmbeddedResource(string path)
        {
            var assembly = typeof(InferenceSampleApi).Assembly;
            byte[] model = null;

            using (Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.{path}"))
            {
                using (MemoryStream memoryStream = new MemoryStream())
                {
                    stream.CopyTo(memoryStream);
                    model = memoryStream.ToArray();
                }
            }

            return model;
        }
    }
}
