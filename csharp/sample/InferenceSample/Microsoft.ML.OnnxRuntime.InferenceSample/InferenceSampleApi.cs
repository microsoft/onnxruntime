using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Microsoft.ML.OnnxRuntime.InferenceSample
{
    public class InferenceSampleApi : IDisposable
    {
        public InferenceSampleApi()
        {
            model = LoadModelFromEmbeddedResource("TestData.squeezenet.onnx");

            // this is the data for only one input tensor for this model
            var inputTensor = LoadTensorFromEmbeddedResource("TestData.bench.in");

            // create default session with default session options
            // Creating an InferenceSession and loading the model is an expensive operation, so generally you would
            // do this once. InferenceSession.Run can be called multiple times, and concurrently.
            CreateInferenceSession();

            // setup sample input data
            inputData = new List<NamedOnnxValue>();
            var inputMeta = inferenceSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                // note: DenseTensor takes a copy of the provided data
                var tensor = new DenseTensor<float>(inputTensor, inputMeta[name].Dimensions);
                inputData.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
            }
        }

        public void CreateInferenceSession(SessionOptions options = null)
        {
            // Optional : Create session options and set any relevant values.
            // If an additional execution provider is needed it should be added to the SessionOptions prior to
            // creating the InferenceSession. The CPU Execution Provider is always added by default.
            if (options == null)
            {
                options = new SessionOptions { LogId = "Sample" };
            }

            inferenceSession = new InferenceSession(model, options);
        }

        public void Execute()
        {
            // Run the inference
            // 'results' is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
            using (var results = inferenceSession.Run(inputData))
            {
                // dump the results
                foreach (var r in results)
                {
                    Console.WriteLine("Output for {0}", r.Name);
                    Console.WriteLine(r.AsTensor<float>().GetArrayString());
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing && inferenceSession != null)
            {
                inferenceSession.Dispose();
                inferenceSession = null;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        static float[] LoadTensorFromEmbeddedResource(string path)
        {
            var tensorData = new List<float>();
            var assembly = typeof(InferenceSampleApi).Assembly;

            using (var inputFile = 
                new StreamReader(assembly.GetManifestResourceStream($"{assembly.GetName().Name}.{path}")))
            {
                inputFile.ReadLine(); // skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, 
                                                              StringSplitOptions.RemoveEmptyEntries);
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

        private readonly byte[] model;
        private readonly List<NamedOnnxValue> inputData;
        private InferenceSession inferenceSession;
    }
}
