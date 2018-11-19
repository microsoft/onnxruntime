// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;

namespace CSharpUsage
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Using API");
            UseApi();
            Console.WriteLine("Done");
        }


        static void UseApi()
        {
            string modelPath = Directory.GetCurrentDirectory() + @"\testdata\squeezenet.onnx";


            using (var session = new InferenceSession(modelPath))
            {
                var inputMeta = session.InputMetadata;

                // User should be able to detect input name/type/shape from the metadata.
                // Currently InputMetadata implementation is inclomplete, so assuming Tensor<flot> of predefined dimension.

                var shape0 = new int[] { 1, 3, 224, 224 };
                float[] inputData0 = LoadInputsFloat();
                var tensor = new DenseTensor<float>(inputData0, shape0);

                var container = new List<NamedOnnxValue>();
                container.Add(new NamedOnnxValue("data_0", tensor));

                // Run the inference
                var results = session.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container

                // dump the results
                foreach (var r in results)
                {
                    Console.WriteLine("Output for {0}", r.Name);
                    Console.WriteLine(r.AsTensor<float>().GetArrayString());
                }

                // Just try some GC collect
                results = null;
                container = null;

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        static int[] LoadInputsInt32()
        {
            return null;
        }

        static float[] LoadInputsFloat()
        {
            // input: data_0 = float32[1,3,224,224] for squeezenet model
            // output: softmaxout_1 =  float32[1,1000,1,1]
            uint size = 1 * 3 * 224 * 224;
            float[] tensor = new float[size];

            // read data from file
            using (var inputFile = new System.IO.StreamReader(@"testdata\bench.in"))
            {
                inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensor[i] = Single.Parse(dataStr[i]);
                }
            }

           return tensor;
        }

    }
}
