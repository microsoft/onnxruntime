// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using System.Diagnostics;


namespace Microsoft.ML.OnnxRuntime.PerfTool
{
    public enum TimingPoint
    {
        Start = 0,
        ModelLoaded = 1,
        InputLoaded = 2,
        RunComplete = 3,
        TotalCount = 4
    }

    class Program
    {

        public static void Main(string[] args)
        {
            /*
             args[0] = model-file-name
             args[1] = input-file-name
             args[2] = iteration count
             */

            if (args.Length < 3)
            {
                PrintUsage();
                Environment.Exit(1);
            }

            string modelPath = args[0];
            string inputPath = args[1];
            int iteration = Int32.Parse(args[2]);
            Console.WriteLine("Running model {0} in OnnxRuntime with input {1} for {2} times", modelPath, inputPath, iteration);
            DateTime[] timestamps = new DateTime[(int)TimingPoint.TotalCount];

            RunModelOnnxRuntime(modelPath, inputPath, iteration, timestamps);
            PrintReport(timestamps, iteration);
            Console.WriteLine("Done");

            Console.WriteLine("Running model {0} in Sonoma with input {1} for {2} times", modelPath, inputPath, iteration);
            RunModelOnnxRuntime(modelPath, inputPath, iteration, timestamps);
            PrintReport(timestamps, iteration);
            Console.WriteLine("Done");

        }


        public static float[] LoadTensorFromFile(string filename)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        static void RunModelOnnxRuntime(string modelPath, string inputPath, int iteration, DateTime[] timestamps)
        {
            if (timestamps.Length != (int)TimingPoint.TotalCount)
            {
                throw new ArgumentException("Timestamps array must have "+(int)TimingPoint.TotalCount+" size");
            }

            timestamps[(int)TimingPoint.Start] = DateTime.Now;

            using (var session = new InferenceSession(modelPath))
            {
                timestamps[(int)TimingPoint.ModelLoaded] = DateTime.Now;
                var inputMeta = session.InputMetadata;

                var container = new List<NamedOnnxValue>();
                foreach (var name in inputMeta.Keys)
                {
                    float[] rawData = LoadTensorFromFile(inputPath);
                    var tensor = new DenseTensor<float>(rawData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                

                timestamps[(int)TimingPoint.InputLoaded] = DateTime.Now;

                // Run the inference
                for (int i=0; i < iteration; i++)
                {
                    var results = session.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container
                    Debug.Assert(results != null);
                    Debug.Assert(results.Count == 1);
                    //results = null;
                    //GC.Collect();
                    //GC.WaitForPendingFinalizers();
                }

                timestamps[(int)TimingPoint.RunComplete] = DateTime.Now;
            }

        }


        static void PrintUsage()
        {
            Console.WriteLine("Usage:\n"
                +"dotnet Microsoft.ML.OnnxRuntime.PerfTool <onnx-model-path> <input-file-path> <iteration-count>"
                );
        }

        static void PrintReport(DateTime[] timestamps, int iterations)
        {
            Console.WriteLine("Model Load Time = " + (timestamps[(int)TimingPoint.ModelLoaded] - timestamps[(int)TimingPoint.Start]).TotalMilliseconds);
            Console.WriteLine("Input Load Time = " + (timestamps[(int)TimingPoint.InputLoaded] - timestamps[(int)TimingPoint.ModelLoaded]).TotalMilliseconds);

            double totalRuntime = (timestamps[(int)TimingPoint.RunComplete] - timestamps[(int)TimingPoint.InputLoaded]).TotalMilliseconds;
            double perIterationTime = totalRuntime / iterations;

            Console.WriteLine("Total Run time for {0} iterations = {1}", iterations, totalRuntime);
            Console.WriteLine("Per iteration time = {0}", perIterationTime);
        }
    }
}
