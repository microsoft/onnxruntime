// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using CommandLine;

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

    class CommandOptions
    {
        [Option('m', "model_file", Required = true, HelpText = "Model Path.")]
        public string ModelFile { get; set; }

        [Option('i', "input_file", Required = true, HelpText = "Input path.")]
        public string InputFile { get; set; }

        [Option('c', "iteration_count", Required = true, HelpText = "Iteration to run.")]
        public int IterationCount { get; set; }

        [Option('p', Required = false, HelpText = "Run with parallel exection. Default is false")]
        public bool ParallelExecution { get; set; } = false;

        [Option('o', "optimization_level", Required = false, HelpText = "Optimization Level. Default is 99, all optimization.")]
        public GraphOptimizationLevel OptimizationLevel { get; set; } = GraphOptimizationLevel.ORT_ENABLE_ALL;
    }

    class Program
    {
        public static void Main(string[] args)
        {
            var cmdOptions = Parser.Default.ParseArguments<CommandOptions>(args);
            cmdOptions.WithParsed(
                options =>
                {
                    Run(options);
                });
        }
        public static void Run(CommandOptions options)
        {
            string modelPath = options.ModelFile;
            string inputPath = options.InputFile;
            int iteration = options.IterationCount;
            bool parallelExecution = options.ParallelExecution;
            GraphOptimizationLevel optLevel = options.OptimizationLevel;
            Console.WriteLine("Running model {0} in OnnxRuntime:", modelPath);
            Console.WriteLine("input:{0}", inputPath);
            Console.WriteLine("iteration count:{0}", iteration);
            Console.WriteLine("parallel execution:{0}", parallelExecution);
            Console.WriteLine("optimization level:{0}", optLevel);
            DateTime[] timestamps = new DateTime[(int)TimingPoint.TotalCount];

            RunModelOnnxRuntime(modelPath, inputPath, iteration, timestamps, parallelExecution, optLevel);
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

        static void RunModelOnnxRuntime(string modelPath, string inputPath, int iteration, DateTime[] timestamps, bool parallelExecution, GraphOptimizationLevel optLevel)
        {
            if (timestamps.Length != (int)TimingPoint.TotalCount)
            {
                throw new ArgumentException("Timestamps array must have " + (int)TimingPoint.TotalCount + " size");
            }

            timestamps[(int)TimingPoint.Start] = DateTime.Now;
            SessionOptions options = new SessionOptions();
            if (parallelExecution) options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            options.GraphOptimizationLevel = optLevel;
            using (var session = new InferenceSession(modelPath, options))
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
                for (int i = 0; i < iteration; i++)
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
                + "dotnet Microsoft.ML.OnnxRuntime.PerfTool <onnx-model-path> <input-file-path> <iteration-count>"
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
