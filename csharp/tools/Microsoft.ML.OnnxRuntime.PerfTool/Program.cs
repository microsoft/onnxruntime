// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using CommandLine;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime.PerfTool
{
    public enum TimingPoint
    {
        Start = 0,
        ModelLoaded = 1,
        InputLoaded = 2,
        WarmUp = 3,
        RunComplete = 4,
        TotalCount = 5
    }

    class CommandOptions
    {
        [Option('m', "model_file", Required = true, HelpText = "Model Path.")]
        public string ModelFile { get; set; }

        [Option('c', "iteration_count", Required = true, HelpText = "Iteration to run.")]
        public int IterationCount { get; set; }

        [Option('i', "input_file", Required = false, HelpText = "Input file.")]
        public string InputFile { get; set; }

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

        static void Run(CommandOptions options)
        {
            string modelPath = options.ModelFile;
            string inputPath = options.InputFile;
            int iteration = options.IterationCount;
            bool parallelExecution = options.ParallelExecution;
            GraphOptimizationLevel optLevel = options.OptimizationLevel;

            Console.WriteLine("Running model {0} in OnnxRuntime:", modelPath);
            Console.WriteLine("iteration count:{0}", iteration);
            Console.WriteLine("input:{0}", inputPath);
            Console.WriteLine("parallel execution:{0}", parallelExecution);
            Console.WriteLine("optimization level:{0}", optLevel);

            DateTime[] timestamps = new DateTime[(int)TimingPoint.TotalCount];
            double[] timecosts = new double[iteration];

            RunModelOnnxRuntime(modelPath, inputPath, iteration, timestamps, timecosts, parallelExecution, optLevel);

            PrintReport(timestamps, timecosts, iteration);
            Console.WriteLine("Done");
        }

        static void RunModelOnnxRuntime(string modelPath, string inputPath, int iteration, DateTime[] timestamps,
                                        double[] timecosts, bool parallelExecution, GraphOptimizationLevel optLevel)
        {
            if (timestamps.Length != (int)TimingPoint.TotalCount)
            {
                throw new ArgumentException("Timestamps array must have " + (int)TimingPoint.TotalCount + " size");
            }

            Random random = new Random();

            timestamps[(int)TimingPoint.Start] = DateTime.Now;
            SessionOptions options = new SessionOptions();
            if (parallelExecution) options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            options.GraphOptimizationLevel = optLevel;
            using (var session = new InferenceSession(modelPath, options))
            {
                timestamps[(int)TimingPoint.ModelLoaded] = DateTime.Now;

                var containers = LoadTestData(modelPath, inputPath, session.InputMetadata);
                timestamps[(int)TimingPoint.InputLoaded] = DateTime.Now;

                // Warm-up
                {
                    var container = containers[random.Next(0, containers.Count)];
                    session.Run(container);
                }
                timestamps[(int)TimingPoint.WarmUp] = DateTime.Now;

                // Run the inference
                for (int i = 0; i < iteration; i++)
                {
                    var next = random.Next(0, containers.Count);
                    var container = containers[next];
                    var startTime = DateTime.Now;

                    var results = session.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container

                    timecosts[i] = (DateTime.Now - startTime).TotalMilliseconds;

                    Debug.Assert(results != null);
                    Debug.Assert(results.Count == 1);
                }
                timestamps[(int)TimingPoint.RunComplete] = DateTime.Now;
            }

        }

        // If inputPath is give, create a tensor from text format of data.
        // Otherwise, create a tensor from proto files. Multiple input directories can be given at the same path as a model file.
        // Each input directory must have the same number of input as a model.
        // In example, if a model has 3 input data, a layout for a model file and two set of input data are as follows,
        // |-- model.onnx
        // |-- test_data_0
        // |     |-- input_0.pb
        // |     |-- input_1.pb
        // |     |-- input_3.pb
        // |-- test_data_1
        // |     |-- input_0.pb
        // |     |-- input_1.pb
        // |     |-- input_3.pb
        static List<List<NamedOnnxValue>> LoadTestData(string modelPath, string inputPath, IReadOnlyDictionary<string, NodeMetadata> inputMeta)
        {
            var containers = new List<List<NamedOnnxValue>>();

            // If inputPath is given, give priority to it
            if (!String.IsNullOrEmpty(inputPath) && File.Exists(inputPath))
            {
                var container = LoadTensorFromText(inputPath, inputMeta);
                containers.Add(container);
            }
            else
            {
                var dirs = from dir in Directory.EnumerateDirectories(Path.GetDirectoryName(modelPath)) select dir;
                foreach (var dir in dirs)
                {
                    var container = LoadTestDataFromProtobuf(dir, inputMeta);
                    containers.Add(container);
                }
            }

            return containers;
        }

        static List<NamedOnnxValue> LoadTensorFromText(string filename, IReadOnlyDictionary<string, NodeMetadata> inputMeta)
        {
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                var tensorData = new List<float>();

                // read data from file
                using (var inputFile = new System.IO.StreamReader(filename))
                {
                    inputFile.ReadLine();   //skip the input name
                    string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                    for (int i = 0; i < dataStr.Length; i++)
                    {
                        tensorData.Add(Single.Parse(dataStr[i]));
                    }
                }

                var tensor = new DenseTensor<float>(tensorData.ToArray(), inputMeta[name].Dimensions);
                container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
            }

            return container;
        }

        static List<NamedOnnxValue> LoadTestDataFromProtobuf(string testDataPath, IReadOnlyDictionary<string, NodeMetadata> inputMeta)
        {
            var container = new List<NamedOnnxValue>();

            var filenames = from filename in Directory.EnumerateFiles(testDataPath, "input_*.pb") select filename;
            foreach (var filename in filenames)
            {
                Onnx.TensorProto tensorProto = null;
                using (var inputFile = File.OpenRead(filename))
                {
                    tensorProto = Onnx.TensorProto.Parser.ParseFrom(inputFile);
                }

                var namedOnnxValue = CreateNamedOnnxValueFromTensorProto(tensorProto, inputMeta);
                container.Add(namedOnnxValue);
            }

            return container;
        }

        static NamedOnnxValue CreateNamedOnnxValueFromTensorProto(Onnx.TensorProto tensorProto, IReadOnlyDictionary<string, NodeMetadata> inputMeta)
        {
            Type tensorElemType = null;
            int elemWidth = 0;
            GetElementTypeAndWidth((TensorElementType)tensorProto.DataType, out tensorElemType, out elemWidth);
            var dims = tensorProto.Dims.ToList().ConvertAll(x => (int)x);

            NodeMetadata nodeMeta = null;
            if (!inputMeta.TryGetValue(tensorProto.Name, out nodeMeta) ||
                nodeMeta.ElementType != tensorElemType)
            {
                throw new Exception("No Matching Tensor found from serialized tensor");
            }

            if (nodeMeta.ElementType == typeof(float))
            {
                return CreateNamedOnnxValueFromRawData<float>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(float), dims);
            }
            else if (nodeMeta.ElementType == typeof(double))
            {
                return CreateNamedOnnxValueFromRawData<double>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(double), dims);
            }
            else if (nodeMeta.ElementType == typeof(int))
            {
                return CreateNamedOnnxValueFromRawData<int>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(int), dims);
            }
            else if (nodeMeta.ElementType == typeof(uint))
            {
                return CreateNamedOnnxValueFromRawData<uint>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(uint), dims);
            }
            else if (nodeMeta.ElementType == typeof(long))
            {
                return CreateNamedOnnxValueFromRawData<long>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(long), dims);
            }
            else if (nodeMeta.ElementType == typeof(ulong))
            {
                return CreateNamedOnnxValueFromRawData<ulong>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(ulong), dims);
            }
            else if (nodeMeta.ElementType == typeof(short))
            {
                return CreateNamedOnnxValueFromRawData<short>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(short), dims);
            }
            else if (nodeMeta.ElementType == typeof(ushort))
            {
                return CreateNamedOnnxValueFromRawData<ushort>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(ushort), dims);
            }
            else if (nodeMeta.ElementType == typeof(byte))
            {
                return CreateNamedOnnxValueFromRawData<byte>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(byte), dims);
            }
            else if (nodeMeta.ElementType == typeof(bool))
            {
                return CreateNamedOnnxValueFromRawData<bool>(tensorProto.Name, tensorProto.RawData.ToArray(), sizeof(bool), dims);
            }
            else
            {
                throw new Exception("Tensors of type " + nameof(nodeMeta.ElementType) + " not currently supported in this tool");
            }
        }

        static void GetElementTypeAndWidth(TensorElementType elemType, out Type type, out int width)
        {
            switch (elemType)
            {
                case TensorElementType.Float:
                    type = typeof(float);
                    width = sizeof(float);
                    break;
                case TensorElementType.Double:
                    type = typeof(double);
                    width = sizeof(double);
                    break;
                case TensorElementType.Int16:
                    type = typeof(short);
                    width = sizeof(short);
                    break;
                case TensorElementType.UInt16:
                    type = typeof(ushort);
                    width = sizeof(ushort);
                    break;
                case TensorElementType.Int32:
                    type = typeof(int);
                    width = sizeof(int);
                    break;
                case TensorElementType.UInt32:
                    type = typeof(uint);
                    width = sizeof(uint);
                    break;
                case TensorElementType.Int64:
                    type = typeof(long);
                    width = sizeof(long);
                    break;
                case TensorElementType.UInt64:
                    type = typeof(ulong);
                    width = sizeof(ulong);
                    break;
                case TensorElementType.UInt8:
                    type = typeof(byte);
                    width = sizeof(byte);
                    break;
                case TensorElementType.Int8:
                    type = typeof(sbyte);
                    width = sizeof(sbyte);
                    break;
                case TensorElementType.String:
                    type = typeof(byte);
                    width = sizeof(byte);
                    break;
                case TensorElementType.Bool:
                    type = typeof(bool);
                    width = sizeof(bool);
                    break;
                default:
                    type = null;
                    width = 0;
                    break;
            }
        }

        static NamedOnnxValue CreateNamedOnnxValueFromRawData<T>(string name, byte[] rawData, int elemWidth, List<int> dimensions)
        {
            T[] data = new T[rawData.Length / elemWidth];
            Buffer.BlockCopy(rawData, 0, data, 0, rawData.Length);
            var denseTensor = new DenseTensor<T>(data, dimensions.ToArray());
            return NamedOnnxValue.CreateFromTensor<T>(name, denseTensor);
        }

        static void PrintUsage()
        {
            Console.WriteLine("Usage:\n"
                + "dotnet Microsoft.ML.OnnxRuntime.PerfTool -m <onnx-model-path> -i <input-file-path> -c <iteration-count>"
                );
        }

        static void PrintReport(DateTime[] timestamps, double[] timecosts, int iterations)
        {
            Console.WriteLine("Model Load Time = " + (timestamps[(int)TimingPoint.ModelLoaded] - timestamps[(int)TimingPoint.Start]).TotalMilliseconds);
            Console.WriteLine("Input Load Time = " + (timestamps[(int)TimingPoint.InputLoaded] - timestamps[(int)TimingPoint.ModelLoaded]).TotalMilliseconds);
            Console.WriteLine("Warm-up Time = " + (timestamps[(int)TimingPoint.WarmUp] - timestamps[(int)TimingPoint.InputLoaded]).TotalMilliseconds);

            double totalRuntime = (timestamps[(int)TimingPoint.RunComplete] - timestamps[(int)TimingPoint.WarmUp]).TotalMilliseconds;
            double perIterationTime = totalRuntime / iterations;

            Console.WriteLine("Total Run time for {0} iterations = {1}", iterations, totalRuntime);
            Console.WriteLine("Per iteration time = {0}", perIterationTime);

            Array.Sort(timecosts);
            Console.WriteLine("Min Latency: {0}", timecosts[0]);
            Console.WriteLine("Max Latency: {0}", timecosts[timecosts.Length - 1]);
            Console.WriteLine("P50 Latency: {0}", timecosts[(int)(timecosts.Length * 0.5)]);
            Console.WriteLine("P90 Latency: {0}", timecosts[(int)(timecosts.Length * 0.9)]);
            Console.WriteLine("P95 Latency: {0}", timecosts[(int)(timecosts.Length * 0.95)]);
            Console.WriteLine("P99 Latency: {0}", timecosts[(int)(timecosts.Length * 0.99)]);
            Console.WriteLine("P999 Latency: {0}", timecosts[(int)(timecosts.Length * 0.999)]);
        }
    }
}
