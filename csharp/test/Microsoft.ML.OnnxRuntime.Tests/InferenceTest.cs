// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Microsoft.ML.OnnxRuntime;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    public class InferenceTest
    {
        [Fact]
        public void CanCreateAndDisposeSessionWithModelPath()
        {
            string modelPath = Directory.GetCurrentDirectory() + @"\squeezenet.onnx";
            using (var session = new InferenceSession(modelPath))
            {
                Assert.NotNull(session);
                Assert.NotNull(session.InputMetadata);
                Assert.Equal(1, session.InputMetadata.Count); // 1 input node
                Assert.True(session.InputMetadata.ContainsKey("data_0")); // input node name
                Assert.Equal(typeof(float), session.InputMetadata["data_0"].ElementType);
                Assert.True(session.InputMetadata["data_0"].IsTensor);
                var expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                Assert.Equal(expectedInputDimensions.Length, session.InputMetadata["data_0"].Dimensions.Length);
                for (int i = 0; i < expectedInputDimensions.Length; i++)
                {
                    Assert.Equal(expectedInputDimensions[i], session.InputMetadata["data_0"].Dimensions[i]);
                }

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output node
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output node name
                Assert.Equal(typeof(float), session.OutputMetadata["softmaxout_1"].ElementType);
                Assert.True(session.OutputMetadata["softmaxout_1"].IsTensor);
                var expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                Assert.Equal(expectedOutputDimensions.Length, session.OutputMetadata["softmaxout_1"].Dimensions.Length);
                for (int i = 0; i < expectedOutputDimensions.Length; i++)
                {
                    Assert.Equal(expectedOutputDimensions[i], session.OutputMetadata["softmaxout_1"].Dimensions[i]);
                }
            }
        }

        [Fact]
        private void CanRunInferenceOnAModel()
        {
            string modelPath = Directory.GetCurrentDirectory() + @"\squeezenet.onnx";

            using (var session = new InferenceSession(modelPath))
            {
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();

                float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model

                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run the inference
                var results = session.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container

                Assert.Equal(1, results.Count);

                float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
                // validate the results
                foreach (var r in results)
                {
                    Assert.Equal("softmaxout_1", r.Name);

                    var resultTensor = r.AsTensor<float>();
                    int[] expectedDimensions = { 1, 1000, 1, 1 };  // hardcoded for now for the test data
                    Assert.Equal(expectedDimensions.Length, resultTensor.Rank);

                    var resultDimensions = resultTensor.Dimensions;
                    for (int i = 0; i < expectedDimensions.Length; i++)
                    {
                        Assert.Equal(expectedDimensions[i], resultDimensions[i]);
                    }

                    var resultArray = r.AsTensor<float>().ToArray();
                    Assert.Equal(expectedOutput.Length, resultArray.Length);
                    Assert.Equal(expectedOutput, resultArray, new floatComparer());
                }
            }
        }

        [Fact]
        private void ThrowWrongInputName()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("wrong_name", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.Equal("[ErrorCode:InvalidArgument] Missing required inputs: data_0", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowWrongInputType()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            int[] inputDataInt = inputData.Select(x => (int)x).ToArray();
            var tensor = new DenseTensor<int>(inputDataInt, inputMeta["data_0"].Dimensions);
            container.Add(NamedOnnxValue.CreateFromTensor<int>("data_0", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.Equal("[ErrorCode:InvalidArgument] Unexpected input data type. Actual: (class onnxruntime::NonOnnxType<int>) , expected: (class onnxruntime::NonOnnxType<float>)", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowWrongDimensions()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            var inputData = new float[] { 0.1f, 0.2f, 0.3f };
            var tensor = new DenseTensor<float>(inputData, new int[] { 1, 3 });
            container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.Equal("[ErrorCode:Fail] X num_dims does not match W num_dims. X: {1,3} W: {64,3,3,3}", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowDuplicateInput()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            var nov = NamedOnnxValue.CreateFromTensor<float>("data_0", tensor);
            container.Add(nov);
            container.Add(nov);
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.Equal("[ErrorCode:InvalidArgument] duplicated input name", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowExtraInputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            var nov1 = NamedOnnxValue.CreateFromTensor<float>("data_0", tensor);
            var nov2 = NamedOnnxValue.CreateFromTensor<float>("extra", tensor);
            container.Add(nov1);
            container.Add(nov2);
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.StartsWith("[ErrorCode:InvalidArgument] Invalid Feed Input Names: extra. Valid input names are: ", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void TestMultiThreads()
        {
            var numThreads = 10;
            var loop = 10;
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var expectedOut = tuple.Item4;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
            var tasks = new Task[numThreads];
            for (int i = 0; i < numThreads; i++)
            {
                tasks[i] = Task.Factory.StartNew(() =>
                {
                    for (int j = 0; j < loop; j++)
                    {
                        var resnov = session.Run(container);
                        var res = resnov.ToArray()[0].AsTensor<float>().ToArray<float>();
                        Assert.Equal(res, expectedOut, new floatComparer());
                    }
                });
            };
            Task.WaitAll(tasks);
            session.Dispose();
        }

        [Fact]
        private void TestPreTrainedModelsOpset7And8()
        {
            var opsets = new[] { "opset7", "opset8" };
            foreach (var opset in opsets)
            {
                var modelRoot = new DirectoryInfo(opset);
                foreach (var model in modelRoot.EnumerateDirectories())
                {
                    // TODO: dims contains 'None'. Session throws error.
                    if (model.ToString() == "test_tiny_yolov2")
                        continue;
                    try
                    {
                        var session = new InferenceSession($"{opset}\\{model}\\model.onnx");
                        var inMeta = session.InputMetadata;
                        var innodepair = inMeta.First();
                        var innodename = innodepair.Key;
                        var innodedims = innodepair.Value.Dimensions;
                        var dataIn = LoadTensorFromFile($"{opset}\\{model}\\test_data_0.input.txt", false);
                        var dataOut = LoadTensorFromFile($"{opset}\\{model}\\test_data_0.output.txt", false);
                        var tensorIn = new DenseTensor<float>(dataIn, innodedims);
                        var nov = new List<NamedOnnxValue>();
                        nov.Add(NamedOnnxValue.CreateFromTensor<float>(innodename, tensorIn));
                        var resnov = session.Run(nov);
                        var res = resnov.ToArray()[0].AsTensor<float>().ToArray<float>();
                        Assert.Equal(res, dataOut, new floatComparer());
                        session.Dispose();
                    }
                    catch (Exception ex)
                    {
                        var msg = $"Opset {opset}: Model {model}: error = {ex.Message}";
                        throw new Exception(msg);
                    }
                } //model
            } //opset
        }

        [Fact]
        private void TestModelInputFloat()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_FLOAT.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<float>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact(Skip = "Boolean tensor not supported yet")]
        private void TestModelInputBOOL()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_BOOL.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<bool>(new bool[] { true, false, true, false, true }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<bool>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact]
        private void TestModelInputINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT32.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<int>(new int[] { 1, -2, -3, int.MinValue, int.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<int>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact]
        private void TestModelInputDOUBLE()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_DOUBLE.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<double>(new double[] { 1.0, 2.0, -3.0, 5, 5 }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<double>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact(Skip = "String tensor not supported yet")]
        private void TestModelInputSTRING()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_STRING.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<string>(new string[] { "a", "c", "d", "z", "f" }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<string>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact(Skip = "Int8 not supported yet")]
        private void TestModelInputINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT8.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<sbyte>(new sbyte[] { 1, 2, -3, sbyte.MinValue, sbyte.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<sbyte>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact]
        private void TestModelInputUINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT8.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<byte>(new byte[] { 1, 2, 3, byte.MinValue, byte.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<byte>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact]
        private void TestModelInputUINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT16.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<UInt16>(new UInt16[] { 1, 2, 3, UInt16.MinValue, UInt16.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<UInt16>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact]
        private void TestModelInputINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT16.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<Int16>(new Int16[] { 1, 2, 3, Int16.MinValue, Int16.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<Int16>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact]
        private void TestModelInputINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT64.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<Int64>(new Int64[] { 1, 2, -3, Int64.MinValue, Int64.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<Int64>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact]
        private void TestModelInputUINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT32.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<UInt32>(new UInt32[] { 1, 2, 3, UInt32.MinValue, UInt32.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<UInt32>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }
        [Fact]
        private void TestModelInputUINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT64.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<UInt64>(new UInt64[] { 1, 2, 3, UInt64.MinValue, UInt64.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<UInt64>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        [Fact(Skip = "Boolean FLOAT16 not available in C#")]
        private void TestModelInputFLOAT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_FLOAT16.pb";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<float>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
            session.Dispose();
        }

        static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        static Tuple<InferenceSession, float[], DenseTensor<float>, float[]> OpenSessionSqueezeNet()
        {
            string modelPath = Directory.GetCurrentDirectory() + @"\squeezenet.onnx";
            var session = new InferenceSession(modelPath);
            float[] inputData = LoadTensorFromFile(@"bench.in");
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
            var inputMeta = session.InputMetadata;
            var tensor = new DenseTensor<float>(inputData, inputMeta["data_0"].Dimensions);
            return new Tuple<InferenceSession, float[], DenseTensor<float>, float[]>(session, inputData, tensor, expectedOutput);
        }

        class floatComparer : IEqualityComparer<float>
        {
            private float tol = 1e-6f;
            private float divtol = 1e-3f;
            public bool Equals(float x, float y)
            {
                if (y == 0)
                    return (Math.Abs(x - y) < tol);
                return (Math.Abs(1 - x / y) < divtol);
            }
            public int GetHashCode(float x)
            {
                return 0;
            }
        }
    }
}