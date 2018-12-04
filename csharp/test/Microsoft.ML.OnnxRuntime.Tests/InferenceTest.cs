// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics.Tensors;
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
                float errorMargin = 1e-6F;
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

                    for (int i = 0; i < expectedOutput.Length; i++)
                    {
                        Assert.InRange<float>(resultArray[i], expectedOutput[i] - errorMargin, expectedOutput[i] + errorMargin);
                    }
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
            Assert.Equal("[ErrorCode:InvalidArgument] Invalid Feed Input Names: wrong_name Valid input names are: data_0 ", ex.Message);
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
            Assert.Equal("[ErrorCode:InvalidArgument] The number of feeds is not same as the number of the model input, expect 1 got 2", ex.Message);
            session.Dispose();
        }

        [Fact] 
        private void TestModelInputFloat()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_FLOAT.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var nov = NamedOnnxValue.CreateFromTensor("input", 
                new DenseTensor<float>(new float[] {1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, 
                new int[] { 1, 5 }) );
            container.Add(nov);
            var res = session.Run(container);  
        }

        [Fact(Skip = "Boolean tensor not supported yet")]
        private void TestModelInputBOOL()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_BOOL.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<bool>(new bool[] { true, false, true, false, true }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<bool>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }
        [Fact]

        private void TestModelInputINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT32.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<int>(new int[] { 1, -2, -3, int.MinValue, int.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<int>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact]
        private void TestModelInputDOUBLE()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_DOUBLE.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<double>(new double[] { 1.0, 2.0, -3.0, 5, 5},new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                //new DenseTensor<double>(new double[] { 1.0, 2.0, -3.0, double.MinValue, double.MaxValue},
                //new int[] { 1, 5 }));
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<double>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
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
        }

        [Fact(Skip = "Int8 not supported yet")]
        private void TestModelInputINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT8.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<sbyte>(new sbyte[] { 1, 2, -3, sbyte.MinValue, sbyte.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<sbyte>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact]
        private void TestModelInputUINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT8.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<byte>(new byte[] { 1, 2, 3, byte.MinValue, byte.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<byte>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact]
        private void TestModelInputUINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT16.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<UInt16>(new UInt16[] { 1, 2, 3, UInt16.MinValue, UInt16.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<UInt16>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact]
        private void TestModelInputINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT16.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<Int16>(new Int16[] { 1, 2, 3, Int16.MinValue, Int16.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<Int16>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact]
        private void TestModelInputINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_INT64.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<Int64>(new Int64[] { 1, 2, -3, Int64.MinValue, Int64.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<Int64>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact]
        private void TestModelInputUINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT32.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<UInt32>(new UInt32[] { 1, 2, 3, UInt32.MinValue, UInt32.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<UInt32>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }
        [Fact]
        private void TestModelInputUINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_UINT64.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<UInt64>(new UInt64[] { 1, 2, 3, UInt64.MinValue, UInt64.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<UInt64>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact(Skip = "Boolean FLOAT16 not available in C#")]
        private void TestModelInputFLOAT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Directory.GetCurrentDirectory() + @"\test_types_FLOAT16.onnx";
            var session = new InferenceSession(modelPath);
            var container = new List<NamedOnnxValue>();
            var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
            var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
            container.Add(nov);
            var res = session.Run(container);
            var tensorOut = res.First().AsTensor<float>();
            Assert.True(tensorOut.SequenceEqual(tensorIn));
        }

        [Fact]
        private void Yunsong()
        {
            var session = new InferenceSession(@"model_181031_12.onnx");

            float[] zerof = new float[] { 0 };
            long[] zerol = new long[] { 1 };
            var data = new List<NamedOnnxValue>() {
                 NamedOnnxValue.CreateFromTensor<float>("input_0_0", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_0_1", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_1_0", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_1_1", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_1_2", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_1_3", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_1_4", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_2_0", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_2_1", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_2_2", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_2_3", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_2_4", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_2_5", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_3_0", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_3_1", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_0", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_1", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_2", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_3", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_4", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_5", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_6", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_7", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_8", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_9", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_10", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_11", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_12", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_13", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_14", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_15", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_16", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_17", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_18", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_19", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_20", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_21", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_22", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_23", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_24", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_25", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_26", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_27", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_28", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_29", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_30", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_31", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_32", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_33", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_34", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_35", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_36", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_37", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_38", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_39", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_40", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_41", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_42", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_43", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_44", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_45", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_46", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_47", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_48", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_49", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_50", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_51", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_52", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_53", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_54", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_55", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_56", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_57", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_58", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_59", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_60", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_61", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_62", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_63", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_64", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_65", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_66", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_67", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_68", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_69", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_70", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_71", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_72", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_73", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_74", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_75", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_76", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_77", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_78", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_79", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_80", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_81", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_82", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_83", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_84", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_85", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_86", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_87", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_88", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_89", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_90", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_91", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_92", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_93", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_94", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_95", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_96", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_97", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_98", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_99", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_100", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_101", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_102", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_103", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_104", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_105", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_106", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_107", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_108", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_109", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_110", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_111", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_112", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_113", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_114", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_115", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_116", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_117", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_118", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_119", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_120", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_121", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_122", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_123", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_124", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_125", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_126", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_127", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_128", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_129", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_130", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_131", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_132", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_133", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_134", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_135", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_136", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_137", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_138", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_139", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_140", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_141", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_142", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_143", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_144", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_145", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_146", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<float>("input_4_147", new DenseTensor<float>(zerof, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_0", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_1", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_2", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_3", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_4", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_5", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_6", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_7", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_8", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_9", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_10", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_11", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_12", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_13", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_14", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_15", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_16", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_17", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_18", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_19", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_20", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_21", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_22", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_23", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_24", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_25", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_26", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_27", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_28", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_29", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_30", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_31", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_32", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_33", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_34", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_35", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_36", new DenseTensor<long>(zerol, new int[] { 1 })),
                 NamedOnnxValue.CreateFromTensor<long>("input_5_37", new DenseTensor<long>(zerol, new int[] { 1 })),
                 };

            var result = session.Run(data); session.Run(data);
            Assert.NotNull(result);
            Assert.Equal(1, result.Count);
            var value = result.First<NamedOnnxValue>();
            Assert.Equal("label", value.Name);
            Assert.NotNull(value.AsTensor<long>());
            Assert.Equal(1, value.AsTensor<long>().Length);
        }


        static float[] LoadTensorFromFile(string filename)
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

        static Tuple<InferenceSession, float[], DenseTensor<float>> OpenSessionSqueezeNet()
        {
            string modelPath = Directory.GetCurrentDirectory() + @"\squeezenet.onnx";
            var session = new InferenceSession(modelPath);
            float[] inputData = LoadTensorFromFile(@"bench.in");
            var inputMeta = session.InputMetadata;
            var tensor = new DenseTensor<float>(inputData, inputMeta["data_0"].Dimensions);
            return new Tuple<InferenceSession, float[], DenseTensor<float>>(session, inputData, tensor);
        }

    }
}