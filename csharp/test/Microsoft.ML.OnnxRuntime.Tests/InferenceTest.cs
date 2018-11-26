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
    public class InfereceTest
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