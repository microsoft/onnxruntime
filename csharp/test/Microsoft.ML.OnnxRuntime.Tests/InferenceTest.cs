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
                Assert.Equal(typeof(float), session.InputMetadata["data_0"].Type);
                var expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                Assert.Equal(expectedInputDimensions.Length, session.InputMetadata["data_0"].Dimensions.Length);
                for (int i = 0; i < expectedInputDimensions.Length; i++)
                {
                    Assert.Equal(expectedInputDimensions[i], session.InputMetadata["data_0"].Dimensions[i]);
                }

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output node
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output node name
                Assert.Equal(typeof(float), session.OutputMetadata["softmaxout_1"].Type);
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
                    Assert.Equal(typeof(float), inputMeta[name].Type);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(new NamedOnnxValue(name, tensor));
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


    }
}