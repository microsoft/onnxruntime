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

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output node
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output node name

                //TODO: verify shape/type of the input/output nodes when API available
            }
        }

        [Fact]
        private void CanRunInferenceOnAModel()
        {
            string modelPath = Directory.GetCurrentDirectory() + @"\squeezenet.onnx";

            using (var session = new InferenceSession(modelPath))
            {
                var inputMeta = session.InputMetadata;

                // User should be able to detect input name/type/shape from the metadata.
                // Currently InputMetadata implementation is inclomplete, so assuming Tensor<flot> of predefined dimension.

                var shape0 = new int[] { 1, 3, 224, 224 };
                float[] inputData0 = LoadTensorFromFile(@"bench.in");
                var tensor = new DenseTensor<float>(inputData0, shape0);

                var container = new List<NamedOnnxValue>();
                container.Add(new NamedOnnxValue("data_0", tensor));

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