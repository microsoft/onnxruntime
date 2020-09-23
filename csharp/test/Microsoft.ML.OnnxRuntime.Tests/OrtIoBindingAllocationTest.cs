using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;
using static Microsoft.ML.OnnxRuntime.Tests.InferenceTest;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    public class OrtIoBindingAllocationTest
    {
        /// <summary>
        /// This works only for allocations accessible from host memory
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="elements"></param>
        private static void PopulateNativeBufferFloat(OrtMemoryAllocation buffer, float[] elements)
        {
            if (buffer.Size < elements.Length * sizeof(float))
            {
                Assert.True(false);
            }

            unsafe
            {
                float* p = (float*)buffer.Pointer;
                for (int i = 0; i < elements.Length; ++i)
                {
                    *p++ = elements[i];
                }
            }
        }

        [Fact]
        public void TestIOBindingWithOrtAllocation()
        {
            var inputName = "data_0";
            var outputName = "softmaxout_1";
            var allocator = OrtAllocator.DefaultInstance;
            // From the model
            using (var dispList = new DisposableListTest<IDisposable>())
            {
                var tuple = OpenSessionSqueezeNet();
                var session = tuple.Item1;
                var inputData = tuple.Item2;
                var inputTensor = tuple.Item3;
                var outputData = tuple.Item4;
                dispList.Add(session);
                var inputMeta = session.InputMetadata;
                var outputMeta = session.OutputMetadata;
                var outputTensor = new DenseTensor<float>(outputData, outputMeta[outputName].Dimensions);

                var ioBinding = session.CreateIoBinding();
                dispList.Add(ioBinding);

                var ortAllocationInput = allocator.Allocate((uint)inputData.Length * sizeof(float));
                dispList.Add(ortAllocationInput);
                var inputShape = Array.ConvertAll<int, long>(inputMeta[inputName].Dimensions, d => d);
                PopulateNativeBufferFloat(ortAllocationInput, inputData);

                var ortAllocationOutput = allocator.Allocate((uint)outputData.Length * sizeof(float));
                dispList.Add(ortAllocationOutput);

                var outputShape = Array.ConvertAll<int, long>(outputMeta[outputName].Dimensions, i => i);

                // Test 1. Bind the output to OrtAllocated buffer
                using (FixedBufferOnnxValue fixedInputBuffer = FixedBufferOnnxValue.CreateFromTensor(inputTensor))
                {
                    ioBinding.BindInput(inputName, fixedInputBuffer);
                    ioBinding.BindOutput(outputName, Tensors.TensorElementType.Float, outputShape, ortAllocationOutput);
                    using (var outputs = session.RunWithBindingAndNames(new RunOptions(), ioBinding))
                    {
                        Assert.Equal(1, outputs.Count);
                        var output = outputs.ElementAt(0);
                        Assert.Equal(outputName, output.Name);
                        var tensor = output.AsTensor<float>();
                        Assert.True(tensor.IsFixedSize);
                        Assert.Equal(outputData, tensor.ToArray<float>(), new floatComparer());
                    }
                }

                // Test 2. Bind the input to memory allocation and output to a fixedBuffer
                {
                    ioBinding.BindInput(inputName, Tensors.TensorElementType.Float, inputShape, ortAllocationInput);
                    ioBinding.BindOutput(outputName, Tensors.TensorElementType.Float, outputShape, ortAllocationOutput);
                    using (var outputs = session.RunWithBindingAndNames(new RunOptions(), ioBinding))
                    {
                        Assert.Equal(1, outputs.Count);
                        var output = outputs.ElementAt(0);
                        Assert.Equal(outputName, output.Name);
                        var tensor = output.AsTensor<float>();
                        Assert.True(tensor.IsFixedSize);
                        Assert.Equal(outputData, tensor.ToArray<float>(), new floatComparer());
                    }
                }
            }
        }
    }
}
