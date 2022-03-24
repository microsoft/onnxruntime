// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32.SafeHandles;
using System;
using System.Linq;
using System.Runtime.InteropServices;
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

            PopulateNativeBuffer(buffer.Pointer, elements);
        }

        private static void PopulateNativeBuffer(IntPtr buffer, float[] elements)
        {
            unsafe
            {
                float* p = (float*)buffer;
                for (int i = 0; i < elements.Length; ++i)
                {
                    *p++ = elements[i];
                }
            }
        }
        /// <summary>
        /// Use to free globally allocated memory
        /// </summary>
        class OrtSafeMemoryHandle : SafeHandle
        {
            public OrtSafeMemoryHandle(IntPtr allocPtr) : base(allocPtr, true) { }

            public override bool IsInvalid => handle == IntPtr.Zero;

            protected override bool ReleaseHandle()
            {
                Marshal.FreeHGlobal(handle);
                handle = IntPtr.Zero;
                return true;
            }
        }

        [Fact(DisplayName = "TestIOBindingWithOrtAllocation")]
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
                var runOptions = new RunOptions();
                dispList.Add(runOptions);

                var inputMeta = session.InputMetadata;
                var outputMeta = session.OutputMetadata;
                var outputTensor = new DenseTensor<float>(outputData, outputMeta[outputName].Dimensions);

                var ioBinding = session.CreateIoBinding();
                dispList.Add(ioBinding);

                var ortAllocationInput = allocator.Allocate((uint)inputData.Length * sizeof(float));
                dispList.Add(ortAllocationInput);
                var inputShape = Array.ConvertAll<int, long>(inputMeta[inputName].Dimensions, d => d);
                var shapeSize = ArrayUtilities.GetSizeForShape(inputShape);
                Assert.Equal(shapeSize, inputData.Length);
                PopulateNativeBufferFloat(ortAllocationInput, inputData);

                // Create an external allocation for testing OrtExternalAllocation
                var cpuMemInfo = OrtMemoryInfo.DefaultInstance;
                var sizeInBytes = shapeSize * sizeof(float);
                IntPtr allocPtr = Marshal.AllocHGlobal((int)sizeInBytes);
                dispList.Add(new OrtSafeMemoryHandle(allocPtr));
                PopulateNativeBuffer(allocPtr, inputData);

                var ortAllocationOutput = allocator.Allocate((uint)outputData.Length * sizeof(float));
                dispList.Add(ortAllocationOutput);

                var outputShape = Array.ConvertAll<int, long>(outputMeta[outputName].Dimensions, i => i);

                // Test 1. Bind the output to OrtAllocated buffer
                using (FixedBufferOnnxValue fixedInputBuffer = FixedBufferOnnxValue.CreateFromTensor(inputTensor))
                {
                    ioBinding.BindInput(inputName, fixedInputBuffer);
                    ioBinding.BindOutput(outputName, Tensors.TensorElementType.Float, outputShape, ortAllocationOutput);
                    ioBinding.SynchronizeBoundInputs();
                    using (var outputs = session.RunWithBindingAndNames(runOptions, ioBinding))
                    {
                        ioBinding.SynchronizeBoundOutputs();
                        Assert.Equal(1, outputs.Count);
                        var output = outputs.ElementAt(0);
                        Assert.Equal(outputName, output.Name);
                        var tensor = output.AsTensor<float>();
                        Assert.True(tensor.IsFixedSize);
                        Assert.Equal(outputData, tensor.ToArray<float>(), new FloatComparer());
                    }
                }

                // Test 2. Bind the input to memory allocation and output to a fixedBuffer
                {
                    ioBinding.BindInput(inputName, Tensors.TensorElementType.Float, inputShape, ortAllocationInput);
                    ioBinding.BindOutput(outputName, Tensors.TensorElementType.Float, outputShape, ortAllocationOutput);
                    ioBinding.SynchronizeBoundInputs();
                    using (var outputs = session.RunWithBindingAndNames(runOptions, ioBinding))
                    {
                        ioBinding.SynchronizeBoundOutputs();
                        Assert.Equal(1, outputs.Count);
                        var output = outputs.ElementAt(0);
                        Assert.Equal(outputName, output.Name);
                        var tensor = output.AsTensor<float>();
                        Assert.True(tensor.IsFixedSize);
                        Assert.Equal(outputData, tensor.ToArray<float>(), new FloatComparer());
                    }
                }
                // 3. Test external allocation
                {
                    var externalInputAllocation = new OrtExternalAllocation(cpuMemInfo, inputShape,
                        Tensors.TensorElementType.Float, allocPtr, sizeInBytes);

                    ioBinding.BindInput(inputName, externalInputAllocation);
                    ioBinding.BindOutput(outputName, Tensors.TensorElementType.Float, outputShape, ortAllocationOutput);
                    ioBinding.SynchronizeBoundInputs();
                    using (var outputs = session.RunWithBindingAndNames(runOptions, ioBinding))
                    {
                        ioBinding.SynchronizeBoundOutputs();
                        Assert.Equal(1, outputs.Count);
                        var output = outputs.ElementAt(0);
                        Assert.Equal(outputName, output.Name);
                        var tensor = output.AsTensor<float>();
                        Assert.True(tensor.IsFixedSize);
                        Assert.Equal(outputData, tensor.ToArray<float>(), new FloatComparer());
                    }
                }
                // 4. Some negative tests for external allocation
                {
                    // Small buffer size
                    Action smallBuffer = delegate ()
                        {
                            new OrtExternalAllocation(cpuMemInfo, inputShape,
                            Tensors.TensorElementType.Float, allocPtr, sizeInBytes - 10);
                        };

                    Assert.Throws<OnnxRuntimeException>(smallBuffer);

                    Action stringType = delegate ()
                    {
                        new OrtExternalAllocation(cpuMemInfo, inputShape,
                        Tensors.TensorElementType.String, allocPtr, sizeInBytes);
                    };

                    Assert.Throws<OnnxRuntimeException>(stringType);

                }

            }
        }
    }
}