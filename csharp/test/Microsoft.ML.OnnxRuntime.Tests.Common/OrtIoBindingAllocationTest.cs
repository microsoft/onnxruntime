// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using static Microsoft.ML.OnnxRuntime.Tests.InferenceTest;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    [Collection("OrtBinding Tests")]
    public class OrtIoBindingAllocationTests : IDisposable
    {
        private const string _inputName = "data_0";
        private const string _outputName = "softmaxout_1";
        private static readonly OrtAllocator _allocator = OrtAllocator.DefaultInstance;

        private readonly RunOptions _runOptions;
        private readonly InferenceSession _session;
        private readonly float[] _inputData;
        private readonly float[] _outputData;

        private readonly long[] _inputShape;
        private readonly long _inputSizeInBytes;
        private readonly long[] _outputShape;
        private readonly long _outputSizeInBytes;

        private readonly OrtSafeMemoryHandle _inputNativeAllocation;
        private readonly OrtSafeMemoryHandle _outputNativeAllocation;

        private readonly DisposableListTest<IDisposable> _dispList = new DisposableListTest<IDisposable>();

        private bool _disposed = false;

        public OrtIoBindingAllocationTests()
        {
            var tuple = OpenSessionSqueezeNet();
            _session = tuple.Item1;
            _dispList.Add(_session);
            _runOptions = new RunOptions();
            _dispList.Add(_runOptions);

            _inputData = tuple.Item2;
            _outputData = tuple.Item4;

            var inputMeta = _session.InputMetadata;
            var outputMeta = _session.OutputMetadata;

            _inputShape = Array.ConvertAll<int, long>(inputMeta[_inputName].Dimensions, Convert.ToInt64);
            _outputShape = Array.ConvertAll<int, long>(outputMeta[_outputName].Dimensions, Convert.ToInt64);

            var inputShapeSize = ShapeUtils.GetSizeForShape(_inputShape);
            Assert.Equal(inputShapeSize, _inputData.Length);

            var outputShapeSize = ShapeUtils.GetSizeForShape(_outputShape);
            Assert.Equal(outputShapeSize, _outputData.Length);

            _inputSizeInBytes = inputShapeSize * sizeof(float);
            IntPtr allocPtr = Marshal.AllocHGlobal((int)_inputSizeInBytes);
            _inputNativeAllocation = new OrtSafeMemoryHandle(allocPtr);
            _dispList.Add(_inputNativeAllocation);

            PopulateNativeBuffer<float>(allocPtr, _inputData);

            _outputSizeInBytes = outputShapeSize * sizeof(float);
            allocPtr = Marshal.AllocHGlobal((int)_outputSizeInBytes);
            _outputNativeAllocation = new OrtSafeMemoryHandle(allocPtr);
        }

        // Probably redundant as we have no native resources
        ~OrtIoBindingAllocationTests()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                _dispList.Dispose();
            }
            _disposed = true;
        }

        /// <summary>
        /// This works only for allocations accessible from host memory
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="elements"></param>
        private static void PopulateNativeBuffer<T>(OrtMemoryAllocation buffer, T[] elements)
        {
            PopulateNativeBuffer(buffer.Pointer, elements);
        }

        private static void PopulateNativeBuffer<T>(IntPtr buffer, T[] elements)
        {
            Span<T> bufferSpan;
            unsafe
            {
                bufferSpan = new Span<T>(buffer.ToPointer(), elements.Length);
            }
            elements.CopyTo(bufferSpan);
        }

        /// <summary>
        /// Checks that the contents of the native buffer matches the expected output.
        /// </summary>
        private void CheckOutput(IntPtr resultBuffer)
        {
            Span<byte> bufferSpan;
            unsafe
            {
                bufferSpan = new Span<byte>(resultBuffer.ToPointer(), (int)_outputSizeInBytes);
            }
            var outputSpan = MemoryMarshal.Cast<byte, float>(bufferSpan);
            Assert.Equal(_outputData, outputSpan.ToArray(), new FloatComparer());
        }

        private void ClearOutput()
        {
            Span<byte> bufferSpan;
            unsafe
            {
                bufferSpan = new Span<byte>(_outputNativeAllocation.Handle.ToPointer(), (int)_outputSizeInBytes);
            }
            bufferSpan.Clear();
        }

        /// <summary>
        /// Use to free globally allocated memory. Could not find
        /// a framework class.
        /// </summary>
        class OrtSafeMemoryHandle : SafeHandle
        {
            public OrtSafeMemoryHandle(IntPtr allocPtr) : base(allocPtr, true) { }

            public override bool IsInvalid => handle == IntPtr.Zero;

            public IntPtr Handle => handle;

            protected override bool ReleaseHandle()
            {
                Marshal.FreeHGlobal(handle);
                handle = IntPtr.Zero;
                return true;
            }
        }

        [Fact(DisplayName = "TestIOBindingWithOrtValues")]
        public void TestIOBindingWithOrtValues()
        {
            ClearOutput();

            using (var ioBinding = _session.CreateIoBinding())
            {
                // Input OrtValue on top if input buffer
                using (var tensor = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance,
                       TensorElementType.Float,
                       _inputShape, _inputNativeAllocation.Handle, _inputSizeInBytes))
                {
                    ioBinding.BindInput(_inputName, tensor);
                }

                // Output OrtValue on top if output buffer
                using (var tensor = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance,
                       TensorElementType.Float,
                       _outputShape, _outputNativeAllocation.Handle, _outputSizeInBytes))
                {
                    ioBinding.BindOutput(_outputName, tensor);
                }

                ioBinding.SynchronizeBoundInputs();

                using (var results = _session.RunWithBoundResults(_runOptions, ioBinding))
                {
                    ioBinding.SynchronizeBoundOutputs();
                    Assert.Single(results);
                    var res = results.First();
                    Assert.True(res.IsTensor);

                    var typeAndShape = res.GetTensorTypeAndShape();
                    Assert.Equal(_outputData.LongLength, typeAndShape.ElementCount);

                    var dataSpan = res.GetTensorDataAsSpan<float>();
                    Assert.Equal(_outputData, dataSpan.ToArray(), new FloatComparer());

                    // The result is good, but we want to make sure that the result actually is
                    // in the output memory, not some other place
                    CheckOutput(_outputNativeAllocation.Handle);
                }

                var outputNames = ioBinding.GetOutputNames();
                Assert.Single(outputNames);
                Assert.Equal(_outputName, outputNames[0]);
            }
        }

        [Fact(DisplayName = "TestIOBindingWithDeviceBoundOutput")]
        public void TestIOBindingWithDeviceBoundOutput()
        {
            ClearOutput();

            using (var ioBinding = _session.CreateIoBinding())
            {
                // Input OrtValue on top if input buffer
                using (var tensor = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance,
                       TensorElementType.Float,
                       _inputShape, _inputNativeAllocation.Handle, _inputSizeInBytes))
                {
                    ioBinding.BindInput(_inputName, tensor);
                }

                // The output will go into the Ort allocated OrtValue
                ioBinding.BindOutputToDevice(_outputName, OrtMemoryInfo.DefaultInstance);
                ioBinding.SynchronizeBoundInputs();

                using (var results = _session.RunWithBoundResults(_runOptions, ioBinding))
                {
                    ioBinding.SynchronizeBoundOutputs();
                    Assert.Single(results);
                    var res = results.First();
                    Assert.True(res.IsTensor);

                    var typeAndShape = res.GetTensorTypeAndShape();
                    Assert.Equal(_outputData.LongLength, typeAndShape.ElementCount);

                    var dataSpan = res.GetTensorDataAsSpan<float>();
                    Assert.Equal(_outputData, dataSpan.ToArray(), new FloatComparer());
                }
            }
        }

        [Fact(DisplayName = "TestIOBindingToOrtAllocatedBuffer")]
        public void TestIOBindingToOrtAllocatedBuffer()
        {
            var ortAllocationInput = _allocator.Allocate((uint)_inputSizeInBytes);
            _dispList.Add(ortAllocationInput);
            PopulateNativeBuffer<float>(ortAllocationInput, _inputData);

            var ortAllocationOutput = _allocator.Allocate((uint)_outputSizeInBytes);
            _dispList.Add(ortAllocationOutput);

            using (var ioBinding = _session.CreateIoBinding())
            {
                // Still supporting OrtAllocations overload
                ioBinding.BindInput(_inputName, Tensors.TensorElementType.Float, _inputShape, ortAllocationInput);
                ioBinding.BindOutput(_outputName, Tensors.TensorElementType.Float, _outputShape, ortAllocationOutput);
                ioBinding.SynchronizeBoundInputs();
                using (var outputs = _session.RunWithBoundResults(_runOptions, ioBinding))
                {
                    ioBinding.SynchronizeBoundOutputs();
                    Assert.Single(outputs);
                    var res = outputs.First();
                    Assert.True(res.IsTensor);

                    var typeAndShape = res.GetTensorTypeAndShape();
                    Assert.Equal(_outputData.LongLength, typeAndShape.ElementCount);

                    var dataSpan = res.GetTensorDataAsSpan<float>();
                    Assert.Equal(_outputData, dataSpan.ToArray(), new FloatComparer());
                    CheckOutput(ortAllocationOutput.Pointer);
                }
            }
        }
    }

    [Collection("OrtBinding Tests")]
    public class OrtBidingCircularTest
    {
        [Fact(DisplayName = "TestIOBinding Demonstrate Circular")]
        public void TestIOBindingDemonstrateCircular()
        {
            // This model has input and output of the same shape, so we can easily feed
            // output to input using binding, or not using one. The example makes use of
            // the binding to demonstrate the circular feeding.
            // With the OrtValue API exposed, one create OrtValues over arbitrary buffers and feed them to the model using
            // OrtValues based Run APIs. Thus, the Binding is not necessary any longer
            //
            // However, here is the demonstration by popular request.
            var model = TestDataLoader.LoadModelFromEmbeddedResource("mul_1.onnx");

            const string inputName = "X";
            const string outputName = "Y";

            long[] inputOutputShape = { 3, 2 };
            float[] input = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
            var inputOutputShapeSize = ShapeUtils.GetSizeForShape(inputOutputShape);
            Assert.Equal(inputOutputShapeSize, input.LongLength);

            var memInput = new Memory<float>(input);
            IntPtr inputPtr;

            // Output data on the first iteration
            float[] firstIterExpectedOutput = { 1.0F, 4.0F, 9.0F, 16.0F, 25.0F, 36.0F };
            Assert.Equal(inputOutputShapeSize, firstIterExpectedOutput.LongLength);

            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var runOptions = new RunOptions();
                cleanUp.Add(runOptions);

                var session = new InferenceSession(model);
                cleanUp.Add(session);

                var ioBinding = session.CreateIoBinding();
                cleanUp.Add(ioBinding);



                var pinInput = memInput.Pin();
                cleanUp.Add(pinInput);

                // This can be a ptr to arbitrary buffer, not necessarily a pinned one
                unsafe
                {
                    inputPtr = (IntPtr)pinInput.Pointer;
                }

                // Bind the input
                using (var ortInput = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance,
                                       TensorElementType.Float, inputOutputShape, inputPtr, input.Length * sizeof(float)))
                {
                    ioBinding.BindInput(inputName, ortInput);
                }

                // We could have bound the output as well, but we simply bind it to a device in this case.
                // Just check the result the first time around.
                ioBinding.BindOutputToDevice(outputName, OrtMemoryInfo.DefaultInstance);

                ioBinding.SynchronizeBoundInputs();

                // We dispose the output after we rebind it to the input because it will be copied during binding.
                using (var results = session.RunWithBoundResults(runOptions, ioBinding))
                {
                    ioBinding.SynchronizeBoundOutputs();
                    Assert.Single(results); // One output

                    var res = results.First();
                    Assert.True(res.IsTensor);

                    var typeShape = res.GetTensorTypeAndShape();
                    Assert.Equal(TensorElementType.Float, typeShape.ElementDataType);
                    Assert.Equal(inputOutputShape, typeShape.Shape);
                    Assert.Equal(inputOutputShapeSize, typeShape.ElementCount);

                    // First time around the output should match the expected
                    Assert.Equal(firstIterExpectedOutput, res.GetTensorDataAsSpan<float>().ToArray());

                    // Now we rebind the output to the input
                    // It is the same name, so the OrtValue would be replaced.
                    ioBinding.BindInput(inputName, res);
                }

                // Let's do it 2 more times.
                const int iterations = 2;
                for (int i = 0; i < iterations; ++i)
                {
                    using (var results = session.RunWithBoundResults(runOptions, ioBinding))
                    {
                        ioBinding.SynchronizeBoundOutputs();
                        Assert.Single(results); // One output
                        var res = results.First();
                        Assert.True(res.IsTensor);
                        var typeShape = res.GetTensorTypeAndShape();
                        Assert.Equal(TensorElementType.Float, typeShape.ElementDataType);
                        Assert.Equal(inputOutputShapeSize, typeShape.ElementCount);
                        Assert.Equal(inputOutputShape, typeShape.Shape);

                        ioBinding.BindInput(inputName, res);
                    }
                }
            }

        }
    }
}