// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This is a legacy class that is kept for backward compatibility.
    /// Use OrtValue based API.
    /// 
    /// Represents an OrtValue with its underlying buffer pinned
    /// </summary>
    public class FixedBufferOnnxValue : IDisposable
    {
        private bool _disposed = false;
        internal OrtValue Value { get; private set; }
        internal OnnxValueType OnnxValueType { get; private set; }
        internal TensorElementType ElementType { get; private set; }

        private FixedBufferOnnxValue(ref OrtValue ortValue, OnnxValueType onnxValueType, TensorElementType elementType)
        {
            Value = ortValue;
            ortValue = null;
            OnnxValueType = onnxValueType;
            ElementType = elementType;
        }

        /// <summary>
        /// Creates a <see cref="FixedBufferOnnxValue"/> object from the tensor and pins its underlying buffer.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <returns>a disposable instance of FixedBufferOnnxValue</returns>
        public static FixedBufferOnnxValue CreateFromTensor<T>(Tensor<T> value)
        {
            var ortValue = OrtValue.CreateFromTensorObject(value, out TensorElementType elementType);
            return new FixedBufferOnnxValue(ref ortValue, OnnxValueType.ONNX_TYPE_TENSOR, elementType);
        }

        /// <summary>
        /// This is a factory method that creates a disposable instance of FixedBufferOnnxValue
        /// on top of a buffer. Internally, it will pin managed buffer and will create
        /// an OrtValue containing a tensor that will not own the memory.
        /// Such instance of FixedBufferOnnxValue can be used both as input and output in InferenceSession.Run()
        /// overload. As compared to CreateFromTensor(), this allows you to pass in buffers with custom data types
        /// that are blittable as defined in https://docs.microsoft.com/en-us/dotnet/framework/interop/blittable-and-non-blittable-types
        /// I.e. those that have the same binary representation as the original type. This includes all existing types
        /// but may also allow using custom types for Float16 and BFloat16 providing they have the same layout and size.
        /// The resulting instance must be disposed of to release pinned memory and deallocate native OrtValue
        /// See example below.
        /// </summary>
        /// <typeparam name="T">Blittable data type, compatible with supported types</typeparam>
        /// <param name="memoryInfo">memoryInfo. For managed buffers simply use OrtMemoryInfo.DefaultInstance</param>
        /// <param name="memory"></param>
        /// <param name="elementType">TensorElementType</param>
        /// <param name="shape">shape of the tensor to be created</param>
        /// <param name="bytesSize">size of the allocation in bytes</param>
        /// <returns>a disposable instance of FixedBufferOnnxValue</returns>
        /// <example>
        /// Here is an example of using a 3rd party library class for processing float16/bfloat16.
        /// Currently, to pass tensor data and create a tensor one must copy data to Float16/BFloat16 structures
        /// so DenseTensor can recognize it.
        /// 
        /// If you are using a library that has a class Half and it is blittable, that is its managed in memory representation
        /// matches native one and its size is 16-bits, you can use the following conceptual example
        /// to feed/fetch data for inference using Half array. This allows you to avoid copying data from your Half[] to Float16[]
        ///
        /// \code{.cs}
        /// unsafe { Debug.Assert(sizeof(ushort) == sizeof(Half)); }
        /// Half[] input = new Half[] { 5646, 12345 };
        /// var input_shape = new long[] {input.Length};
        /// Half[] output = new Half[40]; // Whatever the expected len/shape is must match
        /// var output_shape = new long[] {output.Length};
        /// 
        /// var memInfo = OrtMemoryInfo.DefaultInstance; // CPU
        ///
        /// using(var fixedBufferInput = FixedBufferOnnxvalue.CreateFromMemory{Half}(memInfo,
        ///                         input, TensorElementType.Float16, input_shape, input.Length * sizeof(ushort))
        /// using(var fixedBufferOutput = FixedBufferOnnxvalue.CreateFromMemory{Half}(memInfo,
        ///                               output, TensorElementType.Float16, output_shape, output.Length * sizeof(ushort))
        /// {
        ///    FixedBufferOnnxvalue[] inputValues = new FixedBufferOnnxvalue[]{fixedBufferInput};
        ///    FixedBufferOnnxvalue[] outputValues = new FixedBufferOnnxvalue[]{fixedBufferOutput};
        ///    session.Run(inputNames, inputValues, outputNames, outputValues);
        ///   // Output is now in output[]
        /// }
        /// \endcode
        /// </example>
        public static FixedBufferOnnxValue CreateFromMemory<T>(OrtMemoryInfo memoryInfo, Memory<T> memory,
            TensorElementType elementType, long[] shape, long bytesSize) where T : unmanaged
        {
            if(elementType == TensorElementType.String)
            {
                throw new ArgumentException("String data type is not supported");
            }

            var ortValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, memory, shape);
            try
            {
                return new FixedBufferOnnxValue(ref ortValue, OnnxValueType.ONNX_TYPE_TENSOR, elementType);
            }
            catch (Exception)
            {
                ortValue?.Dispose();
                throw;
            }
        }

        #region IDisposable Support

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose()</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                Value.Dispose();
            }
            _disposed = true;
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
