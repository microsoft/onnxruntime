using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents an Onnx Value with its underlying buffer pinned
    /// </summary>
    public class FixedBufferOnnxValue : IDisposable
    {
        internal MemoryHandle PinnedMemory { get; private set; }
        internal OrtValue Value { get; private set; }
        internal OnnxValueType OnnxValueType { get; private set; }
        internal TensorElementType ElementType { get; private set; }

        private FixedBufferOnnxValue(MemoryHandle pinnedMemory, OrtValue ortValue, OnnxValueType onnxValueType, TensorElementType elementType)
        {
            PinnedMemory = pinnedMemory;
            Value = ortValue;
            OnnxValueType = onnxValueType;
            ElementType = elementType;
        }

        /// <summary>
        /// Creates a <see cref="FixedBufferOnnxValue"/> object from the tensor and pins its underlying buffer.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <returns></returns>
        public static FixedBufferOnnxValue CreateFromTensor<T>(Tensor<T> value)
        {
            MemoryHandle? memHandle;
            var ortValue = OrtValue.CreateFromTensorObject(value, out memHandle, out TensorElementType elementType);
            if (memHandle.HasValue)
            {
                return new FixedBufferOnnxValue((MemoryHandle)memHandle, ortValue, OnnxValueType.ONNX_TYPE_TENSOR, elementType);
            }
            else
            {
                return new FixedBufferOnnxValue(default(MemoryHandle), ortValue, OnnxValueType.ONNX_TYPE_TENSOR, elementType);
            }
        }

        #region IDisposable Support

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                Value.Dispose();
                PinnedMemory.Dispose();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
