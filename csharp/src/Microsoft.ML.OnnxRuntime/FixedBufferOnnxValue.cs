using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Diagnostics;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents an Onnx Value with its underlying buffer pinned
    /// </summary>
    public class FixedBufferOnnxValue : IDisposable
    {
        internal MemoryHandle PinnedMemory { get; private set; }
        internal IntPtr Value { get; private set; }
        internal OnnxValueType OnnxValueType { get; private set; }
        internal TensorElementType ElementType { get; private set; }

        private FixedBufferOnnxValue(MemoryHandle pinnedMemory, IntPtr onnxValue, OnnxValueType onnxValueType, TensorElementType elementType)
        {
            PinnedMemory = pinnedMemory;
            Value = onnxValue;
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
            NativeOnnxValueHelper.CreateNativeOnnxValue(value, out IntPtr onnxValue, out MemoryHandle pinnedMemoryHandle, out OnnxValueType onnxValueType, out TensorElementType elementType);

            Debug.Assert(onnxValueType == OnnxValueType.ONNX_TYPE_TENSOR, "the value should always be a tensor");

            return new FixedBufferOnnxValue(pinnedMemoryHandle, onnxValue, onnxValueType, elementType);
        }

        #region IDisposable Support

        // standard dispose pattern to deal with both managed and native resources

        private bool disposed = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    ((IDisposable)PinnedMemory).Dispose();
                }

                if (Value != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseValue(Value);
                    Value = IntPtr.Zero;
                }

                disposed = true;
            }
        }

        ~FixedBufferOnnxValue()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
