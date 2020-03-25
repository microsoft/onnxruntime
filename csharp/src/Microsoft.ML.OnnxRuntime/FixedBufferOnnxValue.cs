using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents an Onnx Value with its underlying buffer pinned
    /// </summary>
    public class FixedBufferOnnxValue : IDisposable
    {
        internal MemoryHandle PinnedMemory { get; private set; }
        internal IntPtr Value { get; private set; }

        internal FixedBufferOnnxValue(MemoryHandle pinnedMemory, IntPtr onnxValue)
        {
            PinnedMemory = pinnedMemory;
            Value = onnxValue;
        }

        /// <summary>
        /// Creates a <see cref="FixedBufferOnnxValue"/> object from the tensor and pins its underlying buffer.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <returns></returns>
        public static FixedBufferOnnxValue CreateFromTensor<T>(Tensor<T> value)
        {
            NativeOnnxValueHelper.CreateNativeOnnxValue(value, out IntPtr onnxValue, out MemoryHandle pinnedMemoryHandle);
            return new FixedBufferOnnxValue(pinnedMemoryHandle, onnxValue);
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
