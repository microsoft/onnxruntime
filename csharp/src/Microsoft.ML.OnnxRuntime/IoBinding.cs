using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This class enable to bind inputs and outputs to pre-allocated
    /// memory. This enables interesting scenarios. For example, if your input
    /// already resides in some pre-allocated memory even if on a device you bind
    /// that piece of memory to an input name and shape and onnxruntime will use that as input.
    /// Other traditional inputs can also be bound that already exists as Tensors
    /// </summary>
    public class IoBinding : IDisposable
    {
        private IntPtr _handle;

        internal IoBinding(InferenceSession session)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateIoBinding(session.Handle, out _handle));
        }

        internal IntPtr Handle
        {
            get
            {
                return _handle;
            }
        }

        public void BindInput(string name, Tensors.TensorElementType elementType, long[] shape, MemoryAllocation allocation)
        {
            using (var ortValue = OrtValue.CreateTensorValueWithData(allocation.Info,
                                                                    elementType,
                                                                    shape,
                                                                    allocation.Pointer, allocation.Size))
                BindIntputOrOutput(name, ortValue.Handle, true);
        }

        public void BindInput(string name, FixedBufferOnnxValue fixedValue)
        {
            if(fixedValue.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Binding works only with Tensors");
            }
            BindIntputOrOutput(name, fixedValue.Value, true);
        }

        public void BindOutput(string name, Tensors.TensorElementType elementType, long[] shape, MemoryAllocation allocation)
        {
            using (var ortValue = OrtValue.CreateTensorValueWithData(allocation.Info,
                                                                    elementType,
                                                                    shape,
                                                                    allocation.Pointer, allocation.Size))
                BindIntputOrOutput(name, ortValue.Handle, false);
        }

        public void BindOutput(string name, FixedBufferOnnxValue fixedValue)
        {
            if (fixedValue.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Binding works only with Tensors");
            }
            BindIntputOrOutput(name, fixedValue.Value, false);
        }

        private void BindIntputOrOutput(string name, IntPtr ortValue, bool isInput)
        {
            var utf8_str_pinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name), GCHandleType.Pinned);
            using (var pinnedName = new PinnedGCHandle(utf8_str_pinned))
            {
                if (isInput)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtBindInput(_handle, pinnedName.Pointer, ortValue));
                }
                else
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtBindOutput(_handle, pinnedName.Pointer, ortValue));
                }
            }
        }

        public void ClearBoundInputs()
        {
            NativeMethods.OrtClearBoundInputs(_handle);
        }

        public void ClearBoundOutputs()
        {
            NativeMethods.OrtClearBoundOutputs(_handle);
        }

        #region Disposable Support
        protected virtual void Dispose(bool disposing)
        {
            if(disposing)
            {
                NativeMethods.OrtReleaseIoBinding(_handle);
                _handle = IntPtr.Zero;
            }
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // No need for the finalizer
        #endregion
    }
}
