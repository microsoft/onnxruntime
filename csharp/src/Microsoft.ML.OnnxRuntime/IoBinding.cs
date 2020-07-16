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

        public void BindInput(string name, MemoryAllocation allocation)
        {
        }

        public void BindInputs(ReadOnlySequence<string> names, ReadOnlySequence<MemoryAllocation> allocations)
        {
            if(names.IsEmpty || names.Length != allocations.Length)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Names and Allocations must be of equal length");
            }

            using (var ortValues = new DisposableList<OrtValue>())
            using (var pinnedNames = new DisposableList<MemoryHandle>())
            {
                for(int i = 0; i < names.Length; ++i)
                {
                    var first = names.Start;
                    ReadOnlyMemory<string> name;
                    bool next = true;
                    //while(next)
                    //{
                    //    names.TryGet(ref first, out name, true);
                    //MemoryHandle pinned_utf8 = new Memory<byte>(Encoding.UTF8.GetBytes(string.Concat(name, '\0'))).Pin();
                    //pinnedNames.Add(pinned_utf8);
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
