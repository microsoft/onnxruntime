using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{

    internal class NativeOnnxObjectHandle : SafeHandle
    {
        public NativeOnnxObjectHandle(IntPtr ptr)
            : base(IntPtr.Zero, true)
        {
            handle = ptr;
        }
        public override bool IsInvalid
        {
            get
            {
                return (handle == IntPtr.Zero);
            }
        }

        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseObject(handle);
            return true;
        }
    }
}
