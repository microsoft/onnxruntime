using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    /// <summary>
    /// The OrtValueCollection holds ort values returned by the error function, but does not actually own any of them
    /// and therefore does not release them.
    /// </summary>
    internal class OrtValueCollection : SafeHandle
    {
        public OrtValueCollection(IntPtr h)
            : base(IntPtr.Zero, true)
        {
            handle = h;
        }

        internal IntPtr Handle
        {
            get { return handle; }
        }

        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        protected override bool ReleaseHandle()
        {
            return true;
        }

        #region Public Methods

        public int Count
        {
            get
            {
                UIntPtr val = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetCount(handle, out val));
                return (int)val;
            }
        }

        public int Capacity
        {
            get
            {
                UIntPtr val = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetCapacity(handle, out val));
                return (int)val;
            }
        }

        public OrtValue GetAt(int nIdx, out string strName)
        {
            IntPtr valData;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr valName;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetAt(handle, nIdx, out valData, allocator.Pointer, out valName));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, valName, 0))
            {
                strName = NativeOnnxValueHelper.StringFromNativeUtf8(valName);
            }

            return new OrtValue(valData, false);
        }

        public void SetAt(int nIdx, OrtValue val, string strName = "")
        {
            byte[] rgName = (string.IsNullOrEmpty(strName)) ? null : NativeMethods.GetPlatformSerializedString(strName);
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetAt(handle, nIdx, val.Handle, rgName));
        }

        #endregion
    }
}
