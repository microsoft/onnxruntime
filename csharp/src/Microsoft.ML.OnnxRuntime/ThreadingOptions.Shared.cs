using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    public class ThreadingOptions : SafeHandle
    {
        internal IntPtr Handle => handle;
        
        
        public ThreadingOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateThreadingOptions(out handle));
        }

        public int GlobalInterOpNumThreads
        {
            get => _globalInterOpNumThreads;
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtThreadingOptionsSetGlobalInterOpNumThreads(handle, value));
                _globalInterOpNumThreads = value;
            }
        }
        private int _globalInterOpNumThreads;
        
        public int GlobalIntraOpNumThreads
        {
            get => _globalIntraOpNumThreads;
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtThreadingOptionsSetGlobalIntraOpNumThreads(handle, value));
                _globalIntraOpNumThreads = value;
            }
        }
        private int _globalIntraOpNumThreads;
        
        public bool GlobalSpinControl
        {
            get => _globalSpinControl;
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtThreadingOptionsSetGlobalSpinControl(handle, value ? 1 : 0));
                _globalSpinControl = value;
            }
        }
        private bool _globalSpinControl = true; // Default: spin is on

        public void SetGlobalDenormalAsZero()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtThreadingOptionsSetGlobalDenormalAsZero(handle));
        }
    
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseThreadingOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid => handle == IntPtr.Zero;
    }
}
