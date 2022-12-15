using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    public class ThreadingOptions : SafeHandle
    {
        /// <summary>
        /// A pointer to a underlying native instance of ThreadingOptions
        /// </summary>
        internal IntPtr Handle => handle;
        
        /// <summary>
        /// Create an empty threading options.
        /// </summary>
        public ThreadingOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateThreadingOptions(out handle));
        }

        /// <summary>
        /// Set global inter-op thread count.
        /// Setting it to 0 will allow ORT to choose the number of threads, setting it to 1 will cause the main thread to be used (i.e., no thread pools will be used).
        /// </summary>
        /// <param name="value">The number of threads.</param>
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
        
        /// <summary>
        /// Sets the number of threads available for intra-op parallelism (i.e. within a single op).
        /// Setting it to 0 will allow ORT to choose the number of threads, setting it to 1 will cause the main thread to be used (i.e., no thread pools will be used).
        /// </summary>
        /// <param name="value">The number of threads.</param>
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
        
        /// <summary>
        /// Allows spinning of thread pools when their queues are empty. This call sets the value for both inter-op and intra-op thread pools.
        /// If the CPU usage is very high then do not enable this.
        /// </summary>
        /// <param name="value">If true allow the thread pools to spin.</param>
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

        /// <summary>
        /// When this is set it causes intra-op and inter-op thread pools to flush denormal values to zero.
        /// </summary>
        public void SetGlobalDenormalAsZero()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtThreadingOptionsSetGlobalDenormalAsZero(handle));
        }
    
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of ThreadingOptions
        /// </summary>
        /// <returns>always returns true</returns>
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
