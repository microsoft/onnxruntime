// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Text;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Various providers of ONNX operators
    /// </summary>
    public enum ExecutionProvider
    {
        Cpu,
        MklDnn,
        Cuda
        //TODO: add more providers gradually
    };

    /// <summary>
    /// Holds the options for creating an InferenceSession
    /// </summary>
    public class SessionOptions:IDisposable
    {
        public IntPtr _nativePtr;
        protected static readonly Lazy<SessionOptions> _default = new Lazy<SessionOptions>(MakeSessionOptionWithCpuProvider);
        private static string[] cudaDelayLoadedLibs = { "cublas64_91.dll", "cudnn64_7.dll" };

        /// <summary>
        /// Constructs an empty SessionOptions
        /// </summary>
        public SessionOptions()
        {
            _nativePtr = NativeMethods.OrtCreateSessionOptions();
        }

        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to 1.        
        /// </summary>
        /// <param name="optimization_level">optimization level for the session</param>
        /// Available options are : 0, 1, 2
        /// 0 -> Disable all optimizations
        /// 1 -> Enable basic optimizations
        /// 2 -> Enable all optimizations
        /// <returns>True on success and false otherwise</returns>
        public bool SetSessionGraphOptimizationLevel(uint optimization_level)
        {
            var result = NativeMethods.OrtSetSessionGraphOptimizationLevel(_nativePtr, optimization_level);
            return result == 0;
        }

        /// <summary>
        /// Default instance
        /// </summary>
        public static SessionOptions Default
        {
            get
            {
                return _default.Value;
            }
        }

        private static SessionOptions MakeSessionOptionWithCpuProvider()
        {
            CheckLibcVersionGreaterThanMinimum();
            SessionOptions options = new SessionOptions();
            NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options._nativePtr, 1);
            return options;
        }

        /// <summary>
        /// A helper method to constuct a SessionOptions object for CUDA execution
        /// </summary>
        /// <returns>A SessionsOptions() object configured for execution on deviceId=0</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider()
        {
            return MakeSessionOptionWithCudaProvider(0);
        }

        /// <summary>
        /// A helper method to constuct a SessionOptions object for CUDA execution
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider(int deviceId=0)
        {
            CheckLibcVersionGreaterThanMinimum();
            CheckCudaExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options._nativePtr, deviceId);
            NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options._nativePtr, 1);
            return options;
        }

        // Declared, but called only if OS = Windows.
        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport("kernel32.dll")]
        static extern uint GetSystemDirectory([Out] StringBuilder lpBuffer, uint uSize);
        private static bool CheckCudaExecutionProviderDLLs()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var dll in cudaDelayLoadedLibs)
                {
                    IntPtr handle = LoadLibrary(dll);
                    if (handle != IntPtr.Zero)
                        continue;                    
                    var sysdir = new StringBuilder(String.Empty, 2048);
                    GetSystemDirectory(sysdir, (uint)sysdir.Capacity);
                    throw new OnnxRuntimeException(
                        ErrorCode.NoSuchFile, 
                        $"kernel32.LoadLibrary():'{dll}' not found. CUDA is required for GPU execution. " +
                        $". Verify it is available in the system directory={sysdir}. Else copy it to the output folder."
                        );               
                }
            }   
            return true;
        }

        [DllImport("libc", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr gnu_get_libc_version();

        private static void CheckLibcVersionGreaterThanMinimum()
        {
            // require libc version 2.23 or higher
            var minVersion = new Version(2, 23);
            var curVersion = new Version(0, 0);
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                try
                {
                    curVersion = Version.Parse(Marshal.PtrToStringAnsi(gnu_get_libc_version()));
                    if (curVersion >= minVersion)
                        return;
                }
                catch (Exception)
                {
                    // trap any obscure exception
                }
                throw new OnnxRuntimeException(ErrorCode.RuntimeException,
                        $"libc.so version={curVersion} does not meet the minimun of 2.23 required by OnnxRuntime. " +
                        "Linux distribution should be similar to Ubuntu 16.04 or higher");
            }
        }

        #region destructors disposers

        ~SessionOptions()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // cleanup managed resources
            }
            NativeMethods.OrtReleaseSessionOptions(_nativePtr);
        }

        #endregion
    }
}
