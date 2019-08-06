// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Text;
using System.Runtime.InteropServices;
using System.IO;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Holds the options for creating an InferenceSession
    /// </summary>
    public class SessionOptions : IDisposable
    {
        private IntPtr _nativePtr;
//        protected static readonly Lazy<SessionOptions> _default = new Lazy<SessionOptions>(MakeSessionOptionWithCpuProvider);
        private static string[] cudaDelayLoadedLibs = { "cublas64_100.dll", "cudnn64_7.dll" };

        #region Constructor and Factory methods

        /// <summary>
        /// Constructs an empty SessionOptions
        /// </summary>
        public SessionOptions()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionOptions(out _nativePtr));
        }

        /// <summary>
        /// Default instance
        /// </summary>
        //public static SessionOptions Default
        //{
        //    get
        //    {
        //        return _default.Value;
        //    }
        //}

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
        public static SessionOptions MakeSessionOptionWithCudaProvider(int deviceId = 0)
        {
            CheckLibcVersionGreaterThanMinimum();
            CheckCudaExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(options._nativePtr, deviceId);
            NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(options._nativePtr, 1);
            return options;
        }

        #endregion

        #region Public Properties

        internal IntPtr Handle
        {
            get
            {
                return _nativePtr;
            }
        }

        /// <summary>
        /// Enable Sequential Execution. By default, it is enabled.
        /// </summary>
        /// </param>
        /// 
        private bool _enableSequentialExecution = true;
        public bool EnableSequentialExecution
        {
            get
            {
                return _enableSequentialExecution;
            }
            set
            {
                if (!_enableSequentialExecution && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableSequentialExecution(_nativePtr));
                    _enableSequentialExecution = true;
                }
                else if (_enableSequentialExecution && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableSequentialExecution(_nativePtr));
                    _enableSequentialExecution = false;
                }
            }
        }

        /// <summary>
        /// Enable Mem Pattern. By default, it is enabled
        /// </summary>
        /// </param>
        /// 
        private bool _enableMemoryPattern = true;
        public bool EnableMemoryPattern
        {
            get
            {
                return _enableMemoryPattern;
            }
            set
            {
                if (!_enableMemoryPattern && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableMemPattern(_nativePtr));
                    _enableMemoryPattern = true;
                }
                else if (_enableMemoryPattern && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableMemPattern(_nativePtr));
                    _enableMemoryPattern = false;
                }
            }
        }

        public string ProfileOutputPathPrefix
        {
            get; set;
        } = "onnxruntime_profile_";   // this is the same default in C++ implementation

        private bool _enableProfiling = false;
        public bool EnableProfiling
        {
            get
            {
                return _enableProfiling;
            }
            set
            {
                if (!_enableProfiling && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableProfiling(_nativePtr, ProfileOutputPathPrefix));
                    _enableProfiling = true;
                }
                else if (_enableProfiling && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableProfiling(_nativePtr));
                    _enableProfiling = false;
                }
            }
        }

        private bool _enableCpuMemArena = true;
        public bool EnableCpuMemArena
        {
            get
            {
                return _enableCpuMemArena;
            }
            set
            {
                if (!_enableCpuMemArena && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableCpuMemArena(_nativePtr));
                    _enableCpuMemArena = true;
                } 
                else if (_enableCpuMemArena && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableCpuMemArena(_nativePtr));
                    _enableCpuMemArena = false;
                }
            }
        }

        private string _logId = "";
        public string LogId
        {
            get
            {
                return _logId;
            }

            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogId(_nativePtr, value));
                _logId = value;
            }
        }

        private LogLevel _logVerbosityLevel = LogLevel.Verbose;
        public LogLevel LogVerbosityLevel
        {
            get
            {
                return _logVerbosityLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogVerbosityLevel(_nativePtr, value));
                _logVerbosityLevel = value;
            }
        }


        private int _threadPoolSize = 0; // set to what is set in C++ SessionOptions by default;
        public int ThreadPoolSize
        {
            get
            {
                return _threadPoolSize;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionThreadPoolSize(_nativePtr, value));
                _threadPoolSize = value;
            }
        }

        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to 1.        
        /// </summary>
        /// <param name="optimization_level">optimization level for the session</param>
        /// Available options are : 0, 1, 2
        /// 0 -> Disable all optimizations
        /// 1 -> Enable basic optimizations
        /// 2 -> Enable all optimizations

        private uint _graphOptimizationLevel = 1;
        public uint GraphOptimizationLevel
        {
            get
            {
                return _graphOptimizationLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionGraphOptimizationLevel(_nativePtr, value));
                _graphOptimizationLevel = value;
            }
        }

        #endregion

        #region Private Methods


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
        #endregion
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
