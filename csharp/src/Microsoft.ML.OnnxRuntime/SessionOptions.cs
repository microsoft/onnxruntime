// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Text;
using System.Runtime.InteropServices;
using System.IO;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// TODO Add documentation about which optimizations are enabled for each value.
    /// </summary>
    public enum GraphOptimizationLevel
    {
        ORT_DISABLE_ALL = 0,
        ORT_ENABLE_BASIC = 1,
        ORT_ENABLE_EXTENDED = 2,
        ORT_ENABLE_ALL = 99
    }

    /// <summary>
    /// Holds the options for creating an InferenceSession
    /// </summary>
    public class SessionOptions : IDisposable
    {
        private IntPtr _nativePtr;
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
        /// Enable Sequential Execution. Default = true.
        /// </summary>
        /// </param>
        /// 
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
        private bool _enableSequentialExecution = true;


        /// <summary>
        /// Enables the use of the memory allocation patterns in the first Run() call for subsequent runs. Default = true.
        /// </summary>
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
        private bool _enableMemoryPattern = true;


        /// <summary>
        /// Path prefix to use for output of profiling data
        /// </summary>
        public string ProfileOutputPathPrefix
        {
            get; set;
        } = "onnxruntime_profile_";   // this is the same default in C++ implementation



        /// <summary>
        /// Enables profiling of InferenceSession.Run() calls. Default is false
        /// </summary>
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
        private bool _enableProfiling = false;

        /// <summary>
        ///  Set filepath to save optimized model after graph level transformations. Default is empty, which implies saving is disabled.
        /// </summary>
        public string OptimizedModelFilePath
        {
            get
            {
                return _optimizedModelFilePath;
            }
            set
            {
                if (value != _optimizedModelFilePath)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtSetOptimizedModelFilePath(_nativePtr, value));
                    _optimizedModelFilePath = value;
                }
            }
        }
        private string _optimizedModelFilePath = "";



        /// <summary>
        /// Enables Arena allocator for the CPU memory allocations. Default is true.
        /// </summary>
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
        private bool _enableCpuMemArena = true;


        /// <summary>
        /// Log Id to be used for the session. Default is empty string.
        /// TODO: Should it be named LogTag as in RunOptions?
        /// </summary>
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
        private string _logId = "";


        /// <summary>
        /// Log Verbosity Level for the session logs. Default = LogLevel.Verbose
        /// </summary>
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
        private LogLevel _logVerbosityLevel = LogLevel.Verbose;


        /// <summary>
        /// Threadpool size for the session.Run() calls. 
        /// Default = 0, meaning threadpool size is aumatically selected from number of available cores.
        /// </summary>
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
        private int _threadPoolSize = 0; // set to what is set in C++ SessionOptions by default;


        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to ORT_ENABLE_BASIC.        
        /// </summary>
        public GraphOptimizationLevel GraphOptimizationLevel
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
        private GraphOptimizationLevel _graphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;

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
