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
    /// Controls whether you want to execute operators in the graph sequentially or in parallel.
    /// Usually when the model has many branches, setting this option to ExecutionMode.ORT_PARALLEL
    /// will give you better performance.
    /// See [ONNX_Runtime_Perf_Tuning.md] for more details.
    /// </summary>
    public enum ExecutionMode
    {
        ORT_SEQUENTIAL = 0,
        ORT_PARALLEL = 1,
    }

    /// <summary>
    /// Holds the options for creating an InferenceSession
    /// </summary>
    public class SessionOptions : IDisposable
    {
        private IntPtr _nativePtr;
        private static string[] cudaDelayLoadedLibs = { "cublas64_10.dll", "cudnn64_7.dll", "curand64_10.dll" };

        #region Constructor and Factory methods

        /// <summary>
        /// Constructs an empty SessionOptions
        /// </summary>
        public SessionOptions()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionOptions(out _nativePtr));
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for CUDA execution.
        /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <returns>A SessionsOptions() object configured for execution on deviceId=0</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider()
        {
            return MakeSessionOptionWithCudaProvider(0);
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for CUDA execution.
        /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
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

        /// <summary>
        /// A helper method to construct a SessionOptions object for Nuphar execution.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">settings string, comprises of comma separated key:value pairs. default is empty</param>
        /// <returns>A SessionsOptions() object configured for execution with Nuphar</returns>
        public static SessionOptions MakeSessionOptionWithNupharProvider(String settings = "")
        {
            SessionOptions options = new SessionOptions();
            NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nuphar(options._nativePtr, 1, settings);
            return options;
        }

        #endregion

        #region ExecutionProviderAppends
        public void AppendExecutionProvider_CPU(int useArena)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(_nativePtr, useArena));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_Dnnl(int useArena)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Dnnl(_nativePtr, useArena));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_CUDA(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(_nativePtr, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_NGraph(string nGraphBackendType)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_NGraph(_nativePtr, nGraphBackendType));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_OpenVINO(string deviceId = "")
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_OpenVINO(_nativePtr, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_Tensorrt(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Tensorrt(_nativePtr, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_MIGraphX(int deviceId)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_MIGraphX(_nativePtr, deviceId));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_Nnapi()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(_nativePtr));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        public void AppendExecutionProvider_Nuphar(string settings = "")
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nuphar(_nativePtr, 1, settings));
        }
        #endregion //ExecutionProviderAppends

        #region Public Methods
        public void RegisterCustomOpLibrary(string libraryPath)
        {
            IntPtr libraryHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRegisterCustomOpsLibrary(_nativePtr, libraryPath, out libraryHandle));
        }

        #endregion

        internal IntPtr Handle
        {
            get
            {
                return _nativePtr;
            }
        }

        #region Public Properties
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
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableProfiling(_nativePtr, NativeMethods.GetPlatformSerializedString(ProfileOutputPathPrefix)));
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
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtSetOptimizedModelFilePath(_nativePtr, NativeMethods.GetPlatformSerializedString(value)));
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
        /// Log Severity Level for the session logs. Default = ORT_LOGGING_LEVEL_WARNING
        /// </summary>
        public OrtLoggingLevel LogSeverityLevel
        {
            get
            {
                return _logSeverityLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogSeverityLevel(_nativePtr, value));
                _logSeverityLevel = value;
            }
        }
        private OrtLoggingLevel _logSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        /// <summary>
        /// Log Verbosity Level for the session logs. Default = 0. Valid values are >=0.
        /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
        /// </summary>
        public int LogVerbosityLevel
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
        private int _logVerbosityLevel = 0;


        /// <summary>
        // Sets the number of threads used to parallelize the execution within nodes
        // A value of 0 means ORT will pick a default
        /// </summary>
        public int IntraOpNumThreads
        {
            get
            {
                return _intraOpNumThreads;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetIntraOpNumThreads(_nativePtr, value));
                _intraOpNumThreads = value;
            }
        }
        private int _intraOpNumThreads = 0; // set to what is set in C++ SessionOptions by default;

        /// <summary>
        // Sets the number of threads used to parallelize the execution of the graph (across nodes)
        // If sequential execution is enabled this value is ignored
        // A value of 0 means ORT will pick a default
        /// </summary>
        public int InterOpNumThreads
        {
            get
            {
                return _interOpNumThreads;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetInterOpNumThreads(_nativePtr, value));
                _interOpNumThreads = value;
            }
        }
        private int _interOpNumThreads = 0; // set to what is set in C++ SessionOptions by default;

        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to ORT_ENABLE_ALL.
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
        private GraphOptimizationLevel _graphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        /// <summary>
        /// Sets the execution mode for the session. Default is set to ORT_SEQUENTIAL.
        /// See [ONNX_Runtime_Perf_Tuning.md] for more details.
        /// </summary>
        public ExecutionMode ExecutionMode
        {
            get
            {
                return _executionMode;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionExecutionMode(_nativePtr, value));
                _executionMode = value;
            }
        }
        private ExecutionMode _executionMode = ExecutionMode.ORT_SEQUENTIAL;

        /// <summary>
        /// Enables the use of pre-packing. Default = true.
        /// </summary>
        public bool EnablePrePacking
        {
            get
            {
                return _enablePrePacking;
            }
            set
            {
                if (!_enablePrePacking && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnablePrePacking(_nativePtr));
                    _enablePrePacking = true;
                }
                else if (_enablePrePacking && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisablePrePacking(_nativePtr));
                    _enablePrePacking = false;
                }
            }
        }
        private bool _enablePrePacking = true;

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
        #region IDisposable

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                NativeMethods.OrtReleaseSessionOptions(_nativePtr);
                _nativePtr = IntPtr.Zero;
            }
        }

        #endregion
    }
}
