// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Graph optimization level to use with SessionOptions
    ///  [https://github.com/microsoft/onnxruntime/blob/main/docs/ONNX_Runtime_Graph_Optimizations.md]
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
    /// It forces the instantiation of the OrtEnv singleton.
    /// </summary>
    public class SessionOptions : SafeHandle
    {
        // Delay-loaded CUDA or cuDNN DLLs. Currently, delayload is disabled. See cmake/CMakeLists.txt for more information.
        private static string[] cudaDelayLoadedLibs = { };
        private static string[] trtDelayLoadedLibs = { };

        #region Constructor and Factory methods

        /// <summary>
        /// Constructs an empty SessionOptions
        /// </summary>
        public SessionOptions()
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionOptions(out handle));
            // Instantiate the OrtEnv singleton if not already done.
            OrtEnv.Instance();
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
            try
            {
                options.AppendExecutionProvider_CUDA(deviceId);
                return options;
            }
            catch (Exception)
            {
                options.Dispose();
                throw;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for CUDA execution provider.
        /// Use only if CUDA is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="cudaProviderOptions">CUDA EP provider options</param>
        /// <returns>A SessionsOptions() object configured for execution on provider options</returns>
        public static SessionOptions MakeSessionOptionWithCudaProvider(OrtCUDAProviderOptions cudaProviderOptions)
        {
            CheckCudaExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_CUDA(cudaProviderOptions);
                return options;
            }
            catch (Exception)
            {
                options.Dispose();
                throw;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(int deviceId = 0)
        {
            CheckTensorrtExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_Tensorrt(deviceId);
                options.AppendExecutionProvider_CUDA(deviceId);
                return options;
            }
            catch (Exception)
            {
                options.Dispose();
                throw;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for TensorRT execution provider.
        /// Use only if CUDA/TensorRT are installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trtProviderOptions">TensorRT EP provider options</param>
        /// <returns>A SessionsOptions() object configured for execution on provider options</returns>
        public static SessionOptions MakeSessionOptionWithTensorrtProvider(OrtTensorRTProviderOptions trtProviderOptions)
        {
            CheckTensorrtExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            try
            {
                // Make sure that CUDA EP uses the same device id as TensorRT EP.
                options.AppendExecutionProvider_Tensorrt(trtProviderOptions);
                options.AppendExecutionProvider_CUDA(trtProviderOptions.GetDeviceId());
                return options;
            }
            catch (Exception)
            {
                options.Dispose();
                throw;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for TVM execution.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">settings string, comprises of comma separated key:value pairs. default is empty</param>
        /// <returns>A SessionsOptions() object configured for execution with TVM</returns>
        public static SessionOptions MakeSessionOptionWithTvmProvider(String settings = "")
        {
            SessionOptions options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_Tvm(settings);
                return options;
            }
            catch (Exception)
            {
                options.Dispose();
                throw;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for ROCM execution.
        /// Use only if ROCM is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">Device Id</param>
        /// <returns>A SessionsOptions() object configured for execution on deviceId</returns>
        public static SessionOptions MakeSessionOptionWithRocmProvider(int deviceId = 0)
        {
            CheckRocmExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_ROCm(deviceId);
                return options;
            }
            catch (Exception)
            {
                options.Dispose();
                throw;
            }
        }

        /// <summary>
        /// A helper method to construct a SessionOptions object for ROCm execution provider.
        /// Use only if ROCm is installed and you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="rocmProviderOptions">ROCm EP provider options</param>
        /// <returns>A SessionsOptions() object configured for execution on provider options</returns>
        public static SessionOptions MakeSessionOptionWithRocmProvider(OrtROCMProviderOptions rocmProviderOptions)
        {
            CheckRocmExecutionProviderDLLs();
            SessionOptions options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_ROCm(rocmProviderOptions);
                return options;
            }
            catch (Exception)
            {
                options.Dispose();
                throw;
            }
        }
        #endregion

        #region ExecutionProviderAppends
        /// <summary>
        /// Appends CPU EP to a list of available execution providers for the session.
        /// </summary>
        /// <param name="useArena">1 - use arena, 0 - do not use arena</param>
        public void AppendExecutionProvider_CPU(int useArena = 1)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CPU(handle, useArena));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="useArena">1 - use allocation arena, 0 - otherwise</param>
        public void AppendExecutionProvider_Dnnl(int useArena = 1)
        {
#if __MOBILE__
            throw new NotSupportedException("The DNNL Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Dnnl(handle, useArena));
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">integer device ID</param>
        public void AppendExecutionProvider_CUDA(int deviceId = 0)
        {
#if __MOBILE__
            throw new NotSupportedException("The CUDA Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CUDA(handle, deviceId));
#endif
        }

        /// <summary>
        /// Append a CUDA EP instance (based on specified configuration) to the SessionOptions instance.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="cudaProviderOptions">CUDA EP provider options</param>
        public void AppendExecutionProvider_CUDA(OrtCUDAProviderOptions cudaProviderOptions)
        {
#if __MOBILE__
            throw new NotSupportedException("The CUDA Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.SessionOptionsAppendExecutionProvider_CUDA_V2(handle, cudaProviderOptions.Handle));
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_DML(int deviceId = 0)
        {
#if __MOBILE__
            throw new NotSupportedException("The DML Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_DML(handle, deviceId));
#endif
        }


        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification, default empty string</param>
        public void AppendExecutionProvider_OpenVINO(string deviceId = "")
        {
#if __MOBILE__
            throw new NotSupportedException("The OpenVINO Execution Provider is not supported in this build");
#else
            var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(deviceId);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_OpenVINO(handle, utf8));
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_Tensorrt(int deviceId = 0)
        {
#if __MOBILE__
            throw new NotSupportedException("The TensorRT Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Tensorrt(handle, deviceId));
#endif
        }

        /// <summary>
        /// Append a TensorRT EP instance (based on specified configuration) to the SessionOptions instance.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="trtProviderOptions">TensorRT EP provider options</param>
        public void AppendExecutionProvider_Tensorrt(OrtTensorRTProviderOptions trtProviderOptions)
        {
#if __MOBILE__
            throw new NotSupportedException("The TensorRT Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.SessionOptionsAppendExecutionProvider_TensorRT_V2(handle, trtProviderOptions.Handle));
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">Device Id</param>
        public void AppendExecutionProvider_ROCm(int deviceId = 0)
        {
#if __MOBILE__
            throw new NotSupportedException("The ROCM Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtSessionOptionsAppendExecutionProvider_ROCM(handle, deviceId));
#endif
        }

        /// <summary>
        /// Append a ROCm EP instance (based on specified configuration) to the SessionOptions instance.
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="rocmProviderOptions">ROCm EP provider options</param>
        public void AppendExecutionProvider_ROCm(OrtROCMProviderOptions rocmProviderOptions)
        {
#if __MOBILE__
            throw new NotSupportedException("The ROCm Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.SessionOptionsAppendExecutionProvider_ROCM(handle, rocmProviderOptions.Handle));
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="deviceId">device identification</param>
        public void AppendExecutionProvider_MIGraphX(int deviceId = 0)
        {
#if __MOBILE__
            throw new NotSupportedException($"The MIGraphX Execution Provider is not supported in this build");
#else
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_MIGraphX(handle, deviceId));
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="nnapiFlags">NNAPI specific flag mask</param>
        public void AppendExecutionProvider_Nnapi(NnapiFlags nnapiFlags = NnapiFlags.NNAPI_FLAG_USE_NONE)
        {
#if __ANDROID__
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(handle, (uint)nnapiFlags));
#else
            throw new NotSupportedException("The NNAPI Execution Provider is not supported in this build");
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="coremlFlags">CoreML specific flags</param>
        public void AppendExecutionProvider_CoreML(CoreMLFlags coremlFlags = CoreMLFlags.COREML_FLAG_USE_NONE)
        {
#if __IOS__
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtSessionOptionsAppendExecutionProvider_CoreML(handle, (uint)coremlFlags));
#else
#if __ENABLE_COREML__
            // only attempt if this is OSX
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                NativeApiStatus.VerifySuccess(
                    NativeMethods.OrtSessionOptionsAppendExecutionProvider_CoreML(handle, (uint)coremlFlags));
            }
            else
#endif
            {
                throw new NotSupportedException("The CoreML Execution Provider is not supported in this build");
            }
#endif
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="settings">string with TVM specific settings</param>
        public void AppendExecutionProvider_Tvm(string settings = "")
        {
#if __MOBILE__
            throw new NotSupportedException("The TVM Execution Provider is not supported in this build");
#else
            var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(settings);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Tvm(handle, utf8));
#endif
        }

        private class ExecutionProviderAppender
        {
            private byte[] _utf8ProviderName;
            internal ExecutionProviderAppender(byte[] providerName)
            {
                _utf8ProviderName = providerName;
            }

            public IntPtr Appender(IntPtr handle, IntPtr[] optKeys, IntPtr[] optValues, UIntPtr optCount)
            {
                return NativeMethods.SessionOptionsAppendExecutionProvider(
                    handle, _utf8ProviderName, optKeys, optValues, optCount);
            }
        }


        /// <summary>
        /// Append QNN, SNPE or XNNPACK execution provider
        /// </summary>
        /// <param name="providerName">Execution provider to add. 'QNN', 'SNPE' or 'XNNPACK' are currently supported.</param>
        /// <param name="providerOptions">Optional key/value pairs to specify execution provider options.</param>
        public void AppendExecutionProvider(string providerName, Dictionary<string, string> providerOptions = null)
        {
            if (providerName != "SNPE" && providerName != "XNNPACK" && providerName != "QNN" && providerName != "AZURE")
            {
                throw new NotSupportedException(
                    "Only QNN, SNPE, XNNPACK and AZURE execution providers can be enabled by this method.");
            }

            if (providerOptions == null)
            {
                providerOptions = new Dictionary<string, string>();
            }

            var utf8ProviderName = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(providerName);
            var appender = new ExecutionProviderAppender(utf8ProviderName);
            ProviderOptionsUpdater.Update(providerOptions, handle, appender.Appender);
        }
        #endregion //ExecutionProviderAppends

        #region Public Methods
        /// <summary>
        /// Loads a DLL named 'libraryPath' and looks for this entry point:
        ///   OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
        /// It then passes in the provided session options to this function along with the api base.
        ///
        /// Prior to v1.15 this leaked the library handle and RegisterCustomOpLibraryV2
        /// was added to resolve that.
        ///
        /// From v1.15 on ONNX Runtime will manage the lifetime of the handle.
        /// </summary>
        /// <param name="libraryPath">path to the custom op library</param>
        public void RegisterCustomOpLibrary(string libraryPath)
        {
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtRegisterCustomOpsLibrary_V2(
                    handle, NativeOnnxValueHelper.GetPlatformSerializedString(libraryPath))
            );
        }

        /// <summary>
        /// Loads a DLL named 'libraryPath' and looks for this entry point:
        /// OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
        /// It then passes in the provided session options to this function along with the api base.
        /// The handle to the loaded library is returned in 'libraryHandle'.
        /// It can be unloaded by the caller after all sessions using the passed in
        /// session options are destroyed, or if an error occurs and it is non null.
        /// Hint: .NET Core 3.1 has a 'NativeLibrary' class that can be used to free the library handle
        /// </summary>
        /// <param name="libraryPath">Custom op library path</param>
        /// <param name="libraryHandle">out parameter, library handle</param>
        public void RegisterCustomOpLibraryV2(string libraryPath, out IntPtr libraryHandle)
        {
            // NOTE: This is confusing due to the history.
            // SessionOptions.RegisterCustomOpLibrary initially called NativeMethods.OrtRegisterCustomOpsLibrary
            // and leaked the handle.
            // SessionOptions.RegisterCustomOpLibraryV2 was added to resolve that by returning the handle.
            // Later, NativeMethods.OrtRegisterCustomOpsLibrary_V2 was added with ORT owning the handle.
            //
            // End result of that is
            //   SessionOptions.RegisterCustomOpLibrary calls NativeMethods.OrtRegisterCustomOpsLibrary_V2
            //   SessionOptions.RegisterCustomOpLibraryV2 calls NativeMethods.OrtRegisterCustomOpsLibrary
            var utf8Path = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(libraryPath);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRegisterCustomOpsLibrary(handle, utf8Path,
                                                                                    out libraryHandle));
        }

        /// <summary>
        /// Register the custom operators from the Microsoft.ML.OnnxRuntime.Extensions NuGet package.
        /// A reference to Microsoft.ML.OnnxRuntime.Extensions must be manually added to your project.
        /// </summary>
        /// <exception cref="OnnxRuntimeException">Throws if the extensions library is not found.</exception>
        public void RegisterOrtExtensions()
        {
            try
            {
                var ortApiBase = NativeMethods.OrtGetApiBase();
                NativeApiStatus.VerifySuccess(
                    OrtExtensionsNativeMethods.RegisterCustomOps(this.handle, ref ortApiBase)
                );
            }
            catch (DllNotFoundException)
            {
                throw new OnnxRuntimeException(
                    ErrorCode.NoSuchFile,
                    "The ONNX Runtime extensions library was not found. The Microsoft.ML.OnnxRuntime.Extensions " +
                    "NuGet package must be referenced by the project to use 'OrtExtensions.RegisterCustomOps.");
            }
        }

        /// <summary>
        /// Add a pre-allocated initializer to a session. If a model contains an initializer with a name
        /// that is same as the name passed to this API call, ORT will use this initializer instance
        /// instead of deserializing one from the model file. This is useful when you want to share
        /// the same initializer across sessions.
        /// </summary>
        /// <param name="name">name of the initializer</param>
        /// <param name="ortValue">OrtValue containing the initializer. Lifetime of 'val' and the underlying initializer buffer must be
        /// managed by the user (created using the CreateTensorWithDataAsOrtValue API) and it must outlive the session object</param>
        public void AddInitializer(string name, OrtValue ortValue)
        {
            var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtAddInitializer(handle, utf8, ortValue.Handle));
        }

        /// <summary>
        /// Set a single session configuration entry as a pair of strings
        /// If a configuration with same key exists, this will overwrite the configuration with the given configValue
        /// </summary>
        /// <param name="configKey">config key name</param>
        /// <param name="configValue">config key value</param>
        public void AddSessionConfigEntry(string configKey, string configValue)
        {
            var utf8Key = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(configKey);
            var utf8Value = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(configValue);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtAddSessionConfigEntry(handle, utf8Key, utf8Value));
        }

        /// <summary>
        /// Override symbolic dimensions (by specific denotation strings) with actual values if known at session initialization time to enable
        /// optimizations that can take advantage of fixed values (such as memory planning, etc)
        /// </summary>
        /// <param name="dimDenotation">denotation name</param>
        /// <param name="dimValue">denotation value</param>
        public void AddFreeDimensionOverride(string dimDenotation, long dimValue)
        {
            var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(dimDenotation);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtAddFreeDimensionOverride(handle, utf8, dimValue));
        }

        /// <summary>
        /// Override symbolic dimensions (by specific name strings) with actual values if known at session initialization time to enable
        /// optimizations that can take advantage of fixed values (such as memory planning, etc)
        /// </summary>
        /// <param name="dimName">dimension name</param>
        /// <param name="dimValue">dimension value</param>
        public void AddFreeDimensionOverrideByName(string dimName, long dimValue)
        {
            var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(dimName);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtAddFreeDimensionOverrideByName(handle, utf8, dimValue));
        }
        #endregion

        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        #region Public Properties
        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Enables the use of the memory allocation patterns in the first Run() call for subsequent runs. Default = true.
        /// </summary>
        /// <value>returns enableMemoryPattern flag value</value>
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
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableMemPattern(handle));
                    _enableMemoryPattern = true;
                }
                else if (_enableMemoryPattern && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableMemPattern(handle));
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
        /// <value>returns _enableProfiling flag value</value>
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
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableProfiling(handle, NativeOnnxValueHelper.GetPlatformSerializedString(ProfileOutputPathPrefix)));
                    _enableProfiling = true;
                }
                else if (_enableProfiling && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableProfiling(handle));
                    _enableProfiling = false;
                }
            }
        }
        private bool _enableProfiling = false;

        /// <summary>
        ///  Set filepath to save optimized model after graph level transformations. Default is empty, which implies saving is disabled.
        /// </summary>
        /// <value>returns _optimizedModelFilePath flag value</value>
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
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtSetOptimizedModelFilePath(handle, NativeOnnxValueHelper.GetPlatformSerializedString(value)));
                    _optimizedModelFilePath = value;
                }
            }
        }
        private string _optimizedModelFilePath = "";

        /// <summary>
        /// Enables Arena allocator for the CPU memory allocations. Default is true.
        /// </summary>
        /// <value>returns _enableCpuMemArena flag value</value>
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
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableCpuMemArena(handle));
                    _enableCpuMemArena = true;
                }
                else if (_enableCpuMemArena && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableCpuMemArena(handle));
                    _enableCpuMemArena = false;
                }
            }
        }
        private bool _enableCpuMemArena = true;

        /// <summary>
        /// Log Id to be used for the session. Default is empty string.
        /// </summary>
        /// <value>returns _logId value</value>
        public string LogId
        {
            get
            {
                return _logId;
            }

            set
            {
                var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(value);
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogId(handle, utf8));
                _logId = value;
            }
        }
        private string _logId = string.Empty;

        /// <summary>
        /// Log Severity Level for the session logs. Default = ORT_LOGGING_LEVEL_WARNING
        /// </summary>
        /// <value>returns _logSeverityLevel value</value>
        public OrtLoggingLevel LogSeverityLevel
        {
            get
            {
                return _logSeverityLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogSeverityLevel(handle, value));
                _logSeverityLevel = value;
            }
        }
        private OrtLoggingLevel _logSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        /// <summary>
        /// Log Verbosity Level for the session logs. Default = 0. Valid values are >=0.
        /// This takes into effect only when the LogSeverityLevel is set to ORT_LOGGING_LEVEL_VERBOSE.
        /// </summary>
        /// <value>returns _logVerbosityLevel value</value>
        public int LogVerbosityLevel
        {
            get
            {
                return _logVerbosityLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionLogVerbosityLevel(handle, value));
                _logVerbosityLevel = value;
            }
        }
        private int _logVerbosityLevel = 0;

        /// <summary>
        // Sets the number of threads used to parallelize the execution within nodes
        // A value of 0 means ORT will pick a default
        /// </summary>
        /// <value>returns _intraOpNumThreads value</value>
        public int IntraOpNumThreads
        {
            get
            {
                return _intraOpNumThreads;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetIntraOpNumThreads(handle, value));
                _intraOpNumThreads = value;
            }
        }
        private int _intraOpNumThreads = 0; // set to what is set in C++ SessionOptions by default;

        /// <summary>
        // Sets the number of threads used to parallelize the execution of the graph (across nodes)
        // If sequential execution is enabled this value is ignored
        // A value of 0 means ORT will pick a default
        /// </summary>
        /// <value>returns _interOpNumThreads value</value>
        public int InterOpNumThreads
        {
            get
            {
                return _interOpNumThreads;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetInterOpNumThreads(handle, value));
                _interOpNumThreads = value;
            }
        }
        private int _interOpNumThreads = 0; // set to what is set in C++ SessionOptions by default;

        /// <summary>
        /// Sets the graph optimization level for the session. Default is set to ORT_ENABLE_ALL.
        /// </summary>
        /// <value>returns _graphOptimizationLevel value</value>
        public GraphOptimizationLevel GraphOptimizationLevel
        {
            get
            {
                return _graphOptimizationLevel;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionGraphOptimizationLevel(handle, value));
                _graphOptimizationLevel = value;
            }
        }
        private GraphOptimizationLevel _graphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        /// <summary>
        /// Sets the execution mode for the session. Default is set to ORT_SEQUENTIAL.
        /// See [ONNX_Runtime_Perf_Tuning.md] for more details.
        /// </summary>
        /// <value>returns _executionMode value</value>
        public ExecutionMode ExecutionMode
        {
            get
            {
                return _executionMode;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetSessionExecutionMode(handle, value));
                _executionMode = value;
            }
        }
        private ExecutionMode _executionMode = ExecutionMode.ORT_SEQUENTIAL;

        #endregion

        #region Private Methods

#if !__MOBILE__
        // Declared, but called only if OS = Windows.
        [DllImport("kernel32.dll")]
        private static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport("kernel32.dll")]
        static extern uint GetSystemDirectory([Out] StringBuilder lpBuffer, uint uSize);
#else
        private static IntPtr LoadLibrary(string dllToLoad)
        {
            throw new NotSupportedException();
        }
        static uint GetSystemDirectory([Out] StringBuilder lpBuffer, uint uSize)
        {
            throw new NotSupportedException();
        }
#endif

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

        private static bool CheckTensorrtExecutionProviderDLLs()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var dll in trtDelayLoadedLibs)
                {
                    IntPtr handle = LoadLibrary(dll);
                    if (handle != IntPtr.Zero)
                        continue;
                    var sysdir = new StringBuilder(String.Empty, 2048);
                    GetSystemDirectory(sysdir, (uint)sysdir.Capacity);
                    throw new OnnxRuntimeException(
                        ErrorCode.NoSuchFile,
                        $"kernel32.LoadLibrary():'{dll}' not found. TensorRT/CUDA are required for GPU execution. " +
                        $". Verify it is available in the system directory={sysdir}. Else copy it to the output folder."
                        );
                }
            }
            return true;
        }

        private static bool CheckRocmExecutionProviderDLLs()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new NotSupportedException("ROCm Execution Provider is not currently supported on Windows.");
            }
            return true;
        }

        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of SessionOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseSessionOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }
        #endregion
    }
}
