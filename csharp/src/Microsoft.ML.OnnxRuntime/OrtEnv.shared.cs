// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Delegate for logging function callback.
    /// Supply your function and register it with the environment to receive logging callbacks via
    /// EnvironmentCreationOptions
    /// </summary>
    /// <param name="param">Pointer to data passed into Constructor `log_param` parameter.</param>
    /// <param name="severity">Log severity level.</param>
    /// <param name="category">Log category</param>
    /// <param name="logId">Log Id.</param>
    /// <param name="codeLocation">Code location detail.</param>
    /// <param name="message">Log message.</param>
    public delegate void DOrtLoggingFunction(IntPtr param,
        OrtLoggingLevel severity,
        string category,
        string logId,
        string codeLocation,
        string message);

    /// <summary>
    /// Options you might want to supply when creating the environment.
    /// Everything is optional.
    /// </summary>
    public struct EnvironmentCreationOptions
    {
        /// <summary>
        ///  Supply a log id to identify the application using ORT, otherwise, a default one will be used
        /// </summary>
        public string logId;

        /// <summary>
        /// Initial logging level so that you can see what is going on during environment creation
        /// Default is LogLevel.Warning
        /// </summary>
        public OrtLoggingLevel? logLevel;

        /// <summary>
        /// Supply OrtThreadingOptions instance, otherwise null
        /// </summary>
        public OrtThreadingOptions threadOptions;

        /// <summary>
        /// Supply IntPtr logging param when registering logging function, otherwise IntPtr.Zero
        /// This param will be passed to the logging function when called, it is opaque for the API
        /// </summary>
        public IntPtr? loggingParam;

        /// <summary>
        /// Supply custom logging function otherwise null
        /// </summary>
        public DOrtLoggingFunction loggingFunction;
    }

    /// <summary>
    /// The singleton class OrtEnv contains the process-global ONNX Runtime environment.
    /// It sets up logging, creates system wide thread-pools (if Thread Pool options are provided)
    /// and other necessary things for OnnxRuntime to function. 
    /// 
    /// Create or access OrtEnv by calling the Instance() method. Instance() can be called multiple times.
    /// It would return the same instance.
    /// 
    /// CreateInstanceWithOptions() provides a way to create environment with options.
    /// It must be called once before Instance() is called, otherwise it would not have effect.
    /// 
    /// If the environment is not explicitly created, it will be created as needed, e.g.,
    /// when creating a SessionOptions instance.
    /// </summary>
    public sealed class OrtEnv : SafeHandle
    {
        #region Static members
        private static readonly int ORT_PROJECTION_CSHARP = 2;

        private static readonly byte[] _defaultLogId = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(@"CSharpOnnxRuntime");

        // This must be static and set before the first creation call, otherwise, has no effect.
        private static EnvironmentCreationOptions? _createOptions;

        // Lazy instantiation. _createOptions must be set before the first creation of the instance.
        private static Lazy<OrtEnv> _instance = new Lazy<OrtEnv>(CreateInstance);

        // Internal logging function that will be called from native code
        private delegate void DOrtLoggingFunctionInternal(IntPtr param,
                IntPtr severity,
                IntPtr /* utf-8 const char* */ category,
                IntPtr /* utf-8 const char* */ logid,
                IntPtr /* utf-8 const char* */ codeLocation,
                IntPtr /* utf-8 const char* */ message);

        // Must keep this delegate alive, otherwise GC will collect it and native code will call into freed memory
        private static readonly DOrtLoggingFunctionInternal _loggingFunctionInternal = LoggingFunctionThunk;

        // Customer supplied logging function (if specified)
        private static DOrtLoggingFunction _userLoggingFunction;

        #endregion

        #region Instance members

        private OrtLoggingLevel _envLogLevel;

        #endregion

        #region Private methods
        /// <summary>
        /// The only __ctor__ for OrtEnv.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="logLevel"></param>
        private OrtEnv(IntPtr handle, OrtLoggingLevel logLevel)
            : base(handle, true)
        {
            _envLogLevel = logLevel;
        }

        /// <summary>
        /// The actual logging callback to the native code
        /// </summary>
        /// <param name="param"></param>
        /// <param name="severity"></param>
        /// <param name="category"></param>
        /// <param name="logid"></param>
        /// <param name="codeLocation"></param>
        /// <param name="message"></param>
        private static void LoggingFunctionThunk(IntPtr param,
                IntPtr severity,
                IntPtr /* utf-8 const char* */ category,
                IntPtr /* utf-8 const char* */ logid,
                IntPtr /* utf-8 const char* */ codeLocation,
                IntPtr /* utf-8 const char* */ message)
        {
            var categoryStr = NativeOnnxValueHelper.StringFromNativeUtf8(category);
            var logidStr = NativeOnnxValueHelper.StringFromNativeUtf8(logid);
            var codeLocationStr = NativeOnnxValueHelper.StringFromNativeUtf8(codeLocation);
            var messageStr = NativeOnnxValueHelper.StringFromNativeUtf8(message);
            _userLoggingFunction(param, (OrtLoggingLevel)severity, categoryStr, logidStr, codeLocationStr, messageStr);
        }

        /// <summary>
        /// This is invoked only once when the first call refers _instance.Value.
        /// </summary>
        private static OrtEnv CreateInstance()
        {
            OrtEnv result = null;

            if (!_createOptions.HasValue)
            {
                // Default creation
                result = CreateDefaultEnv(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, _defaultLogId);
            }
            else
            {
                var opts = _createOptions.Value;

                var logId = (string.IsNullOrEmpty(opts.logId)) ? _defaultLogId :
                    NativeOnnxValueHelper.StringToZeroTerminatedUtf8(opts.logId);
                var logLevel = opts.logLevel ?? OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

                var threadOps = opts.threadOptions;
                var loggingFunc = opts.loggingFunction;
                var logParam = opts.loggingParam ?? IntPtr.Zero;

                if (threadOps is null && loggingFunc is null)
                {
                    result = CreateDefaultEnv(logLevel, logId);
                }
                else if (threadOps == null)
                {
                    result = CreateWithCustomLogger(logLevel, logId, logParam, loggingFunc);
                }
                else if (loggingFunc == null)
                {
                    result = CreateWithThreadingOptions(logLevel, logId, threadOps);
                }
                else
                {
                    result = CreateEnvWithCustomLoggerAndGlobalThreadPools(logLevel, logId, logParam, threadOps, loggingFunc);
                }
            }

            return result;
        }

        private static OrtEnv CreateDefaultEnv(OrtLoggingLevel logLevel, byte[] logIdUtf8)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnv(logLevel, logIdUtf8, out IntPtr handle));
            var result = new OrtEnv(handle, logLevel);
            SetLanguageProjection(result);
            return result;
        }

        private static OrtEnv CreateWithCustomLogger(OrtLoggingLevel logLevel, byte[] logIdUtf8, IntPtr loggerParam, DOrtLoggingFunction loggingFunction)
        {
            System.Diagnostics.Debug.Assert(loggingFunction != null);

            // We pass _loggingFunctionInternal which then call user supplied _userLoggingFunction
            _userLoggingFunction = loggingFunction;
            var nativeFuncPtr = Marshal.GetFunctionPointerForDelegate(_loggingFunctionInternal);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnvWithCustomLogger(
                                nativeFuncPtr, loggerParam, logLevel, logIdUtf8, out IntPtr handle));
            var result = new OrtEnv(handle, logLevel);
            SetLanguageProjection(result);
            return result;
        }

        private static OrtEnv CreateWithThreadingOptions(OrtLoggingLevel logLevel, byte[] logIdUtf8, OrtThreadingOptions threadingOptions)
        {
            System.Diagnostics.Debug.Assert(threadingOptions != null);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnvWithGlobalThreadPools(
                logLevel, logIdUtf8, threadingOptions.Handle, out IntPtr handle));
            var result = new OrtEnv(handle, logLevel);
            SetLanguageProjection(result);
            return result;
        }

        private static OrtEnv CreateEnvWithCustomLoggerAndGlobalThreadPools(OrtLoggingLevel logLevel, byte[] logIdUtf8, IntPtr logParam,
            OrtThreadingOptions threadingOptions, DOrtLoggingFunction loggingFunction)
        {
            System.Diagnostics.Debug.Assert(threadingOptions != null);
            System.Diagnostics.Debug.Assert(loggingFunction != null);

            // We pass _loggingFunctionInternal which then call user supplied _userLoggingFunction
            _userLoggingFunction = loggingFunction;
            var nativeFuncPtr = Marshal.GetFunctionPointerForDelegate(_loggingFunctionInternal);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnvWithCustomLoggerAndGlobalThreadPools(nativeFuncPtr,
                logParam, logLevel, logIdUtf8, threadingOptions.Handle, out IntPtr handle));
            var result = new OrtEnv(handle, logLevel);
            SetLanguageProjection(result);
            return result;
        }

        /// <summary>
        /// To be called only from constructor
        /// </summary>
        private static void SetLanguageProjection(OrtEnv env)
        {
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetLanguageProjection(env.Handle, ORT_PROJECTION_CSHARP));
            }
            catch (Exception)
            {
                env.Dispose();
                throw;
            }
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Instantiates (if not already done so) a new OrtEnv instance with the default logging level
        /// and no other options. Otherwise returns the existing instance.
        /// 
        /// It returns the same instance on every call - `OrtEnv` is singleton
        /// </summary>
        /// <returns>Returns a singleton instance of OrtEnv that represents native OrtEnv object</returns>
        public static OrtEnv Instance()
        {
            return _instance.Value;
        }


        /// <summary>
        /// Provides a way to create an instance with options.
        /// It throws if the instance already exists and the specified options
        /// not have effect.
        /// </summary>
        /// <param name="options"></param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException">if the singleton has already been created</exception>
        public static OrtEnv CreateInstanceWithOptions(ref EnvironmentCreationOptions options)
        {
            // Non-thread safe, best effort hopefully helpful check.
            // Environment is usually created once per process, so this should be fine.
            if (_instance.IsValueCreated)
            {
                throw new OnnxRuntimeException(ErrorCode.RuntimeException,
                    "OrtEnv singleton instance already exists, supplied options would not have effect");
            }

            _createOptions = options;
            return _instance.Value;
        }

        /// <summary>
        /// Provides visibility if singleton already been instantiated
        /// </summary>
        public static bool IsCreated { get { return _instance.IsValueCreated; } }

        /// <summary>
        /// Enable platform telemetry collection where applicable
        /// (currently only official Windows ORT builds have telemetry collection capabilities)
        /// </summary>
        public void EnableTelemetryEvents()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtEnableTelemetryEvents(Handle));
        }

        /// <summary>
        /// Disable platform telemetry collection
        /// </summary>
        public void DisableTelemetryEvents()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtDisableTelemetryEvents(Handle));
        }

        /// <summary>
        /// Create and register an allocator to the OrtEnv instance
        /// so as to enable sharing across all sessions using the OrtEnv instance
        /// <param name="memInfo">OrtMemoryInfo instance to be used for allocator creation</param>
        /// <param name="arenaCfg">OrtArenaCfg instance that will be used to define the behavior of the arena based allocator</param>
        /// </summary>
        public void CreateAndRegisterAllocator(OrtMemoryInfo memInfo, OrtArenaCfg arenaCfg)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateAndRegisterAllocator(Handle, memInfo.Pointer, arenaCfg.Pointer));
        }

        /// <summary>
        /// This function returns the onnxruntime version string
        /// </summary>
        /// <returns>version string</returns>
        public string GetVersionString()
        {
            IntPtr versionString = NativeMethods.OrtGetVersionString();
            return NativeOnnxValueHelper.StringFromNativeUtf8(versionString);
        }

        /// <summary>
        /// Queries all the execution providers supported in the native onnxruntime shared library
        /// </summary>
        /// <returns>an array of strings that represent execution provider names</returns>
        public string[] GetAvailableProviders()
        {
            IntPtr availableProvidersHandle = IntPtr.Zero;
            int numProviders;

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetAvailableProviders(out availableProvidersHandle, out numProviders));
            try
            {
                var availableProviders = new string[numProviders];
                for (int i = 0; i < numProviders; ++i)
                {
                    availableProviders[i] = NativeOnnxValueHelper.StringFromNativeUtf8(Marshal.ReadIntPtr(availableProvidersHandle, IntPtr.Size * i));
                }
                return availableProviders;
            }
            finally
            {
                // This should never throw. The original C API should have never returned status in the first place.
                // If it does, it is BUG and we would like to propagate that to the user in the form of an exception
                NativeApiStatus.VerifySuccess(NativeMethods.OrtReleaseAvailableProviders(availableProvidersHandle, numProviders));
            }
        }


        /// <summary>
        /// Get/Set log level property of OrtEnv instance
        /// Default LogLevel.Warning
        /// </summary>
        /// <returns>env log level</returns>
        public OrtLoggingLevel EnvLogLevel
        {
            get { return _envLogLevel; }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtUpdateEnvWithCustomLogLevel(Handle, value));
                _envLogLevel = value;
            }
        }

        #endregion

        #region SafeHandle overrides
        /// <summary>
        /// Returns a handle to the native `OrtEnv` instance held by the singleton C# `OrtEnv` instance
        /// Exception caching: May throw an exception on every call, if the `OrtEnv` constructor threw an exception
        /// during lazy initialization
        /// </summary>
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Destroys native object
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseEnv(handle);
            handle = IntPtr.Zero;
            // Re-create empty Lazy initializer
            // This is great for tests
            _instance = new Lazy<OrtEnv>(CreateInstance);
            return true;
        }
        #endregion
    }
}
