using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This class handles the process-global ONNX Runtime environment instance (OrtEnv)
    /// </summary>
    public sealed class OrtEnv : SafeHandle
    {
        private static OrtEnv _instance;
        private static LogLevel envLogLevel = LogLevel.Warning;

        public const string DefaultName = @"CSharpOnnxRuntime";

        #region private methods
        private OrtEnv()  //Problem: it is not possible to pass any option for a Singleton
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnv(envLogLevel, DefaultName, out handle));
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetLanguageProjection(handle, OrtLanguageProjection.ORT_PROJECTION_CSHARP));
            }
            catch (OnnxRuntimeException)
            {
                ReleaseHandle();
                throw;
            }
        }

        private OrtEnv(ThreadingOptions threadingOptions) : base(IntPtr.Zero,true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnvWithGlobalThreadPools(envLogLevel, DefaultName, threadingOptions.Handle, out handle));
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetLanguageProjection(handle, OrtLanguageProjection.ORT_PROJECTION_CSHARP));
            }
            catch (OnnxRuntimeException)
            {
                ReleaseHandle();
                throw;
            }
        }
        
        #endregion

        #region internal methods
        /// <summary>
        /// Returns a handle to the native `OrtEnv` instance held by the singleton C# `OrtEnv` instance
        /// Exception caching: May throw an exception on every call, if the `OrtEnv` constructor threw an exception
        /// during lazy initialization
        /// </summary>
        internal IntPtr Handle => _instance.handle;

        #endregion

        #region public methods

        /// <summary>
        /// Returns an instance of OrtEnv
        /// It returns the same instance on every call - `OrtEnv` is singleton
        /// </summary>
        /// <returns>Returns a singleton instance of OrtEnv that represents native OrtEnv object</returns>
        public static OrtEnv Instance()
        {
            return _instance ?? (_instance = new OrtEnv());
        }

        /// <summary>
        /// Creates an OrtEnvironment using the specified global thread pool options.
        /// </summary>
        /// <param name="threadingOptions">The global thread pool options.</param>
        /// <returns>Returns a singleton instance of OrtEnv that represents native OrtEnv object</returns>
        /// <remarks>Unlike the other getEnvironment methods if there already is an existing OrtEnvironment this call
        /// throws `InvalidOperationException` as we cannot guarantee that the environment has the appropriate thread pool configuration.</remarks>
        public static OrtEnv GetEnvironment(ThreadingOptions threadingOptions)
        {
            if (_instance == null)
            {
                return _instance = new OrtEnv(threadingOptions);
            }

            throw new InvalidOperationException(
                "Tried to specify the thread pool when creating an OrtEnv, but one already exists.");
        }

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
        /// Queries all the execution providers supported in the native onnxruntime shared library
        /// </summary>
        /// <returns>an array of strings that represent execution provider names</returns>
        public string[] GetAvailableProviders()
        {
            IntPtr availableProvidersHandle = IntPtr.Zero;
            int numProviders;

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetAvailableProviders(out availableProvidersHandle, out numProviders));

            var availableProviders = new string[numProviders];

            try
            {
                for(int i=0; i<numProviders; ++i)
                {
                    availableProviders[i] = NativeOnnxValueHelper.StringFromNativeUtf8(Marshal.ReadIntPtr(availableProvidersHandle, IntPtr.Size * i));
                }
            }

            finally
            {
                // Looks a bit weird that we might throw in finally(...)
                // But the native method OrtReleaseAvailableProviders actually doesn't return a failure status
                // If it does, it is BUG and we would like to propagate that to the user in the form of an exception
                NativeApiStatus.VerifySuccess(NativeMethods.OrtReleaseAvailableProviders(availableProvidersHandle, numProviders));
            }

            return availableProviders;
        }


        /// <summary>
        /// Get/Set log level property of OrtEnv instance
        /// </summary>
        /// <returns>env log level</returns>
        public LogLevel EnvLogLevel
        {
            get { return envLogLevel; }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtUpdateEnvWithCustomLogLevel(Handle, value));
                envLogLevel = value;
            }
        }
        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid
        {
            get
            {
                return (handle == IntPtr.Zero);
            }
        }

        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtEnv
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseEnv(handle);
            handle = IntPtr.Zero;
            _instance = null;
            return true;
        }
        #endregion
    }
}
