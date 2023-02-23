// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace Microsoft.ML.OnnxRuntime
{
    internal struct GlobalOptions  //Options are currently not accessible to user
    {
        public string LogId { get; set; }
        public LogLevel LogLevel { get; set; }
    }

    /// <summary>
    /// Logging level used to specify amount of logging when
    /// creating environment. The lower the value is the more logging
    /// will be output. A specific value output includes everything
    /// that higher values output.
    /// </summary>
    public enum LogLevel
    {
        Verbose = 0, // Everything
        Info = 1,    // Informational
        Warning = 2, // Warnings
        Error = 3,   // Errors
        Fatal = 4    // Results in the termination of the application.
    }

    /// <summary>
    /// Language projection property for telemetry event for tracking the source usage of ONNXRUNTIME
    /// </summary>
    public enum OrtLanguageProjection
    {
        ORT_PROJECTION_C = 0,
        ORT_PROJECTION_CPLUSPLUS = 1 ,
        ORT_PROJECTION_CSHARP = 2,
        ORT_PROJECTION_PYTHON = 3,
        ORT_PROJECTION_JAVA = 4,
        ORT_PROJECTION_WINML = 5,
    }

    /// <summary>
    /// This class initializes the process-global ONNX Runtime environment instance (OrtEnv).
    /// The singleton class OrtEnv contains the process-global ONNX Runtime environment.
    /// It sets up logging, creates system wide thread-pools (if Thread Pool options are provided)
    /// and other necessary things for OnnxRuntime to function. Create or access OrtEnv by calling
    /// the Instance() method. Call this method before doing anything else in your application.
    /// </summary>
    public sealed class OrtEnv : IDisposable
    {
        private static Object _lock = new Object();
        private static OrtEnv _instance = null;

        private bool  _disposed = false;
        private IntPtr handle = IntPtr.Zero;
        private LogLevel envLogLevel = LogLevel.Warning;

#region private methods
        /// <summary>
        /// To be called only from constructor
        /// </summary>
        private void SetLanguageProjection()
        {
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

        private OrtEnv()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnv(envLogLevel, @"CSharpOnnxRuntime", out handle));
            SetLangugageProjection();
        }

        private OrtEnv(OrtThreadingOptions opt)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnvWithGlobalThreadPools(envLogLevel, @"CSharpOnnxRuntime", opt.Handle, out handle));
            SetLangugageProjection();
        }

#endregion

#region internal methods
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
        #endregion

#region public methods

        /// <summary>
        /// Creates and return instance of OrtEnv, or returns the one that already was created
        /// possibly with another Instance() overload
        /// It returns the same instance on every call - `OrtEnv` is singleton
        /// </summary>
        /// <returns>Returns a singleton instance of OrtEnv that represents native OrtEnv object</returns>
        public static OrtEnv Instance() 
        {
            if (_instance == null)
            {
                lock (_lock)
                {
                    if (_instance == null)
                    {
                        var inst = new OrtEnv();
                        Interlocked.MemoryBarrier();
                        _instance = inst;
                    }
                }
            }
            return _instance;
        }

        /// <summary>
        /// Creates and returns a new instance of OrtEnv created with OrtThreadingOptions
        /// and assigns it _instance. If successful, the Instance() method will always return
        /// OrtEnv instance created by this method.
        /// 
        /// This function can only be called once.
        /// </summary>
        /// <param name="opt">threading options instance</param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException">when the singleton was already initialized</exception>
        public static OrtEnv Instance(OrtThreadingOptions opt)
        {
            if(_instance != null)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail,
                    "Singleton object already instantiated. Threading options would not take effect");
            }

            lock (_lock)
            {
                if (_instance == null)
                {
                    var inst = new OrtEnv(opt);
                    Interlocked.MemoryBarrier();
                    _instance = inst;
                }
            }
            return _instance;
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
            try
            {
                var availableProviders = new string[numProviders];
                for (int i=0; i<numProviders; ++i)
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

#region IDisposable
        /// <summary>
        /// Destroys native object
        /// </summary>
        /// <returns>always returns true</returns>
        void ReleaseHandle()
        {
            NativeMethods.OrtReleaseEnv(handle);
            handle = IntPtr.Zero;
        }

        /// <summary>
        /// Finalizer. to cleanup session in case it runs
        /// and the user forgets to Dispose() of the session
        /// </summary>
        ~OrtEnv()
        {
            Dispose(false);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose() method</param>
        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            // cleanup unmanaged resources
            if (handle != IntPtr.Zero)
            {
                ReleaseHandle();
            }

            // we are assuming this is the last thing the program is doing
            var p = Interlocked.Exchange(ref _instance, null);
            // Expecting that this was the instance, otherwise a bug
            Debug.Assert(p == this);

            _disposed = true;
        }
#endregion
    }
}
