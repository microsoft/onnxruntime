// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Collections.Generic;


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
        Fatal = 4    // This is it
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
    /// This class initializes the process-global ONNX Runtime environment instance (OrtEnv)
    /// </summary>
    public sealed class OrtEnv : SafeHandle
    {
        private static readonly Lazy<OrtEnv> _instance = new Lazy<OrtEnv>(()=> new OrtEnv());

        #region private methods
        private OrtEnv()  //Problem: it is not possible to pass any option for a Singleton
    : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnv(LogLevel.Warning, @"CSharpOnnxRuntime", out handle));
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSetLanguageProjection(handle, OrtLanguageProjection.ORT_PROJECTION_CSHARP));
            }
            catch (OnnxRuntimeException e)
            {
                ReleaseHandle();
                throw e;
            }
        }
        #endregion

        #region internal methods
        /// <summary>
        /// Returns a handle to the native `OrtEnv` instance held by the singleton C# `OrtEnv` instance
        /// Exception caching: May throw an exception on every call, if the `OrtEnv` constructor threw an exception
        /// during lazy initialization
        /// </summary>
        internal static IntPtr Handle  
        {
            get
            {
                return _instance.Value.handle;
            }
        }
        #endregion

        #region public methods

        /// <summary>
        /// Returns an instance of OrtEnv
        /// It returns the same instance on every call - `OrtEnv` is singleton
        /// </summary>
        /// <returns>Returns a singleton instance or OrtEnv that represents native OrtEnv object</returns>
        public static OrtEnv Instance() { return _instance.Value; }

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
            return true;
        }
        #endregion
    }
}