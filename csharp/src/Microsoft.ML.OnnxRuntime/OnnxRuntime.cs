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

    public enum LogLevel
    {
        Verbose = 0,
        Info = 1,
        Warning = 2,
        Error = 3,
        Fatal = 4
    }

    /// <summary>
    /// This class intializes the process-global ONNX runtime
    /// C# API users do not need to access this, thus kept as internal
    /// </summary>
    internal sealed class OnnxRuntime : SafeHandle
    {
        private static readonly Lazy<OnnxRuntime> _instance = new Lazy<OnnxRuntime>(()=> new OnnxRuntime());
        
        internal static IntPtr Handle  // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return _instance.Value.handle;
            }
        }

        public override bool IsInvalid
        {
            get
            {
                return (handle == IntPtr.Zero);
            }
        }

        private OnnxRuntime()  //Problem: it is not possible to pass any option for a Singleton
            :base(IntPtr.Zero, true)
        {
            // Check LibC version on Linux, before doing any onnxruntime initialization
            CheckLibcVersionGreaterThanMinimum();

            handle = IntPtr.Zero;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateEnv(LogLevel.Warning, @"CSharpOnnxRuntime", out handle));
            }
            catch (OnnxRuntimeException e)
            {
                if (handle != IntPtr.Zero)
                {
                    Delete(handle);
                    handle = IntPtr.Zero;
                }
                throw e;
            }
            
        }

        private static void Delete(IntPtr nativePtr)
        {
            NativeMethods.OrtReleaseEnv(nativePtr);
        }

        protected override bool ReleaseHandle()
        {
            Delete(handle);
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

    }
}