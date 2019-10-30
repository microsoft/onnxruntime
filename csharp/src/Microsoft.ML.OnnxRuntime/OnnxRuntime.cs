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
    }
}