// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;
using System.Collections.Generic;


namespace Microsoft.ML.OnnxRuntime
{
    internal struct GlobalOptions  //Options are currently not accessible to user
    {
        public string LogId { get; set; }
        public LogLevel LogLevel { get; set; }
    }

    internal enum LogLevel
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
    internal sealed class OnnxRuntime : IDisposable
    {
        // static singleton
        private static readonly Lazy<OnnxRuntime> _instance = new Lazy<OnnxRuntime>(() => new OnnxRuntime());

        // member variables
        private IntPtr _nativeHandle;

        internal static OnnxRuntime Instance  // May throw exception in every access, if the constructor have thrown an exception
        {
            get
            {
                return _instance.Value;
            }
        }


        private OnnxRuntime()  //Problem: it is not possible to pass any option for a Singleton
        {
            _nativeHandle = IntPtr.Zero;

            IntPtr outPtr;
            IntPtr status = NativeMethods.ONNXRuntimeInitialize(LogLevel.Warning, @"CSharpOnnxRuntime", out outPtr);

            NativeApiStatus.VerifySuccess(status);
            _nativeHandle = outPtr;
        }


        internal IntPtr NativeHandle
        {
            get
            {
                return _nativeHandle;
            }
        }


        ~OnnxRuntime()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                //release managed resource
            }

            //release unmanaged resource
            if (_nativeHandle != IntPtr.Zero)
            {
                NativeMethods.ReleaseONNXEnv(_nativeHandle);
            }
        }
    }
}
