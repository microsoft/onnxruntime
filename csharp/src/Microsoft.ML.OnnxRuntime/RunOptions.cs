// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// Sets various runtime options. 
    public class RunOptions: IDisposable
    {
        private IntPtr _nativePtr;
        internal IntPtr Handle
        {
            get
            {
                return _nativePtr;
            }
        }


        public RunOptions()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateRunOptions(out _nativePtr));
        }


        /// <summary>
        /// LogVerbosityLevel for the Run 
        /// default == LogLevel.Verbose
        /// </summary>
        public LogLevel LogVerbosityLevel 
        {
            get
            {
                LogLevel level;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsGetRunLogVerbosityLevel(_nativePtr, out level));
                return level;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetRunLogVerbosityLevel(_nativePtr, value));
            }
        }


        /// <summary>
        /// Log tag to be used during the run. default = ""
        /// </summary>
        public string LogId 
        {
            get
            {
                string tag = null;
                IntPtr tagPtr = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsGetRunTag(_nativePtr, out tagPtr));
                tag = Marshal.PtrToStringAnsi(tagPtr); // assume ANSI string
                // should not release the memory of the tagPtr, because it returns the c_str() of the std::string being used inside RunOptions C++ class
                return tag;
            }
            set
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetRunTag(_nativePtr, value));
            }
        }


        /// <summary>
        /// Sets a flag to terminate all Run() calls that are currently using this RunOptions object 
        /// Default = false
        /// </summary>
        public bool Terminate
        {
            get
            {
                return _terminate;
            }
            set
            {
                if (!_terminate && value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetTerminate(_nativePtr));
                    _terminate = true;
                }
                else if (_terminate && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsUnsetTerminate(_nativePtr));
                    _terminate = false;
                }
            }
        }
        private bool _terminate = false; //value set to default value of the C++ RunOptions


        #region destructors disposers

        ~RunOptions()
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
            NativeMethods.OrtReleaseRunOptions(_nativePtr);
        }

        #endregion
    }
}