// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// Sets various runtime options. 
    public class RunOptions : IDisposable
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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetRunLogSeverityLevel(_nativePtr, value));
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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetRunLogVerbosityLevel(_nativePtr, value));
                _logVerbosityLevel = value;
            }
        }
        private int _logVerbosityLevel = 0;

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
                NativeMethods.OrtReleaseRunOptions(_nativePtr);
            }
        }

        #endregion
    }
}