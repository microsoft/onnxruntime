// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    ///  Sets various runtime options. 
    /// </summary>
    public class RunOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        /// <summary>
        /// Default __ctor. Creates default RuntimeOptions
        /// </summary>
        public RunOptions()
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateRunOptions(out handle));
        }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetRunLogSeverityLevel(handle, value));
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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetRunLogVerbosityLevel(handle, value));
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
                return _logId;
            }
            set
            {
                var logIdPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(value), GCHandleType.Pinned);
                using (var pinnedlogIdName = new PinnedGCHandle(logIdPinned))
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetRunTag(handle, pinnedlogIdName.Pointer));
                }

                _logId = value;
            }
        }

        private string _logId = "";


        /// <summary>
        /// Sets a flag to terminate all Run() calls that are currently using this RunOptions object 
        /// Default = false
        /// </summary>
        /// <value>terminate flag value</value>
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
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsSetTerminate(handle));
                    _terminate = true;
                }
                else if (_terminate && !value)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtRunOptionsUnsetTerminate(handle));
                    _terminate = false;
                }
            }
        }
        private bool _terminate = false; //value set to default value of the C++ RunOptions


        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of RunOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseRunOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }
}