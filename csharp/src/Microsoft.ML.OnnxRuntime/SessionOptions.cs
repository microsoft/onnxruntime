// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    public class SessionOptions : IDisposable
    {
        private static SessionOptions _defaultOptions = new SessionOptions();
        private IntPtr _nativeHandle;

        public SessionOptions()
        {
            _nativeHandle = NativeMethods.ONNXRuntimeCreateSessionOptions();
        }

        internal IntPtr NativeHandle
        {
            get
            {
                return _nativeHandle;
            }
        }

        public static SessionOptions Default
        {
            get
            {
                return _defaultOptions;
            }
        }

        #region destructors disposers
        ~SessionOptions()
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

            // cleanup unmanaged resources
        }
        #endregion
    }
}
