using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents a disposable OrtValue
    /// If necessary maybe made public and more
    /// functionality added. Right now it is disposable
    /// </summary>
    internal class OrtValue : IDisposable
    {
        private IntPtr _handle;

        internal OrtValue(IntPtr handle)
        {
            _handle = handle;
        }

        internal IntPtr Handle
        {
            get
            {
                return _handle; 
            }
        }

        #region Disposable Support
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                NativeMethods.OrtReleaseValue(_handle);
                _handle = IntPtr.Zero;
            }
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        // No need for the finalizer
        #endregion
    }
}
