// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Holds the options for configuring a CUDA Execution Provider instance
    /// </summary>
    public class OrtCUDAProviderOptions : SafeHandle
    {
        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        #region Constructor

        /// <summary>
        /// Constructs an empty OrtCUDAProviderOptions instance
        /// </summary>
        public OrtCUDAProviderOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCUDAProviderOptions(out handle));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Updates  the configuration knobs of OrtCUDAProviderOptions that will eventually be used to configure a CUDA EP
        /// Please refer to the following on different key/value pairs to configure a CUDA EP and their meaning:
        /// https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/reference/execution-providers/CUDA-ExecutionProvider.md
        /// </summary>
        /// <param name="keys">keys of all the configuration knobs of a CUDA Execution Provider</param>
        /// <param name="values">values of all the configuration knobs of a CUDA Execution Provider (must match number of keys)</param>

        public void UpdateOptions(string[] keys, string[] values)
        {
            Debug.Assert(keys.Length == values.Length);

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var keysArray = NativeOnnxValueHelper.ConvertNamesToUtf8(keys, n => n, cleanupList);
                var valuesArray = NativeOnnxValueHelper.ConvertNamesToUtf8(values, n => n, cleanupList);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtUpdateCUDAProviderOptions(handle, keysArray, valuesArray, (UIntPtr)keys.Length));
            }
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #endregion

        #region Private Methods


        #endregion

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtCUDAProviderOptions
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseCUDAProviderOptions(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }

}
