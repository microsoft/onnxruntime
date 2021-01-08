// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Holds the options for creating an InferenceSession
    /// </summary>
    public class OrtCUDAProviderOptions : SafeHandle
    {
        #region Constructor and Factory methods

        /// <summary>
        /// Constructs an empty OrtCUDAProviderOptions
        /// </summary>
        public OrtCUDAProviderOptions() : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateCUDAProviderOptions(out handle));
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Constructs an empty OrtCUDAProviderOptions
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>

        public void UpdateOptions(string[] keys, string[] vals)
        {
            Debug.Assert(keys.Length == vals.Length);

        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }


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
