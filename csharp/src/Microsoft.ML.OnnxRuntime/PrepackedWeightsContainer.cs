// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{

    /// <summary>
    /// This class holds pre-packed weights of shared initializers to be shared across sessions using these initializers
    /// and thereby provide memory savings by sharing the same pre-packed versions of these shared initializers
    /// </summary>
    public class PrepackedWeightsContainer : SafeHandle
    {

        /// <summary>
        /// Constructs an empty PrepackedWeightsContainer
        /// </summary>
        public PrepackedWeightsContainer()
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreatePrepackedWeightsContainer(out handle));
        }

        /// <summary>
        /// Internal accessor to call native methods
        /// </summary>
        internal IntPtr Pointer { get { return handle; } }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to deallocate
        /// a chunk of memory using the specified allocator.
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleasePrepackedWeightsContainer(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }

}
