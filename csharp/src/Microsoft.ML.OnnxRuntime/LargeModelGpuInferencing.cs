// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;

namespace Microsoft.ML.OnnxRuntime
{
    public class RequestBatch : SafeHandle
    {
        public RequestBatch()
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateRequestBatch(out handle));
        }

        public void Clear()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtClearRequestBatch(handle));
        }
        internal IntPtr Pointer
        {
            get
            {
                return handle;
            }
        }

        #region SafeHandle

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtEnv
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseRequestBatch(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion

    }

    public class ResponseBatch : SafeHandle
    {
        public ResponseBatch()
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateResponseBatch(out handle));
        }

        internal ResponseBatch(IntPtr handle)
    : base(handle, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateResponseBatch(out handle));
        }

        public void Clear()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtClearResponseBatch(handle));
        }


        internal IntPtr Pointer
        {
            get
            {
                return handle;
            }
        }

        #region SafeHandle

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtEnv
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseResponseBatch(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion

    }

    public class PipelineSession : SafeHandle
    {
        public PipelineSession(string ensembleConfigFilePath)
            : base(IntPtr.Zero, true)
        {
            var ensembleConfigFilePathPinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(ensembleConfigFilePath), GCHandleType.Pinned);
            using (var ensembleConfigFilePathPinnedHandle = new PinnedGCHandle(ensembleConfigFilePathPinned))
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreatePipelineSession(OrtEnv.Handle, ensembleConfigFilePathPinnedHandle.Pointer, out handle));
            }
        }

        public ResponseBatch Run(RequestBatch requestBatch, int numSteps)
        {
            IntPtr responseBatch = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtPipelineSessionRun(handle, requestBatch.Handle, out responseBatch, numSteps));
            return new ResponseBatch(responseBatch);
        }

        internal IntPtr Pointer
        {
            get
            {
                return handle;
            }
        }

        #region SafeHandle

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }

        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtEnv
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleasePipelineSession(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion

    }

}
