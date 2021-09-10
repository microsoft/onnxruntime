// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntime
{
    [Flags]
    public enum NNAPIFlags
    {
        NNAPI_FLAG_USE_NONE = 0x000,
        NNAPI_FLAG_USE_FP16 = 0x001,
        NNAPI_FLAG_USE_NCHW = 0x002,
        NNAPI_FLAG_CPU_DISABLED = 0x004,
        NNAPI_FLAG_LAST = NNAPI_FLAG_CPU_DISABLED
    }

    /// <summary>
    /// Represents the platform-specific options for creating an InferenceSession.
    /// </summary>
    /// <see cref="https://onnxruntime.ai"/>
    /// <remarks>Exposes only those APIs supported on Android.</remarks>
    public partial class SessionOptions
    {
        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="nnapi_flags">nnapi specific flags</param>
        public void AppendExecutionProvider_Nnapi(NNAPIFlags nnapi_flags = NNAPIFlags.NNAPI_FLAG_USE_NONE)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(handle, (uint)nnapi_flags));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="nnapi_flags">nnapi specific flag mask</param>
        public void AppendExecutionProvider_Nnapi(uint nnapi_flags = 0)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(handle, nnapi_flags));
        }
    }
}