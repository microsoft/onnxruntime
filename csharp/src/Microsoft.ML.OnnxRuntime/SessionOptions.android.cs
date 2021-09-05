// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime
{
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
        /// <param name="nnapi_flags">nnapi specific flag mask</param>
        public void AppendExecutionProvider_Nnapi(uint nnapi_flags = 0)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_Nnapi(handle, nnapi_flags));
        }
    }
}