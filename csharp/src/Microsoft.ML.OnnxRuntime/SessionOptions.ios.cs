// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntime
{
    [Flags]
    public enum COREMLFlags : uint
    {
        COREML_FLAG_USE_NONE = 0x000,
        COREML_FLAG_USE_CPU_ONLY = 0x001,
        COREML_FLAG_ENABLE_ON_SUBGRAPH = 0x002,
        COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004,
        COREML_FLAG_LAST = COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE,
    }

    /// <summary>
    /// Represents the platform-specific options for creating an InferenceSession.
    /// </summary>
    /// <see cref="https://onnxruntime.ai"/>
    /// <remarks>Exposes only those APIs supported on iOS.</remarks>
    public partial class SessionOptions
    {
        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="coreml_flags">CoreML specific flags</param>
        public void AppendExecutionProvider_CoreML(COREMLFlags coreml_flags = COREMLFlags.COREML_FLAG_USE_NONE)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CoreML(handle, (uint)coreml_flags));
        }

        /// <summary>
        /// Use only if you have the onnxruntime package specific to this Execution Provider.
        /// </summary>
        /// <param name="coreml_flags">CoreML specific flag mask</param>
        public void AppendExecutionProvider_CoreML(uint coreml_flags = 0)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionOptionsAppendExecutionProvider_CoreML(handle, coreml_flags));
        }
    }
}