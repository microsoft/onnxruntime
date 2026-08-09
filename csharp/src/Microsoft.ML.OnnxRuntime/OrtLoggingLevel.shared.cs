// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Log severity levels
    /// 
    /// Must in sync with OrtLoggingLevel in onnxruntime_c_api.h
    /// </summary>
    public enum OrtLoggingLevel
    {
        ORT_LOGGING_LEVEL_VERBOSE = 0,
        ORT_LOGGING_LEVEL_INFO = 1,
        ORT_LOGGING_LEVEL_WARNING = 2,
        ORT_LOGGING_LEVEL_ERROR = 3,
        ORT_LOGGING_LEVEL_FATAL = 4,
    }
}