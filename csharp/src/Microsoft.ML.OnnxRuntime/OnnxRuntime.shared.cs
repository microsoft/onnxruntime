// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using System.Collections.Generic;


namespace Microsoft.ML.OnnxRuntime
{
    internal struct GlobalOptions  //Options are currently not accessible to user
    {
        public string LogId { get; set; }
        public LogLevel LogLevel { get; set; }
    }

    /// <summary>
    /// Logging level used to specify amount of logging when
    /// creating environment. The lower the value is the more logging
    /// will be output. A specific value output includes everything
    /// that higher values output.
    /// </summary>
    public enum LogLevel
    {
        Verbose = 0, // Everything
        Info = 1,    // Informational
        Warning = 2, // Warnings
        Error = 3,   // Errors
        Fatal = 4    // Results in the termination of the application.
    }

    /// <summary>
    /// Language projection property for telemetry event for tracking the source usage of ONNXRUNTIME
    /// </summary>
    public enum OrtLanguageProjection
    {
        ORT_PROJECTION_C = 0,
        ORT_PROJECTION_CPLUSPLUS = 1 ,
        ORT_PROJECTION_CSHARP = 2,
        ORT_PROJECTION_PYTHON = 3,
        ORT_PROJECTION_JAVA = 4,
        ORT_PROJECTION_WINML = 5,
    }
}
