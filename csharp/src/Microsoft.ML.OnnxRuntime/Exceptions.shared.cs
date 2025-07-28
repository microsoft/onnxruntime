// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntime
{

    /// <summary>
    /// Enum conresponding to native onnxruntime error codes. Must be in sync with the native API
    /// </summary>
    internal enum ErrorCode
    {
        Ok = 0,
        Fail = 1,
        InvalidArgument = 2,
        NoSuchFile = 3,
        NoModel = 4,
        EngineError = 5,
        RuntimeException = 6,
        InvalidProtobuf = 7,
        ModelLoaded = 8,
        NotImplemented = 9,
        InvalidGraph = 10,
        ShapeInferenceNotRegistered = 11,  // ORT_EP_FAIL
        RequirementNotRegistered = 12,     // ORT_MODEL_LOAD_CANCELED
        ModelRequiresCompilation = 13,
        NotFound = 14,
    }

    /// <summary>
    /// The Exception that is thrown for errors related ton OnnxRuntime
    /// </summary>
    public class OnnxRuntimeException: Exception
    {
        private static Dictionary<ErrorCode, string> errorCodeToString = new Dictionary<ErrorCode, string>()
        {
            { ErrorCode.Ok, "Ok" },
            { ErrorCode.Fail, "Fail" },
            { ErrorCode.InvalidArgument, "InvalidArgument"} ,
            { ErrorCode.NoSuchFile, "NoSuchFile" },
            { ErrorCode.NoModel, "NoModel" },
            { ErrorCode.EngineError, "EngineError" },
            { ErrorCode.RuntimeException, "RuntimeException" },
            { ErrorCode.InvalidProtobuf, "InvalidProtobuf" },
            { ErrorCode.ModelLoaded, "ModelLoaded" },
            { ErrorCode.NotImplemented, "NotImplemented" },
            { ErrorCode.InvalidGraph, "InvalidGraph" },
            { ErrorCode.ShapeInferenceNotRegistered, "ShapeInferenceNotRegistered" },
            { ErrorCode.RequirementNotRegistered, "RequirementNotRegistered" },
            { ErrorCode.ModelRequiresCompilation, "ModelRequiresCompilation" },
            { ErrorCode.NotFound, "NotFound" },
        };

        internal OnnxRuntimeException(ErrorCode errorCode, string message)
            :base("[ErrorCode:" + errorCodeToString[errorCode] + "] " + message)
        {
        }
    }


}
