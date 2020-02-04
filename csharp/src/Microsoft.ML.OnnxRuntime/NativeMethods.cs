// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
    [StructLayout(LayoutKind.Sequential)]
    public struct OrtApiBase
    {
        public IntPtr GetApi;
        public IntPtr GetVersionString;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct OrtApi
    {
        public IntPtr CreateStatus;
        public IntPtr GetErrorCode;
        public IntPtr GetErrorMessage;
        public IntPtr CreateEnv;
        public IntPtr CreateEnvWithCustomLogger;
        public IntPtr EnableTelemetryEvents;
        public IntPtr DisableTelemetryEvents;
        public IntPtr CreateSession;
        public IntPtr CreateSessionFromArray;
        public IntPtr Run;

        public IntPtr CreateSessionOptions;
        public IntPtr SetOptimizedModelFilePath;
        public IntPtr CloneSessionOptions;
        public IntPtr SetSessionExecutionMode;
        public IntPtr EnableProfiling;
        public IntPtr DisableProfiling;
        public IntPtr EnableMemPattern;
        public IntPtr DisableMemPattern;
        public IntPtr EnableCpuMemArena;
        public IntPtr DisableCpuMemArena;
        public IntPtr SetSessionLogId;
        public IntPtr SetSessionLogVerbosityLevel;
        public IntPtr SetSessionLogSeverityLevel;
        public IntPtr SetSessionGraphOptimizationLevel;
        public IntPtr SetIntraOpNumThreads;
        public IntPtr SetInterOpNumThreads;

        public IntPtr CreateCustomOpDomain;
        public IntPtr CustomOpDomain_Add;
        public IntPtr AddCustomOpDomain;
        public IntPtr RegisterCustomOpsLibrary;

        public IntPtr SessionGetInputCount;
        public IntPtr SessionGetOutputCount;
        public IntPtr SessionGetOverridableInitializerCount;
        public IntPtr SessionGetInputTypeInfo;
        public IntPtr SessionGetOutputTypeInfo;
        public IntPtr SessionGetOverridableInitializerTypeInfo;
        public IntPtr SessionGetInputName;
        public IntPtr SessionGetOutputName;
        public IntPtr SessionGetOverridableInitializerName;
        public IntPtr CreateRunOptions;
        public IntPtr RunOptionsSetRunLogVerbosityLevel;
        public IntPtr RunOptionsSetRunLogSeverityLevel;
        public IntPtr RunOptionsSetRunTag;
        public IntPtr RunOptionsGetRunLogVerbosityLevel;
        public IntPtr RunOptionsGetRunLogSeverityLevel;
        public IntPtr RunOptionsGetRunTag;
        public IntPtr RunOptionsSetTerminate;
        public IntPtr RunOptionsUnsetTerminate;

        public IntPtr CreateTensorAsOrtValue;
        public IntPtr CreateTensorWithDataAsOrtValue;
        public IntPtr IsTensor;
        public IntPtr GetTensorMutableData;
        public IntPtr FillStringTensor;

        public IntPtr GetStringTensorDataLength;
        public IntPtr GetStringTensorContent;

        public IntPtr CastTypeInfoToTensorInfo;
        public IntPtr GetOnnxTypeFromTypeInfo;
        public IntPtr CreateTensorTypeAndShapeInfo;
        public IntPtr SetTensorElementType;

        public IntPtr SetDimensions;
        public IntPtr GetTensorElementType;
        public IntPtr GetDimensionsCount;
        public IntPtr GetDimensions;
        public IntPtr GetSymbolicDimensions;
        public IntPtr GetTensorShapeElementCount;
        public IntPtr GetTensorTypeAndShape;
        public IntPtr GetTypeInfo;
        public IntPtr GetValueType;
        public IntPtr CreateMemoryInfo;
        public IntPtr CreateCpuMemoryInfo;
        public IntPtr CompareMemoryInfo;
        public IntPtr MemoryInfoGetName;
        public IntPtr MemoryInfoGetId;
        public IntPtr MemoryInfoGetMemType;
        public IntPtr MemoryInfoGetType;
        public IntPtr AllocatorAlloc;
        public IntPtr AllocatorFree;
        public IntPtr AllocatorGetInfo;
        public IntPtr GetAllocatorWithDefaultOptions;
        public IntPtr AddFreeDimensionOverride;
        public IntPtr GetValue;
        public IntPtr GetValueCount;
        public IntPtr CreateValue;
        public IntPtr CreateOpaqueValue;
        public IntPtr GetOpaqueValue;

        public IntPtr KernelInfoGetAttribute_float;
        public IntPtr KernelInfoGetAttribute_int64;
        public IntPtr KernelInfoGetAttribute_string;
        public IntPtr KernelContext_GetInputCount;
        public IntPtr KernelContext_GetOutputCount;
        public IntPtr KernelContext_GetInput;
        public IntPtr KernelContext_GetOutput;

        public IntPtr ReleaseEnv;
        public IntPtr ReleaseStatus;
        public IntPtr ReleaseMemoryInfo;
        public IntPtr ReleaseSession;
        public IntPtr ReleaseValue;
        public IntPtr ReleaseRunOptions;
        public IntPtr ReleaseTypeInfo;
        public IntPtr ReleaseTensorTypeAndShapeInfo;
        public IntPtr ReleaseSessionOptions;
        public IntPtr ReleaseCustomOpDomain;
    }

    internal static class NativeMethods
    {
        private const string nativeLib = "onnxruntime";
        internal const CharSet charSet = CharSet.Ansi;

        static OrtApi api_;

        public delegate ref OrtApi DOrtGetApi(UInt32 version);

        static NativeMethods()
        {
            DOrtGetApi OrtGetApi = (DOrtGetApi)Marshal.GetDelegateForFunctionPointer(OrtGetApiBase().GetApi, typeof(DOrtGetApi));

            // TODO: Make this save the pointer, and not copy the whole structure across
            api_ = (OrtApi)OrtGetApi(1 /*ORT_API_VERSION*/);

            OrtCreateEnv = (DOrtCreateEnv)Marshal.GetDelegateForFunctionPointer(api_.CreateEnv, typeof(DOrtCreateEnv));
            OrtReleaseEnv = (DOrtReleaseEnv)Marshal.GetDelegateForFunctionPointer(api_.ReleaseEnv, typeof(DOrtReleaseEnv));
            OrtGetErrorCode = (DOrtGetErrorCode)Marshal.GetDelegateForFunctionPointer(api_.GetErrorCode, typeof(DOrtGetErrorCode));
            OrtGetErrorMessage = (DOrtGetErrorMessage)Marshal.GetDelegateForFunctionPointer(api_.GetErrorMessage, typeof(DOrtGetErrorMessage));
            OrtReleaseStatus = (DOrtReleaseStatus)Marshal.GetDelegateForFunctionPointer(api_.ReleaseStatus, typeof(DOrtReleaseStatus));

            OrtCreateSession = (DOrtCreateSession)Marshal.GetDelegateForFunctionPointer(api_.CreateSession, typeof(DOrtCreateSession));
            OrtCreateSessionFromArray = (DOrtCreateSessionFromArray)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionFromArray, typeof(DOrtCreateSessionFromArray));
            OrtRun = (DOrtRun)Marshal.GetDelegateForFunctionPointer(api_.Run, typeof(DOrtRun));
            OrtSessionGetInputCount = (DOrtSessionGetInputCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputCount, typeof(DOrtSessionGetInputCount));
            OrtSessionGetOutputCount = (DOrtSessionGetOutputCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputCount, typeof(DOrtSessionGetOutputCount));
            OrtSessionGetOverridableInitializerCount = (DOrtSessionGetOverridableInitializerCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerCount, typeof(DOrtSessionGetOverridableInitializerCount));

            OrtSessionGetInputName = (DOrtSessionGetInputName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputName, typeof(DOrtSessionGetInputName));
            OrtSessionGetOutputName = (DOrtSessionGetOutputName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputName, typeof(DOrtSessionGetOutputName));
            OrtSessionGetOverridableInitializerName = (DOrtSessionGetOverridableInitializerName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerName, typeof(DOrtSessionGetOverridableInitializerName));
            OrtSessionGetInputTypeInfo = (DOrtSessionGetInputTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputTypeInfo, typeof(DOrtSessionGetInputTypeInfo));
            OrtSessionGetOutputTypeInfo = (DOrtSessionGetOutputTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputTypeInfo, typeof(DOrtSessionGetOutputTypeInfo));
            OrtSessionGetOverridableInitializerTypeInfo = (DOrtSessionGetOverridableInitializerTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerTypeInfo, typeof(DOrtSessionGetOverridableInitializerTypeInfo));

            OrtReleaseTypeInfo = (DOrtReleaseTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTypeInfo, typeof(DOrtReleaseTypeInfo));
            OrtReleaseSession = (DOrtReleaseSession)Marshal.GetDelegateForFunctionPointer(api_.ReleaseSession, typeof(DOrtReleaseSession));

            OrtCreateSessionOptions = (DOrtCreateSessionOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionOptions, typeof(DOrtCreateSessionOptions));
            OrtReleaseSessionOptions = (DOrtReleaseSessionOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseSessionOptions, typeof(DOrtReleaseSessionOptions));
            OrtCloneSessionOptions = (DOrtCloneSessionOptions)Marshal.GetDelegateForFunctionPointer(api_.CloneSessionOptions, typeof(DOrtCloneSessionOptions));
            OrtSetSessionExecutionMode = (DOrtSetSessionExecutionMode)Marshal.GetDelegateForFunctionPointer(api_.SetSessionExecutionMode, typeof(DOrtSetSessionExecutionMode));
            OrtSetOptimizedModelFilePath = (DOrtSetOptimizedModelFilePath)Marshal.GetDelegateForFunctionPointer(api_.SetOptimizedModelFilePath, typeof(DOrtSetOptimizedModelFilePath));
            OrtEnableProfiling = (DOrtEnableProfiling)Marshal.GetDelegateForFunctionPointer(api_.EnableProfiling, typeof(DOrtEnableProfiling));
            OrtDisableProfiling = (DOrtDisableProfiling)Marshal.GetDelegateForFunctionPointer(api_.DisableProfiling, typeof(DOrtDisableProfiling));
            OrtEnableMemPattern = (DOrtEnableMemPattern)Marshal.GetDelegateForFunctionPointer(api_.EnableMemPattern, typeof(DOrtEnableMemPattern));
            OrtDisableMemPattern = (DOrtDisableMemPattern)Marshal.GetDelegateForFunctionPointer(api_.DisableMemPattern, typeof(DOrtDisableMemPattern));
            OrtEnableCpuMemArena = (DOrtEnableCpuMemArena)Marshal.GetDelegateForFunctionPointer(api_.EnableCpuMemArena, typeof(DOrtEnableCpuMemArena));
            OrtDisableCpuMemArena = (DOrtDisableCpuMemArena)Marshal.GetDelegateForFunctionPointer(api_.DisableCpuMemArena, typeof(DOrtDisableCpuMemArena));
            OrtSetSessionLogId = (DOrtSetSessionLogId)Marshal.GetDelegateForFunctionPointer(api_.SetSessionLogId, typeof(DOrtSetSessionLogId));
            OrtSetSessionLogVerbosityLevel = (DOrtSetSessionLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.SetSessionLogVerbosityLevel, typeof(DOrtSetSessionLogVerbosityLevel));
            OrtSetSessionLogSeverityLevel = (DOrtSetSessionLogSeverityLevel)Marshal.GetDelegateForFunctionPointer(api_.SetSessionLogSeverityLevel, typeof(DOrtSetSessionLogSeverityLevel));
            OrtSetInterOpNumThreads = (DOrtSetInterOpNumThreads)Marshal.GetDelegateForFunctionPointer(api_.SetInterOpNumThreads, typeof(DOrtSetInterOpNumThreads));
            OrtSetIntraOpNumThreads = (DOrtSetIntraOpNumThreads)Marshal.GetDelegateForFunctionPointer(api_.SetIntraOpNumThreads, typeof(DOrtSetIntraOpNumThreads));
            OrtSetSessionGraphOptimizationLevel = (DOrtSetSessionGraphOptimizationLevel)Marshal.GetDelegateForFunctionPointer(api_.SetSessionGraphOptimizationLevel, typeof(DOrtSetSessionGraphOptimizationLevel));
            OrtRegisterCustomOpsLibrary = (DOrtRegisterCustomOpsLibrary)Marshal.GetDelegateForFunctionPointer(api_.RegisterCustomOpsLibrary, typeof(DOrtRegisterCustomOpsLibrary));

            OrtCreateRunOptions = (DOrtCreateRunOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateRunOptions, typeof(DOrtCreateRunOptions));
            OrtReleaseRunOptions = (DOrtReleaseRunOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseRunOptions, typeof(DOrtReleaseRunOptions));
            OrtRunOptionsSetRunLogVerbosityLevel = (DOrtRunOptionsSetRunLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunLogVerbosityLevel, typeof(DOrtRunOptionsSetRunLogVerbosityLevel));
            OrtRunOptionsSetRunTag = (DOrtRunOptionsSetRunTag)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunTag, typeof(DOrtRunOptionsSetRunTag));
            OrtRunOptionsGetRunLogVerbosityLevel = (DOrtRunOptionsGetRunLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunLogVerbosityLevel, typeof(DOrtRunOptionsGetRunLogVerbosityLevel));
            OrtRunOptionsGetRunTag = (DOrtRunOptionsGetRunTag)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunTag, typeof(DOrtRunOptionsGetRunTag));
            OrtRunOptionsSetTerminate = (DOrtRunOptionsSetTerminate)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetTerminate, typeof(DOrtRunOptionsSetTerminate));
            OrtRunOptionsUnsetTerminate = (DOrtRunOptionsUnsetTerminate)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsUnsetTerminate, typeof(DOrtRunOptionsUnsetTerminate));

            OrtCreateMemoryInfo = (DOrtCreateMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CreateMemoryInfo, typeof(DOrtCreateMemoryInfo));
            OrtCreateCpuMemoryInfo = (DOrtCreateCpuMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CreateCpuMemoryInfo, typeof(DOrtCreateCpuMemoryInfo));
            OrtReleaseMemoryInfo = (DOrtReleaseMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseMemoryInfo, typeof(DOrtReleaseMemoryInfo));
            OrtGetAllocatorWithDefaultOptions = (DOrtGetAllocatorWithDefaultOptions)Marshal.GetDelegateForFunctionPointer(api_.GetAllocatorWithDefaultOptions, typeof(DOrtGetAllocatorWithDefaultOptions));
            OrtAllocatorFree = (DOrtAllocatorFree)Marshal.GetDelegateForFunctionPointer(api_.AllocatorFree, typeof(DOrtAllocatorFree));
            OrtAllocatorGetInfo = (DOrtAllocatorGetInfo)Marshal.GetDelegateForFunctionPointer(api_.AllocatorGetInfo, typeof(DOrtAllocatorGetInfo));
            OrtAddFreeDimensionOverride = (DOrtAddFreeDimensionOverride)Marshal.GetDelegateForFunctionPointer(api_.AddFreeDimensionOverride, typeof(DOrtAddFreeDimensionOverride));

            OrtGetValue = (DOrtGetValue)Marshal.GetDelegateForFunctionPointer(api_.GetValue, typeof(DOrtGetValue));
            OrtGetValueType = (DOrtGetValueType)Marshal.GetDelegateForFunctionPointer(api_.GetValueType, typeof(DOrtGetValueType));
            OrtGetOnnxTypeFromTypeInfo = (DOrtGetOnnxTypeFromTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.GetOnnxTypeFromTypeInfo, typeof(DOrtGetOnnxTypeFromTypeInfo));
            OrtGetValueCount = (DOrtGetValueCount)Marshal.GetDelegateForFunctionPointer(api_.GetValueCount, typeof(DOrtGetValueCount));
            OrtGetTypeInfo = (DOrtGetTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.GetTypeInfo, typeof(DOrtGetTypeInfo));
            OrtCreateTensorAsOrtValue = (DOrtCreateTensorAsOrtValue)Marshal.GetDelegateForFunctionPointer(api_.CreateTensorAsOrtValue, typeof(DOrtCreateTensorAsOrtValue));
            OrtCreateTensorWithDataAsOrtValue = (DOrtCreateTensorWithDataAsOrtValue)Marshal.GetDelegateForFunctionPointer(api_.CreateTensorWithDataAsOrtValue, typeof(DOrtCreateTensorWithDataAsOrtValue));
            OrtGetTensorMutableData = (DOrtGetTensorMutableData)Marshal.GetDelegateForFunctionPointer(api_.GetTensorMutableData, typeof(DOrtGetTensorMutableData));
            OrtFillStringTensor = (DOrtFillStringTensor)Marshal.GetDelegateForFunctionPointer(api_.FillStringTensor, typeof(DOrtFillStringTensor));
            OrtGetStringTensorContent = (DOrtGetStringTensorContent)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorContent, typeof(DOrtGetStringTensorContent));
            OrtGetStringTensorDataLength = (DOrtGetStringTensorDataLength)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorDataLength, typeof(DOrtGetStringTensorDataLength));
            OrtCastTypeInfoToTensorInfo = (DOrtCastTypeInfoToTensorInfo)Marshal.GetDelegateForFunctionPointer(api_.CastTypeInfoToTensorInfo, typeof(DOrtCastTypeInfoToTensorInfo));
            OrtGetTensorTypeAndShape = (DOrtGetTensorTypeAndShape)Marshal.GetDelegateForFunctionPointer(api_.GetTensorTypeAndShape, typeof(DOrtGetTensorTypeAndShape));
            OrtReleaseTensorTypeAndShapeInfo = (DOrtReleaseTensorTypeAndShapeInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTensorTypeAndShapeInfo, typeof(DOrtReleaseTensorTypeAndShapeInfo));
            OrtGetTensorElementType = (DOrtGetTensorElementType)Marshal.GetDelegateForFunctionPointer(api_.GetTensorElementType, typeof(DOrtGetTensorElementType));
            OrtGetDimensionsCount = (DOrtGetDimensionsCount)Marshal.GetDelegateForFunctionPointer(api_.GetDimensionsCount, typeof(DOrtGetDimensionsCount));
            OrtGetDimensions = (DOrtGetDimensions)Marshal.GetDelegateForFunctionPointer(api_.GetDimensions, typeof(DOrtGetDimensions));
            OrtGetSymbolicDimensions = (DOrtGetSymbolicDimensions)Marshal.GetDelegateForFunctionPointer(api_.GetSymbolicDimensions, typeof(DOrtGetSymbolicDimensions));
            OrtGetTensorShapeElementCount = (DOrtGetTensorShapeElementCount)Marshal.GetDelegateForFunctionPointer(api_.GetTensorShapeElementCount, typeof(DOrtGetTensorShapeElementCount));
            OrtReleaseValue = (DOrtReleaseValue)Marshal.GetDelegateForFunctionPointer(api_.ReleaseValue, typeof(DOrtReleaseValue));
        }

        [DllImport(nativeLib, CharSet = charSet)]
        public static extern ref OrtApiBase OrtGetApiBase();

        #region Runtime/Environment API

        public delegate IntPtr /* OrtStatus* */DOrtCreateEnv(LogLevel default_warning_level, string logId, out IntPtr /*(OrtEnv*)*/ env);
        public static DOrtCreateEnv OrtCreateEnv;

        // OrtReleaseEnv should not be used
        public delegate void DOrtReleaseEnv(IntPtr /*(OrtEnv*)*/ env);
        public static DOrtReleaseEnv OrtReleaseEnv;

        #endregion Runtime/Environment API

        #region Status API
        public delegate ErrorCode DOrtGetErrorCode(IntPtr /*(OrtStatus*)*/status);
        public static DOrtGetErrorCode OrtGetErrorCode;

        // returns char*, need to convert to string by the caller.
        // does not free the underlying OrtStatus*
        public delegate IntPtr /* char* */DOrtGetErrorMessage(IntPtr /* (OrtStatus*) */status);
        public static DOrtGetErrorMessage OrtGetErrorMessage;

        public delegate void DOrtReleaseStatus(IntPtr /*(OrtStatus*)*/ statusPtr);
        public static DOrtReleaseStatus OrtReleaseStatus;

        #endregion Status API

        #region InferenceSession API

        public delegate IntPtr /* OrtStatus* */DOrtCreateSession(
                                                IntPtr /* (OrtEnv*) */ environment,
                                                //[MarshalAs(UnmanagedType.LPStr)]string modelPath
                                                byte[] modelPath,
                                                IntPtr /* (OrtSessionOptions*) */sessopnOptions,
                                                out IntPtr /**/ session);
        public static DOrtCreateSession OrtCreateSession;

        public delegate IntPtr /* OrtStatus* */DOrtCreateSessionFromArray(
                                                IntPtr /* (OrtEnv*) */ environment,
                                                byte[] modelData,
                                                UIntPtr modelSize,
                                                IntPtr /* (OrtSessionOptions*) */sessionOptions,
                                                out IntPtr /**/ session);
        public static DOrtCreateSessionFromArray OrtCreateSessionFromArray;

        public delegate IntPtr /*(ONNStatus*)*/ DOrtRun(
                                                IntPtr /*(OrtSession*)*/ session,
                                                IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                string[] inputNames,
                                                IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                UIntPtr inputCount,
                                                string[] outputNames,
                                                UIntPtr outputCount,
                                                IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                );
        public static DOrtRun OrtRun;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetInputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);
        public static DOrtSessionGetInputCount OrtSessionGetInputCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetOutputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);
        public static DOrtSessionGetOutputCount OrtSessionGetOutputCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetOverridableInitializerCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);
        public static DOrtSessionGetOverridableInitializerCount OrtSessionGetOverridableInitializerCount;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetInputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);
        public static DOrtSessionGetInputName OrtSessionGetInputName;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOutputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);
        public static DOrtSessionGetOutputName OrtSessionGetOutputName;

        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOverridableInitializerName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);
        public static DOrtSessionGetOverridableInitializerName OrtSessionGetOverridableInitializerName;

        // release the typeinfo using OrtReleaseTypeInfo
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetInputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /*(struct OrtTypeInfo**)*/ typeInfo);
        public static DOrtSessionGetInputTypeInfo OrtSessionGetInputTypeInfo;

        // release the typeinfo using OrtReleaseTypeInfo
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOutputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);
        public static DOrtSessionGetOutputTypeInfo OrtSessionGetOutputTypeInfo;

        // release the typeinfo using OrtReleaseTypeInfo
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOverridableInitializerTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);
        public static DOrtSessionGetOverridableInitializerTypeInfo OrtSessionGetOverridableInitializerTypeInfo;


        public delegate void DOrtReleaseTypeInfo(IntPtr /*(OrtTypeInfo*)*/session);
        public static DOrtReleaseTypeInfo OrtReleaseTypeInfo;

        public delegate void DOrtReleaseSession(IntPtr /*(OrtSession*)*/session);
        public static DOrtReleaseSession OrtReleaseSession;

        #endregion InferenceSession API

        #region SessionOptions API

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateSessionOptions(out IntPtr /*(OrtSessionOptions**)*/ sessionOptions);
        public static DOrtCreateSessionOptions OrtCreateSessionOptions;

        public delegate void DOrtReleaseSessionOptions(IntPtr /*(OrtSessionOptions*)*/session);
        public static DOrtReleaseSessionOptions OrtReleaseSessionOptions;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCloneSessionOptions(IntPtr /*(OrtSessionOptions*)*/ sessionOptions, out IntPtr /*(OrtSessionOptions**)*/ output);
        public static DOrtCloneSessionOptions OrtCloneSessionOptions;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionExecutionMode(IntPtr /*(OrtSessionOptions*)*/ options,
        ExecutionMode execution_mode);
        public static DOrtSetSessionExecutionMode OrtSetSessionExecutionMode;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetOptimizedModelFilePath(IntPtr /* OrtSessionOptions* */ options, byte[] optimizedModelFilepath);
        public static DOrtSetOptimizedModelFilePath OrtSetOptimizedModelFilePath;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableProfiling(IntPtr /* OrtSessionOptions* */ options, byte[] profilePathPrefix);
        public static DOrtEnableProfiling OrtEnableProfiling;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableProfiling(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableProfiling OrtDisableProfiling;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableMemPattern(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtEnableMemPattern OrtEnableMemPattern;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableMemPattern(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableMemPattern OrtDisableMemPattern;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtEnableCpuMemArena OrtEnableCpuMemArena;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableCpuMemArena OrtDisableCpuMemArena;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogId(IntPtr /* OrtSessionOptions* */ options, string logId);
        public static DOrtSetSessionLogId OrtSetSessionLogId;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogVerbosityLevel(IntPtr /* OrtSessionOptions* */ options, LogLevel sessionLogVerbosityLevel);
        public static DOrtSetSessionLogVerbosityLevel OrtSetSessionLogVerbosityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogSeverityLevel(IntPtr /* OrtSessionOptions* */ options, LogLevel sessionLogSeverityLevel);
        public static DOrtSetSessionLogSeverityLevel OrtSetSessionLogSeverityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetIntraOpNumThreads(IntPtr /* OrtSessionOptions* */ options, int intraOpNumThreads);
        public static DOrtSetIntraOpNumThreads OrtSetIntraOpNumThreads;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetInterOpNumThreads(IntPtr /* OrtSessionOptions* */ options, int interOpNumThreads);
        public static DOrtSetInterOpNumThreads OrtSetInterOpNumThreads;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionGraphOptimizationLevel(IntPtr /* OrtSessionOptions* */ options, GraphOptimizationLevel graphOptimizationLevel);
        public static DOrtSetSessionGraphOptimizationLevel OrtSetSessionGraphOptimizationLevel;

        ///**
        //  * The order of invocation indicates the preference order as well. In other words call this method
        //  * on your most preferred execution provider first followed by the less preferred ones.
        //  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
        //  */
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CPU(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);

#if USE_DNNL
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Dnnl(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);
#endif

#if USE_CUDA
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CUDA(IntPtr /*(OrtSessionOptions*) */ options, int device_id);
#endif

#if USE_NGRAPH
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_NGraph(IntPtr /*(OrtSessionOptions*) */ options, string /*(const char*)*/ ng_backend_type);
#endif

#if USE_OPENVINO
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_OpenVINO(
                                                    IntPtr /*(OrtSessionOptions*)*/ options, string /*(const char*)*/ device_id);
#endif

#if USE_TENSORRT
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Tensorrt(IntPtr /*(OrtSessionOptions*)*/ options, int device_id);
#endif

#if USE_NNAPI
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Nnapi(IntPtr /*(OrtSessionOptions*)*/ options);
#endif

#if USE_NUPHAR
        [DllImport(nativeLib, CharSet = charSet)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Nuphar(IntPtr /*(OrtSessionOptions*) */ options, int allow_unaligned_buffers, string settings);
#endif
        //[DllImport(nativeLib, CharSet = charSet)]
        //public static extern void OrtAddCustomOp(IntPtr /*(OrtSessionOptions*)*/ options, string custom_op_path);

        public delegate IntPtr /*(OrtStatus*)*/DOrtAddFreeDimensionOverride(IntPtr /*(OrtSessionOptions*) */ options, string /*(const char*)*/ symbolic_dim, int dim_override);
        public static DOrtAddFreeDimensionOverride OrtAddFreeDimensionOverride;

        public delegate IntPtr /*(OrtStatus*)*/DOrtRegisterCustomOpsLibrary(IntPtr /*(OrtSessionOptions*) */ options, string /*(const char*)*/ library_path, out IntPtr /* (void**) */ library_handle);
        public static DOrtRegisterCustomOpsLibrary OrtRegisterCustomOpsLibrary;

        #endregion

        #region RunOptions API
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateRunOptions(out IntPtr /* OrtRunOptions** */ runOptions);
        public static DOrtCreateRunOptions OrtCreateRunOptions;

        public delegate void DOrtReleaseRunOptions(IntPtr /*(OrtRunOptions*)*/options);
        public static DOrtReleaseRunOptions OrtReleaseRunOptions;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, LogLevel value);
        public static DOrtRunOptionsSetRunLogVerbosityLevel OrtRunOptionsSetRunLogVerbosityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunTag(IntPtr /* OrtRunOptions* */ options, string /* const char* */ runTag);
        public static DOrtRunOptionsSetRunTag OrtRunOptionsSetRunTag;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, out LogLevel verbosityLevel);
        public static DOrtRunOptionsGetRunLogVerbosityLevel OrtRunOptionsGetRunLogVerbosityLevel;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunTag(IntPtr /* const OrtRunOptions* */options, out IntPtr /* const char** */ runtag);
        public static DOrtRunOptionsGetRunTag OrtRunOptionsGetRunTag;

        // Set a flag so that any running OrtRun* calls that are using this instance of OrtRunOptions
        // will exit as soon as possible if the flag is true.
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetTerminate(IntPtr /* OrtRunOptions* */ options);
        public static DOrtRunOptionsSetTerminate OrtRunOptionsSetTerminate;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsUnsetTerminate(IntPtr /* OrtRunOptions* */ options);
        public static DOrtRunOptionsUnsetTerminate OrtRunOptionsUnsetTerminate;



        #endregion

        #region Allocator/MemoryInfo API

        //TODO: consider exposing them publicly, when allocator API is exposed
        public enum AllocatorType
        {
            DeviceAllocator = 0,
            ArenaAllocator = 1
        }

        //TODO: consider exposing them publicly when allocator API is exposed
        public enum MemoryType
        {
            CpuInput = -2,                      // Any CPU memory used by non-CPU execution provider
            CpuOutput = -1,                     // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
            Cpu = CpuOutput,                    // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
            Default = 0,                        // the default allocator for execution provider
        }


        public delegate IntPtr /* (OrtStatus*)*/ DOrtCreateMemoryInfo(
                                                            IntPtr /*(const char*) */name,
                                                            AllocatorType allocatorType,
                                                            int identifier,
                                                            MemoryType memType,
                                                            out IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo    // memory ownership transfered to caller
                                                       );
        public static DOrtCreateMemoryInfo OrtCreateMemoryInfo;

        //ORT_API_STATUS(OrtCreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1, _Out_ OrtMemoryInfo** out)
        public delegate IntPtr /* (OrtStatus*)*/ DOrtCreateCpuMemoryInfo(
                                                            AllocatorType allocatorType,
                                                            MemoryType memoryType,
                                                            out IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo
                                                        );
        public static DOrtCreateCpuMemoryInfo OrtCreateCpuMemoryInfo;

        public delegate void DOrtReleaseMemoryInfo(IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo);
        public static DOrtReleaseMemoryInfo OrtReleaseMemoryInfo;

        public delegate IntPtr /*(OrtStatus*)*/DOrtGetAllocatorWithDefaultOptions(out IntPtr /*(OrtAllocator**)*/ allocator);
        public static DOrtGetAllocatorWithDefaultOptions OrtGetAllocatorWithDefaultOptions;

        /// <summary>
        /// Release any object allocated by an allocator
        /// </summary>
        /// <param name="allocator"></param>
        /// <param name="memory"></param>
        public delegate IntPtr /*(OrtStatus*)*/DOrtAllocatorFree(IntPtr allocator, IntPtr memory);
        public static DOrtAllocatorFree OrtAllocatorFree;

        public delegate IntPtr /*(OrtStatus*)*/DOrtAllocatorGetInfo(IntPtr /*(const OrtAllocator*)*/ ptr, out IntPtr /*(const struct OrtMemoryInfo**)*/info);
        public static DOrtAllocatorGetInfo OrtAllocatorGetInfo;

        #endregion Allocator/MemoryInfo API

        #region Tensor/OnnxValue API

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValue(IntPtr /*(OrtValue*)*/ value,
                                                                 int index,
                                                                 IntPtr /*(OrtAllocator*)*/ allocator,
                                                                 out IntPtr /*(OrtValue**)*/ outputValue);
        public static DOrtGetValue OrtGetValue;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValueType(IntPtr /*(OrtValue*)*/ value, IntPtr /*(OnnxValueType*)*/ onnxtype);
        public static DOrtGetValueType OrtGetValueType;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetOnnxTypeFromTypeInfo(IntPtr /*(OrtTypeInfo*)*/ typeinfo, IntPtr /*(OnnxValueType*)*/ onnxtype);
        public static DOrtGetOnnxTypeFromTypeInfo OrtGetOnnxTypeFromTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValueCount(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(size_t*)*/ count);
        public static DOrtGetValueCount OrtGetValueCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTypeInfo(IntPtr /*(OrtValue*)*/ value, IntPtr /*(OrtValue**)*/ typeInfo);
        public static DOrtGetTypeInfo OrtGetTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateTensorAsOrtValue(
                        IntPtr /*_Inout_ OrtAllocator* */ allocator,
                        long[] /*_In_ const int64_t* */ shape,
                        UIntPtr /*size_t*/ shape_len,
                        TensorElementType type,
                        out IntPtr /* OrtValue** */ outputValue);
        public static DOrtCreateTensorAsOrtValue OrtCreateTensorAsOrtValue;

        public delegate IntPtr /* OrtStatus */ DOrtCreateTensorWithDataAsOrtValue(
                                                        IntPtr /* (const OrtMemoryInfo*) */ allocatorInfo,
                                                        IntPtr /* (void*) */dataBufferHandle,
                                                        UIntPtr dataLength,
                                                        long[] shape,
                                                        UIntPtr shapeLength,
                                                        TensorElementType type,
                                                        out IntPtr /* OrtValue** */ outputValue);
        public static DOrtCreateTensorWithDataAsOrtValue OrtCreateTensorWithDataAsOrtValue;

        /// This function doesn't work with string tensor
        /// this is a no-copy method whose pointer is only valid until the backing OrtValue* is free'd.
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorMutableData(IntPtr /*(OrtValue*)*/ value, out IntPtr /* (void**)*/ dataBufferHandle);
        public static DOrtGetTensorMutableData OrtGetTensorMutableData;

        /// \param value A tensor created from OrtCreateTensor... function.
        /// \param len total data length, not including the trailing '\0' chars.
        public delegate IntPtr /*(OrtStatus*)*/ DOrtFillStringTensor(
                                                        IntPtr /* OrtValue */ value,
                                                        IntPtr[] /* const char* const* */s,
                                                        UIntPtr /* size_t */ s_len);
        public static DOrtFillStringTensor OrtFillStringTensor;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorContent(
                                                        IntPtr /*(OrtValue*)*/ value,
                                                        IntPtr /*(void*)*/  dst_buffer,
                                                        UIntPtr dst_buffer_len,
                                                        IntPtr offsets,
                                                        UIntPtr offsets_len);
        public static DOrtGetStringTensorContent OrtGetStringTensorContent;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorDataLength(IntPtr /*(OrtValue*)*/ value,
                                                        out UIntPtr /*(size_t*)*/ len);
        public static DOrtGetStringTensorDataLength OrtGetStringTensorDataLength;

        public delegate IntPtr /*(OrtStatus*)*/
                                DOrtCastTypeInfoToTensorInfo(IntPtr /*(struct OrtTypeInfo*)*/ typeInfo, out IntPtr /*(const struct OrtTensorTypeAndShapeInfo**)*/ typeAndShapeInfo);
        public static DOrtCastTypeInfoToTensorInfo OrtCastTypeInfoToTensorInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorTypeAndShape(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);
        public static DOrtGetTensorTypeAndShape OrtGetTensorTypeAndShape;


        public delegate void DOrtReleaseTensorTypeAndShapeInfo(IntPtr /*(OrtTensorTypeAndShapeInfo*)*/ value);
        public static DOrtReleaseTensorTypeAndShapeInfo OrtReleaseTensorTypeAndShapeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorElementType(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, IntPtr /*(TensorElementType*)*/ output);
        public static DOrtGetTensorElementType OrtGetTensorElementType;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetDimensionsCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, out UIntPtr output);
        public static DOrtGetDimensionsCount OrtGetDimensionsCount;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetDimensions(
                            IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo,
                            long[] dim_values,
                            UIntPtr dim_values_length);
        public static DOrtGetDimensions OrtGetDimensions;

        /**
        * Get the symbolic dimension names for dimensions with a value of -1. 
        * Order and number of entries is the same as values returned by GetDimensions. 
        * The name may be empty for an unnamed symbolic dimension.
        * e.g. 
        * If OrtGetDimensions returns [-1, -1, 2], OrtGetSymbolicDimensions would return an array with 3 entries.
        * If the values returned were ['batch', '', ''] it would indicate that
        *  - the first dimension was a named symbolic dimension (-1 dim value and name in symbolic dimensions), 
        *  - the second dimension was an unnamed symbolic dimension (-1 dim value and empty string), 
        *  - the entry for the third dimension should be ignored as it is not a symbolic dimension (dim value >= 0).
        */
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetSymbolicDimensions(
                    IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo,
                    IntPtr[] dim_params, /* const char* values, converted to string by caller */
                    UIntPtr dim_params_length);
        public static DOrtGetSymbolicDimensions OrtGetSymbolicDimensions;

        /**
         * How many elements does this tensor have.
         * May return a negative value
         * e.g.
         * [] -> 1
         * [1,3,4] -> 12
         * [2,0,4] -> 0
         * [-1,3,4] -> -1
         */
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorShapeElementCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, IntPtr /*(long*)*/ output);
        public static DOrtGetTensorShapeElementCount OrtGetTensorShapeElementCount;

        public delegate void DOrtReleaseValue(IntPtr /*(OrtValue*)*/ value);
        public static DOrtReleaseValue OrtReleaseValue;

        #endregion

        public static byte[] GetPlatformSerializedString(string str)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return System.Text.Encoding.Unicode.GetBytes(str + Char.MinValue);
            else
                return System.Text.Encoding.UTF8.GetBytes(str + Char.MinValue);
        }
    } //class NativeMethods
} //namespace
