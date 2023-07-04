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

    // NOTE: The order of the APIs in this struct should match exactly that in
    // OrtApi ort_api_1_to_<latest_version> (onnxruntime/core/session/onnxruntime_c_api.cc)
    // If syncing your new C API, any other C APIs before yours also need to be synced here if haven't
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
        public IntPtr GetDenotationFromTypeInfo;
        public IntPtr CastTypeInfoToMapTypeInfo;
        public IntPtr CastTypeInfoToSequenceTypeInfo;
        public IntPtr GetMapKeyType;
        public IntPtr GetMapValueType;
        public IntPtr GetSequenceElementType;
        public IntPtr ReleaseMapTypeInfo;
        public IntPtr ReleaseSequenceTypeInfo;
        public IntPtr SessionEndProfiling;

        public IntPtr SessionGetModelMetadata;
        public IntPtr ModelMetadataGetProducerName;
        public IntPtr ModelMetadataGetGraphName;
        public IntPtr ModelMetadataGetDomain;
        public IntPtr ModelMetadataGetDescription;
        public IntPtr ModelMetadataLookupCustomMetadataMap;
        public IntPtr ModelMetadataGetVersion;
        public IntPtr ReleaseModelMetadata;

        public IntPtr CreateEnvWithGlobalThreadPools;
        public IntPtr DisablePerSessionThreads;
        public IntPtr CreateThreadingOptions;
        public IntPtr ReleaseThreadingOptions;
        public IntPtr ModelMetadataGetCustomMetadataMapKeys;
        public IntPtr AddFreeDimensionOverrideByName;

        public IntPtr GetAvailableProviders;
        public IntPtr ReleaseAvailableProviders;
        public IntPtr GetStringTensorElementLength;
        public IntPtr GetStringTensorElement;
        public IntPtr FillStringTensorElement;
        public IntPtr AddSessionConfigEntry;

        public IntPtr CreateAllocator;
        public IntPtr ReleaseAllocator;
        public IntPtr RunWithBinding;
        public IntPtr CreateIoBinding;
        public IntPtr ReleaseIoBinding;
        public IntPtr BindInput;
        public IntPtr BindOutput;
        public IntPtr BindOutputToDevice;
        public IntPtr GetBoundOutputNames;
        public IntPtr GetBoundOutputValues;
        public IntPtr ClearBoundInputs;
        public IntPtr ClearBoundOutputs;
        public IntPtr TensorAt;
        public IntPtr CreateAndRegisterAllocator;
        public IntPtr SetLanguageProjection;
        public IntPtr SessionGetProfilingStartTimeNs;
        public IntPtr SetGlobalIntraOpNumThreads;
        public IntPtr SetGlobalInterOpNumThreads;
        public IntPtr SetGlobalSpinControl;
        public IntPtr AddInitializer;
        public IntPtr CreateEnvWithCustomLoggerAndGlobalThreadPools;
        public IntPtr SessionOptionsAppendExecutionProvider_CUDA;
        public IntPtr SessionOptionsAppendExecutionProvider_ROCM;
        public IntPtr SessionOptionsAppendExecutionProvider_OpenVINO;
        public IntPtr SetGlobalDenormalAsZero;
        public IntPtr CreateArenaCfg;
        public IntPtr ReleaseArenaCfg;
        public IntPtr ModelMetadataGetGraphDescription;
        public IntPtr SessionOptionsAppendExecutionProvider_TensorRT;
        public IntPtr SetCurrentGpuDeviceId;
        public IntPtr GetCurrentGpuDeviceId;
        public IntPtr KernelInfoGetAttributeArray_float;
        public IntPtr KernelInfoGetAttributeArray_int64;
        public IntPtr CreateArenaCfgV2;
        public IntPtr AddRunConfigEntry;
        public IntPtr CreatePrepackedWeightsContainer;
        public IntPtr ReleasePrepackedWeightsContainer;
        public IntPtr CreateSessionWithPrepackedWeightsContainer;
        public IntPtr CreateSessionFromArrayWithPrepackedWeightsContainer;
        public IntPtr SessionOptionsAppendExecutionProvider_TensorRT_V2;
        public IntPtr CreateTensorRTProviderOptions;
        public IntPtr UpdateTensorRTProviderOptions;
        public IntPtr GetTensorRTProviderOptionsAsString;
        public IntPtr ReleaseTensorRTProviderOptions;
        public IntPtr EnableOrtCustomOps;
        public IntPtr RegisterAllocator;
        public IntPtr UnregisterAllocator;
        public IntPtr IsSparseTensor;
        public IntPtr CreateSparseTensorAsOrtValue;
        public IntPtr FillSparseTensorCoo;
        public IntPtr FillSparseTensorCsr;
        public IntPtr FillSparseTensorBlockSparse;
        public IntPtr CreateSparseTensorWithValuesAsOrtValue;
        public IntPtr UseCooIndices;
        public IntPtr UseCsrIndices;
        public IntPtr UseBlockSparseIndices;
        public IntPtr GetSparseTensorFormat;
        public IntPtr GetSparseTensorValuesTypeAndShape;
        public IntPtr GetSparseTensorValues;
        public IntPtr GetSparseTensorIndicesTypeShape;
        public IntPtr GetSparseTensorIndices;
        public IntPtr HasValue;
        public IntPtr KernelContext_GetGPUComputeStream;
        public IntPtr GetTensorMemoryInfo;
        public IntPtr GetExecutionProviderApi;
        public IntPtr SessionOptionsSetCustomCreateThreadFn;
        public IntPtr SessionOptionsSetCustomThreadCreationOptions;
        public IntPtr SessionOptionsSetCustomJoinThreadFn;
        public IntPtr SetGlobalCustomCreateThreadFn;
        public IntPtr SetGlobalCustomThreadCreationOptions;
        public IntPtr SetGlobalCustomJoinThreadFn;
        public IntPtr SynchronizeBoundInputs;
        public IntPtr SynchronizeBoundOutputs;
        public IntPtr SessionOptionsAppendExecutionProvider_CUDA_V2;
        public IntPtr CreateCUDAProviderOptions;
        public IntPtr UpdateCUDAProviderOptions;
        public IntPtr GetCUDAProviderOptionsAsString;
        public IntPtr ReleaseCUDAProviderOptions;
        public IntPtr SessionOptionsAppendExecutionProvider_MIGraphX;
        public IntPtr AddExternalInitializers;
        public IntPtr CreateOpAttr;
        public IntPtr ReleaseOpAttr;
        public IntPtr CreateOp;
        public IntPtr InvokeOp;
        public IntPtr ReleaseOp;
        public IntPtr SessionOptionsAppendExecutionProvider;
        public IntPtr CopyKernelInfo;
        public IntPtr ReleaseKernelInfo;

        public IntPtr GetTrainingApi;
        public IntPtr SessionOptionsAppendExecutionProvider_CANN;
        public IntPtr CreateCANNProviderOptions;
        public IntPtr UpdateCANNProviderOptions;
        public IntPtr GetCANNProviderOptionsAsString;
        public IntPtr ReleaseCANNProviderOptions;
        public IntPtr MemoryInfoGetDeviceType;
        public IntPtr UpdateEnvWithCustomLogLevel;
        public IntPtr SetGlobalIntraOpThreadAffinity;
        public IntPtr RegisterCustomOpsLibrary_V2;
        public IntPtr RegisterCustomOpsUsingFunction;
        public IntPtr KernelInfo_GetInputCount;
        public IntPtr KernelInfo_GetOutputCount;
        public IntPtr KernelInfo_GetInputName;
        public IntPtr KernelInfo_GetOutputName;
        public IntPtr KernelInfo_GetInputTypeInfo;
        public IntPtr KernelInfo_GetOutputTypeInfo;
        public IntPtr KernelInfoGetAttribute_tensor;
        public IntPtr HasSessionConfigEntry;
        public IntPtr GetSessionConfigEntry;
        public IntPtr SessionOptionsAppendExecutionProvider_Dnnl;
        public IntPtr CreateDnnlProviderOptions;
        public IntPtr UpdateDnnlProviderOptions;
        public IntPtr GetDnnlProviderOptionsAsString;
        public IntPtr ReleaseDnnlProviderOptions;
        public IntPtr KernelInfo_GetNodeName;
        public IntPtr KernelInfo_GetLogger;
        public IntPtr KernelContext_GetLogger;
        public IntPtr Logger_LogMessage;
        public IntPtr Logger_GetLoggingSeverityLevel;
        public IntPtr KernelInfoGetConstantInput_tensor;
        public IntPtr CastTypeInfoToOptionalTypeInfo;
        public IntPtr GetOptionalContainedTypeInfo;
        public IntPtr GetResizedStringTensorElementBuffer;
        public IntPtr KernelContext_GetAllocator;
        public IntPtr GetBuildInfoString;
        public IntPtr CreateROCMProviderOptions;
        public IntPtr UpdateROCMProviderOptions;
        public IntPtr GetROCMProviderOptionsAsString;
        public IntPtr ReleaseROCMProviderOptions;
    }

    internal static class NativeMethods
    {
        static OrtApi api_;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate ref OrtApi DOrtGetApi(UInt32 version);

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr DOrtGetVersionString();

        public static DOrtGetVersionString OrtGetVersionString;

        static NativeMethods()
        {
            DOrtGetApi OrtGetApi = (DOrtGetApi)Marshal.GetDelegateForFunctionPointer(OrtGetApiBase().GetApi, typeof(DOrtGetApi));

            // TODO: Make this save the pointer, and not copy the whole structure across
            api_ = (OrtApi)OrtGetApi(14 /*ORT_API_VERSION*/);
            OrtGetVersionString = (DOrtGetVersionString)Marshal.GetDelegateForFunctionPointer(OrtGetApiBase().GetVersionString, typeof(DOrtGetVersionString));

            OrtCreateEnv = (DOrtCreateEnv)Marshal.GetDelegateForFunctionPointer(api_.CreateEnv, typeof(DOrtCreateEnv));
            OrtCreateEnvWithCustomLogger = (DOrtCreateEnvWithCustomLogger)Marshal.GetDelegateForFunctionPointer(api_.CreateEnvWithCustomLogger, typeof(DOrtCreateEnvWithCustomLogger));
            OrtCreateEnvWithGlobalThreadPools = (DOrtCreateEnvWithGlobalThreadPools)Marshal.GetDelegateForFunctionPointer(api_.CreateEnvWithGlobalThreadPools, typeof(DOrtCreateEnvWithGlobalThreadPools));
            OrtCreateEnvWithCustomLoggerAndGlobalThreadPools = (DOrtCreateEnvWithCustomLoggerAndGlobalThreadPools)Marshal.GetDelegateForFunctionPointer(api_.CreateEnvWithCustomLoggerAndGlobalThreadPools, typeof(DOrtCreateEnvWithCustomLoggerAndGlobalThreadPools));
            OrtReleaseEnv = (DOrtReleaseEnv)Marshal.GetDelegateForFunctionPointer(api_.ReleaseEnv, typeof(DOrtReleaseEnv));
            OrtEnableTelemetryEvents = (DOrtEnableTelemetryEvents)Marshal.GetDelegateForFunctionPointer(api_.EnableTelemetryEvents, typeof(DOrtEnableTelemetryEvents));
            OrtDisableTelemetryEvents = (DOrtDisableTelemetryEvents)Marshal.GetDelegateForFunctionPointer(api_.DisableTelemetryEvents, typeof(DOrtDisableTelemetryEvents));

            OrtGetErrorCode = (DOrtGetErrorCode)Marshal.GetDelegateForFunctionPointer(api_.GetErrorCode, typeof(DOrtGetErrorCode));
            OrtGetErrorMessage = (DOrtGetErrorMessage)Marshal.GetDelegateForFunctionPointer(api_.GetErrorMessage, typeof(DOrtGetErrorMessage));
            OrtReleaseStatus = (DOrtReleaseStatus)Marshal.GetDelegateForFunctionPointer(api_.ReleaseStatus, typeof(DOrtReleaseStatus));

            OrtCreateSession = (DOrtCreateSession)Marshal.GetDelegateForFunctionPointer(api_.CreateSession, typeof(DOrtCreateSession));
            OrtCreateSessionWithPrepackedWeightsContainer =
                (DOrtCreateSessionWithPrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionWithPrepackedWeightsContainer, typeof(DOrtCreateSessionWithPrepackedWeightsContainer));
            OrtCreateSessionFromArray = (DOrtCreateSessionFromArray)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionFromArray, typeof(DOrtCreateSessionFromArray));
            OrtCreateSessionFromArrayWithPrepackedWeightsContainer =
                (DOrtCreateSessionFromArrayWithPrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.CreateSessionFromArrayWithPrepackedWeightsContainer, typeof(DOrtCreateSessionFromArrayWithPrepackedWeightsContainer));
            OrtRun = (DOrtRun)Marshal.GetDelegateForFunctionPointer(api_.Run, typeof(DOrtRun));
            OrtRunWithBinding = (DOrtRunWithBinding)Marshal.GetDelegateForFunctionPointer(api_.RunWithBinding, typeof(DOrtRunWithBinding));
            OrtSessionGetInputCount = (DOrtSessionGetInputCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputCount, typeof(DOrtSessionGetInputCount));
            OrtSessionGetOutputCount = (DOrtSessionGetOutputCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputCount, typeof(DOrtSessionGetOutputCount));
            OrtSessionGetOverridableInitializerCount = (DOrtSessionGetOverridableInitializerCount)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerCount, typeof(DOrtSessionGetOverridableInitializerCount));

            OrtSessionGetInputName = (DOrtSessionGetInputName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputName, typeof(DOrtSessionGetInputName));
            OrtSessionGetOutputName = (DOrtSessionGetOutputName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputName, typeof(DOrtSessionGetOutputName));
            OrtSessionEndProfiling = (DOrtSessionEndProfiling)Marshal.GetDelegateForFunctionPointer(api_.SessionEndProfiling, typeof(DOrtSessionEndProfiling));
            OrtSessionGetOverridableInitializerName = (DOrtSessionGetOverridableInitializerName)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerName, typeof(DOrtSessionGetOverridableInitializerName));
            OrtSessionGetInputTypeInfo = (DOrtSessionGetInputTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetInputTypeInfo, typeof(DOrtSessionGetInputTypeInfo));
            OrtSessionGetOutputTypeInfo = (DOrtSessionGetOutputTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOutputTypeInfo, typeof(DOrtSessionGetOutputTypeInfo));
            OrtSessionGetOverridableInitializerTypeInfo = (DOrtSessionGetOverridableInitializerTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.SessionGetOverridableInitializerTypeInfo, typeof(DOrtSessionGetOverridableInitializerTypeInfo));
            OrtReleaseTypeInfo = (DOrtReleaseTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTypeInfo, typeof(DOrtReleaseTypeInfo));
            OrtReleaseSession = (DOrtReleaseSession)Marshal.GetDelegateForFunctionPointer(api_.ReleaseSession, typeof(DOrtReleaseSession));
            OrtSessionGetProfilingStartTimeNs = (DOrtSessionGetProfilingStartTimeNs)Marshal.GetDelegateForFunctionPointer(api_.SessionGetProfilingStartTimeNs, typeof(DOrtSessionGetProfilingStartTimeNs));

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
            OrtRegisterCustomOpsLibrary_V2 = (DOrtRegisterCustomOpsLibrary_V2)Marshal.GetDelegateForFunctionPointer(api_.RegisterCustomOpsLibrary_V2, typeof(DOrtRegisterCustomOpsLibrary_V2));
            OrtAddSessionConfigEntry = (DOrtAddSessionConfigEntry)Marshal.GetDelegateForFunctionPointer(api_.AddSessionConfigEntry, typeof(DOrtAddSessionConfigEntry));
            OrtAddInitializer = (DOrtAddInitializer)Marshal.GetDelegateForFunctionPointer(api_.AddInitializer, typeof(DOrtAddInitializer));
            SessionOptionsAppendExecutionProvider_TensorRT = (DSessionOptionsAppendExecutionProvider_TensorRT)Marshal.GetDelegateForFunctionPointer(
                                                             api_.SessionOptionsAppendExecutionProvider_TensorRT, typeof(DSessionOptionsAppendExecutionProvider_TensorRT));

            OrtCreateRunOptions = (DOrtCreateRunOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateRunOptions, typeof(DOrtCreateRunOptions));
            OrtReleaseRunOptions = (DOrtReleaseRunOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseRunOptions, typeof(DOrtReleaseRunOptions));
            OrtRunOptionsSetRunLogVerbosityLevel = (DOrtRunOptionsSetRunLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunLogVerbosityLevel, typeof(DOrtRunOptionsSetRunLogVerbosityLevel));
            OrtRunOptionsSetRunLogSeverityLevel = (DOrtRunOptionsSetRunLogSeverityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunLogSeverityLevel, typeof(DOrtRunOptionsSetRunLogSeverityLevel));
            OrtRunOptionsSetRunTag = (DOrtRunOptionsSetRunTag)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetRunTag, typeof(DOrtRunOptionsSetRunTag));
            OrtRunOptionsGetRunLogVerbosityLevel = (DOrtRunOptionsGetRunLogVerbosityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunLogVerbosityLevel, typeof(DOrtRunOptionsGetRunLogVerbosityLevel));
            OrtRunOptionsGetRunLogSeverityLevel = (DOrtRunOptionsGetRunLogSeverityLevel)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunLogSeverityLevel, typeof(DOrtRunOptionsGetRunLogSeverityLevel));
            OrtRunOptionsGetRunTag = (DOrtRunOptionsGetRunTag)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsGetRunTag, typeof(DOrtRunOptionsGetRunTag));
            OrtRunOptionsSetTerminate = (DOrtRunOptionsSetTerminate)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsSetTerminate, typeof(DOrtRunOptionsSetTerminate));
            OrtRunOptionsUnsetTerminate = (DOrtRunOptionsUnsetTerminate)Marshal.GetDelegateForFunctionPointer(api_.RunOptionsUnsetTerminate, typeof(DOrtRunOptionsUnsetTerminate));

            OrtCreateThreadingOptions = (DOrtCreateThreadingOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateThreadingOptions, typeof(DOrtCreateThreadingOptions));
            OrtReleaseThreadingOptions = (DOrtReleaseThreadingOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseThreadingOptions, typeof(DOrtReleaseThreadingOptions));
            OrtThreadingOptionsSetGlobalInterOpNumThreads = (DOrtThreadingOptionsSetGlobalInterOpNumThreads)Marshal.GetDelegateForFunctionPointer(api_.SetGlobalInterOpNumThreads, typeof(DOrtThreadingOptionsSetGlobalInterOpNumThreads));
            OrtThreadingOptionsSetGlobalIntraOpNumThreads = (DOrtThreadingOptionsSetGlobalIntraOpNumThreads)Marshal.GetDelegateForFunctionPointer(api_.SetGlobalIntraOpNumThreads, typeof(DOrtThreadingOptionsSetGlobalIntraOpNumThreads));
            OrtThreadingOptionsSetGlobalDenormalAsZero = (DOrtThreadingOptionsSetGlobalDenormalAsZero)Marshal.GetDelegateForFunctionPointer(api_.SetGlobalDenormalAsZero, typeof(DOrtThreadingOptionsSetGlobalDenormalAsZero));
            OrtThreadingOptionsSetGlobalSpinControl = (DOrtThreadingOptionsSetGlobalSpinControl)Marshal.GetDelegateForFunctionPointer(api_.SetGlobalSpinControl, typeof(DOrtThreadingOptionsSetGlobalSpinControl));
            OrtAddRunConfigEntry = (DOrtAddRunConfigEntry)Marshal.GetDelegateForFunctionPointer(api_.AddRunConfigEntry, typeof(DOrtAddRunConfigEntry));

            OrtCreateArenaCfg = (DOrtCreateArenaCfg)Marshal.GetDelegateForFunctionPointer(api_.CreateArenaCfg, typeof(DOrtCreateArenaCfg));
            OrtReleaseArenaCfg = (DOrtReleaseArenaCfg)Marshal.GetDelegateForFunctionPointer(api_.ReleaseArenaCfg, typeof(DOrtReleaseArenaCfg));
            OrtReleaseAllocator = (DOrtReleaseAllocator)Marshal.GetDelegateForFunctionPointer(api_.ReleaseAllocator, typeof(DOrtReleaseAllocator));
            OrtCreateMemoryInfo = (DOrtCreateMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CreateMemoryInfo, typeof(DOrtCreateMemoryInfo));
            OrtCreateCpuMemoryInfo = (DOrtCreateCpuMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CreateCpuMemoryInfo, typeof(DOrtCreateCpuMemoryInfo));
            OrtReleaseMemoryInfo = (DOrtReleaseMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseMemoryInfo, typeof(DOrtReleaseMemoryInfo));
            OrtCompareMemoryInfo = (DOrtCompareMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.CompareMemoryInfo, typeof(DOrtCompareMemoryInfo));
            OrtMemoryInfoGetName = (DOrtMemoryInfoGetName)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetName, typeof(DOrtMemoryInfoGetName));
            OrtMemoryInfoGetId = (DOrtMemoryInfoGetId)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetId, typeof(DOrtMemoryInfoGetId));
            OrtMemoryInfoGetMemType = (DOrtMemoryInfoGetMemType)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetMemType, typeof(DOrtMemoryInfoGetMemType));
            OrtMemoryInfoGetType = (DOrtMemoryInfoGetType)Marshal.GetDelegateForFunctionPointer(api_.MemoryInfoGetType, typeof(DOrtMemoryInfoGetType));
            OrtGetAllocatorWithDefaultOptions = (DOrtGetAllocatorWithDefaultOptions)Marshal.GetDelegateForFunctionPointer(api_.GetAllocatorWithDefaultOptions, typeof(DOrtGetAllocatorWithDefaultOptions));
            OrtCreateAllocator = (DOrtCreateAllocator)Marshal.GetDelegateForFunctionPointer(api_.CreateAllocator, typeof(DOrtCreateAllocator));
            OrtReleaseAllocator = (DOrtReleaseAllocator)Marshal.GetDelegateForFunctionPointer(api_.ReleaseAllocator, typeof(DOrtReleaseAllocator));
            OrtAllocatorAlloc = (DOrtAllocatorAlloc)Marshal.GetDelegateForFunctionPointer(api_.AllocatorAlloc, typeof(DOrtAllocatorAlloc));
            OrtAllocatorFree = (DOrtAllocatorFree)Marshal.GetDelegateForFunctionPointer(api_.AllocatorFree, typeof(DOrtAllocatorFree));
            OrtAllocatorGetInfo = (DOrtAllocatorGetInfo)Marshal.GetDelegateForFunctionPointer(api_.AllocatorGetInfo, typeof(DOrtAllocatorGetInfo));
            OrtAddFreeDimensionOverride = (DOrtAddFreeDimensionOverride)Marshal.GetDelegateForFunctionPointer(api_.AddFreeDimensionOverride, typeof(DOrtAddFreeDimensionOverride));
            OrtAddFreeDimensionOverrideByName = (DOrtAddFreeDimensionOverrideByName)Marshal.GetDelegateForFunctionPointer(api_.AddFreeDimensionOverrideByName, typeof(DOrtAddFreeDimensionOverrideByName));

            OrtCreateIoBinding = (DOrtCreateIoBinding)Marshal.GetDelegateForFunctionPointer(api_.CreateIoBinding, typeof(DOrtCreateIoBinding));
            OrtReleaseIoBinding = (DOrtReleaseIoBinding)Marshal.GetDelegateForFunctionPointer(api_.ReleaseIoBinding, typeof(DOrtReleaseIoBinding));
            OrtBindInput = (DOrtBindInput)Marshal.GetDelegateForFunctionPointer(api_.BindInput, typeof(DOrtBindInput));
            OrtSynchronizeBoundInputs = (DOrtSynchronizeBoundInputs)Marshal.GetDelegateForFunctionPointer(api_.SynchronizeBoundInputs, typeof(DOrtSynchronizeBoundInputs));
            OrtBindOutput = (DOrtBindOutput)Marshal.GetDelegateForFunctionPointer(api_.BindOutput, typeof(DOrtBindOutput));
            OrtBindOutputToDevice = (DOrtBindOutputToDevice)Marshal.GetDelegateForFunctionPointer(api_.BindOutputToDevice, typeof(DOrtBindOutputToDevice));
            OrtSynchronizeBoundOutputs = (DOrtSynchronizeBoundOutputs)Marshal.GetDelegateForFunctionPointer(api_.SynchronizeBoundOutputs, typeof(DOrtSynchronizeBoundOutputs));
            OrtGetBoundOutputNames = (DOrtGetBoundOutputNames)Marshal.GetDelegateForFunctionPointer(api_.GetBoundOutputNames, typeof(DOrtGetBoundOutputNames));
            OrtGetBoundOutputValues = (DOrtGetBoundOutputValues)Marshal.GetDelegateForFunctionPointer(api_.GetBoundOutputValues, typeof(DOrtGetBoundOutputValues));
            OrtClearBoundInputs = (DOrtClearBoundInputs)Marshal.GetDelegateForFunctionPointer(api_.ClearBoundInputs, typeof(DOrtClearBoundInputs));
            OrtClearBoundOutputs = (DOrtClearBoundOutputs)Marshal.GetDelegateForFunctionPointer(api_.ClearBoundOutputs, typeof(DOrtClearBoundOutputs));

            OrtTensorAt = (DOrtTensorAt)Marshal.GetDelegateForFunctionPointer(api_.TensorAt, typeof(DOrtTensorAt));
            OrtCreateAndRegisterAllocator = (DOrtCreateAndRegisterAllocator)Marshal.GetDelegateForFunctionPointer(api_.CreateAndRegisterAllocator, typeof(DOrtCreateAndRegisterAllocator));
            OrtSetLanguageProjection = (DOrtSetLanguageProjection)Marshal.GetDelegateForFunctionPointer(api_.SetLanguageProjection, typeof(DOrtSetLanguageProjection));

            OrtHasValue = (DOrtHasValue)Marshal.GetDelegateForFunctionPointer(api_.HasValue, typeof(DOrtHasValue));
            OrtGetValue = (DOrtGetValue)Marshal.GetDelegateForFunctionPointer(api_.GetValue, typeof(DOrtGetValue));
            OrtGetValueCount = (DOrtGetValueCount)Marshal.GetDelegateForFunctionPointer(api_.GetValueCount, typeof(DOrtGetValueCount));
            OrtCreateValue = (DOrtCreateValue)Marshal.GetDelegateForFunctionPointer(api_.CreateValue, typeof(DOrtCreateValue));
            OrtGetValueType = (DOrtGetValueType)Marshal.GetDelegateForFunctionPointer(api_.GetValueType, typeof(DOrtGetValueType));
            OrtGetOnnxTypeFromTypeInfo = (DOrtGetOnnxTypeFromTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.GetOnnxTypeFromTypeInfo, typeof(DOrtGetOnnxTypeFromTypeInfo));
            OrtGetTypeInfo = (DOrtGetTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.GetTypeInfo, typeof(DOrtGetTypeInfo));
            OrtCreateTensorAsOrtValue = (DOrtCreateTensorAsOrtValue)Marshal.GetDelegateForFunctionPointer(api_.CreateTensorAsOrtValue, typeof(DOrtCreateTensorAsOrtValue));
            OrtCreateTensorWithDataAsOrtValue = (DOrtCreateTensorWithDataAsOrtValue)Marshal.GetDelegateForFunctionPointer(api_.CreateTensorWithDataAsOrtValue, typeof(DOrtCreateTensorWithDataAsOrtValue));
            OrtValueIsTensor = (DOrtValueIsTensor)Marshal.GetDelegateForFunctionPointer(api_.IsTensor, typeof(DOrtValueIsTensor));
            OrtValueIsSparseTensor = (DOrtValueIsSparseTensor)Marshal.GetDelegateForFunctionPointer(api_.IsSparseTensor, typeof(DOrtValueIsSparseTensor));
            OrtGetTensorMutableData = (DOrtGetTensorMutableData)Marshal.GetDelegateForFunctionPointer(api_.GetTensorMutableData, typeof(DOrtGetTensorMutableData));
            OrtFillStringTensor = (DOrtFillStringTensor)Marshal.GetDelegateForFunctionPointer(api_.FillStringTensor, typeof(DOrtFillStringTensor));
            OrtGetResizedStringTensorElementBuffer = (DOrtGetResizedStringTensorElementBuffer)Marshal.GetDelegateForFunctionPointer(api_.GetResizedStringTensorElementBuffer, typeof(DOrtGetResizedStringTensorElementBuffer));
            OrtGetStringTensorContent = (DOrtGetStringTensorContent)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorContent, typeof(DOrtGetStringTensorContent));
            OrtGetStringTensorDataLength = (DOrtGetStringTensorDataLength)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorDataLength, typeof(DOrtGetStringTensorDataLength));
            OrtGetStringTensorElementLength = (DOrtGetStringTensorElementLength)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorElementLength, typeof(DOrtGetStringTensorElementLength));
            OrtGetStringTensorElement = (DOrtGetStringTensorElement)Marshal.GetDelegateForFunctionPointer(api_.GetStringTensorElement, typeof(DOrtGetStringTensorElement));
            OrtCastTypeInfoToTensorInfo = (DOrtCastTypeInfoToTensorInfo)Marshal.GetDelegateForFunctionPointer(api_.CastTypeInfoToTensorInfo, typeof(DOrtCastTypeInfoToTensorInfo));
            OrtGetTensorTypeAndShape = (DOrtGetTensorTypeAndShape)Marshal.GetDelegateForFunctionPointer(api_.GetTensorTypeAndShape, typeof(DOrtGetTensorTypeAndShape));
            OrtReleaseTensorTypeAndShapeInfo = (DOrtReleaseTensorTypeAndShapeInfo)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTensorTypeAndShapeInfo, typeof(DOrtReleaseTensorTypeAndShapeInfo));
            OrtGetTensorElementType = (DOrtGetTensorElementType)Marshal.GetDelegateForFunctionPointer(api_.GetTensorElementType, typeof(DOrtGetTensorElementType));
            OrtGetDimensionsCount = (DOrtGetDimensionsCount)Marshal.GetDelegateForFunctionPointer(api_.GetDimensionsCount, typeof(DOrtGetDimensionsCount));
            OrtGetDimensions = (DOrtGetDimensions)Marshal.GetDelegateForFunctionPointer(api_.GetDimensions, typeof(DOrtGetDimensions));
            OrtGetSymbolicDimensions = (DOrtGetSymbolicDimensions)Marshal.GetDelegateForFunctionPointer(api_.GetSymbolicDimensions, typeof(DOrtGetSymbolicDimensions));
            OrtGetTensorShapeElementCount = (DOrtGetTensorShapeElementCount)Marshal.GetDelegateForFunctionPointer(api_.GetTensorShapeElementCount, typeof(DOrtGetTensorShapeElementCount));
            OrtGetTensorMemoryInfo = (DOrtGetTensorMemoryInfo)Marshal.GetDelegateForFunctionPointer(api_.GetTensorMemoryInfo, typeof(DOrtGetTensorMemoryInfo));
            // MapTypeInfo
            OrtGetMapKeyType = (DGetMapKeyType)Marshal.GetDelegateForFunctionPointer(api_.GetMapKeyType, typeof(DGetMapKeyType));
            OrtCastTypeInfoToMapTypeInfo = (DCastTypeInfoToMapTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.CastTypeInfoToMapTypeInfo, typeof(DCastTypeInfoToMapTypeInfo));
            OrtGetMapValueType = (DGetMapValueType)Marshal.GetDelegateForFunctionPointer(api_.GetMapValueType, typeof(DGetMapValueType));
            // SequenceTypeInfo
            OrtCastTypeInfoToSequenceTypeInfo = (DCastTypeInfoToSequenceTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.CastTypeInfoToSequenceTypeInfo, typeof(DCastTypeInfoToSequenceTypeInfo));
            OrtGetSequenceElementType = (DGetSequenceElementType)Marshal.GetDelegateForFunctionPointer(api_.GetSequenceElementType, typeof(DGetSequenceElementType));
            // Optional Type info
            OrtCastTypeInfoToOptionalTypeInfo = (DOrtCastTypeInfoToOptionalTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.CastTypeInfoToOptionalTypeInfo, typeof(DOrtCastTypeInfoToOptionalTypeInfo));
            OrtGetOptionalContainedTypeInfo = (DGetOptionalContainedTypeInfo)Marshal.GetDelegateForFunctionPointer(api_.GetOptionalContainedTypeInfo, typeof(DGetOptionalContainedTypeInfo));
            OrtReleaseValue = (DOrtReleaseValue)Marshal.GetDelegateForFunctionPointer(api_.ReleaseValue, typeof(DOrtReleaseValue));

            OrtSessionGetModelMetadata = (DOrtSessionGetModelMetadata)Marshal.GetDelegateForFunctionPointer(api_.SessionGetModelMetadata, typeof(DOrtSessionGetModelMetadata));
            OrtModelMetadataGetProducerName = (DOrtModelMetadataGetProducerName)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetProducerName, typeof(DOrtModelMetadataGetProducerName));
            OrtModelMetadataGetGraphName = (DOrtModelMetadataGetGraphName)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetGraphName, typeof(DOrtModelMetadataGetGraphName));
            OrtModelMetadataGetDomain = (DOrtModelMetadataGetDomain)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetDomain, typeof(DOrtModelMetadataGetDomain));
            OrtModelMetadataGetDescription = (DOrtModelMetadataGetDescription)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetDescription, typeof(DOrtModelMetadataGetDescription));
            OrtModelMetadataGetGraphDescription = (DOrtModelMetadataGetGraphDescription)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetGraphDescription, typeof(DOrtModelMetadataGetGraphDescription));
            OrtModelMetadataGetVersion = (DOrtModelMetadataGetVersion)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetVersion, typeof(DOrtModelMetadataGetVersion));
            OrtModelMetadataGetCustomMetadataMapKeys = (DOrtModelMetadataGetCustomMetadataMapKeys)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataGetCustomMetadataMapKeys, typeof(DOrtModelMetadataGetCustomMetadataMapKeys));
            OrtModelMetadataLookupCustomMetadataMap = (DOrtModelMetadataLookupCustomMetadataMap)Marshal.GetDelegateForFunctionPointer(api_.ModelMetadataLookupCustomMetadataMap, typeof(DOrtModelMetadataLookupCustomMetadataMap));
            OrtReleaseModelMetadata = (DOrtReleaseModelMetadata)Marshal.GetDelegateForFunctionPointer(api_.ReleaseModelMetadata, typeof(DOrtReleaseModelMetadata));

            OrtGetAvailableProviders = (DOrtGetAvailableProviders)Marshal.GetDelegateForFunctionPointer(api_.GetAvailableProviders, typeof(DOrtGetAvailableProviders));
            OrtReleaseAvailableProviders = (DOrtReleaseAvailableProviders)Marshal.GetDelegateForFunctionPointer(api_.ReleaseAvailableProviders, typeof(DOrtReleaseAvailableProviders));

            OrtCreatePrepackedWeightsContainer = (DOrtCreatePrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.CreatePrepackedWeightsContainer, typeof(DOrtCreatePrepackedWeightsContainer));
            OrtReleasePrepackedWeightsContainer = (DOrtReleasePrepackedWeightsContainer)Marshal.GetDelegateForFunctionPointer(api_.ReleasePrepackedWeightsContainer, typeof(DOrtReleasePrepackedWeightsContainer));

            SessionOptionsAppendExecutionProvider_TensorRT_V2 = (DSessionOptionsAppendExecutionProvider_TensorRT_V2)Marshal.GetDelegateForFunctionPointer(
                                                             api_.SessionOptionsAppendExecutionProvider_TensorRT_V2, typeof(DSessionOptionsAppendExecutionProvider_TensorRT_V2));
            OrtCreateTensorRTProviderOptions = (DOrtCreateTensorRTProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateTensorRTProviderOptions, typeof(DOrtCreateTensorRTProviderOptions));
            OrtUpdateTensorRTProviderOptions = (DOrtUpdateTensorRTProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.UpdateTensorRTProviderOptions, typeof(DOrtUpdateTensorRTProviderOptions));
            OrtGetTensorRTProviderOptionsAsString = (DOrtGetTensorRTProviderOptionsAsString)Marshal.GetDelegateForFunctionPointer(api_.GetTensorRTProviderOptionsAsString, typeof(DOrtGetTensorRTProviderOptionsAsString));
            OrtReleaseTensorRTProviderOptions = (DOrtReleaseTensorRTProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseTensorRTProviderOptions, typeof(DOrtReleaseTensorRTProviderOptions));

            SessionOptionsAppendExecutionProvider_CUDA = (DSessionOptionsAppendExecutionProvider_CUDA)Marshal.GetDelegateForFunctionPointer(
                                                 api_.SessionOptionsAppendExecutionProvider_CUDA, typeof(DSessionOptionsAppendExecutionProvider_CUDA));
            SessionOptionsAppendExecutionProvider_CUDA_V2 = (DSessionOptionsAppendExecutionProvider_CUDA_V2)Marshal.GetDelegateForFunctionPointer(
                                                 api_.SessionOptionsAppendExecutionProvider_CUDA_V2, typeof(DSessionOptionsAppendExecutionProvider_CUDA_V2));
            OrtCreateCUDAProviderOptions = (DOrtCreateCUDAProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateCUDAProviderOptions, typeof(DOrtCreateCUDAProviderOptions));
            OrtUpdateCUDAProviderOptions = (DOrtUpdateCUDAProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.UpdateCUDAProviderOptions, typeof(DOrtUpdateCUDAProviderOptions));
            OrtGetCUDAProviderOptionsAsString = (DOrtGetCUDAProviderOptionsAsString)Marshal.GetDelegateForFunctionPointer(api_.GetCUDAProviderOptionsAsString, typeof(DOrtGetCUDAProviderOptionsAsString));
            OrtReleaseCUDAProviderOptions = (DOrtReleaseCUDAProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseCUDAProviderOptions, typeof(DOrtReleaseCUDAProviderOptions));
            SessionOptionsAppendExecutionProvider
                = (DSessionOptionsAppendExecutionProvider)Marshal.GetDelegateForFunctionPointer(
                    api_.SessionOptionsAppendExecutionProvider,
                    typeof(DSessionOptionsAppendExecutionProvider));
            OrtUpdateEnvWithCustomLogLevel = (DOrtUpdateEnvWithCustomLogLevel)Marshal.GetDelegateForFunctionPointer(api_.UpdateEnvWithCustomLogLevel, typeof(DOrtUpdateEnvWithCustomLogLevel));
            SessionOptionsAppendExecutionProvider_ROCM = (DSessionOptionsAppendExecutionProvider_ROCM)Marshal.GetDelegateForFunctionPointer(
                                                 api_.SessionOptionsAppendExecutionProvider_ROCM, typeof(DSessionOptionsAppendExecutionProvider_ROCM));
            OrtCreateROCMProviderOptions = (DOrtCreateROCMProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.CreateROCMProviderOptions, typeof(DOrtCreateROCMProviderOptions));
            OrtUpdateROCMProviderOptions = (DOrtUpdateROCMProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.UpdateROCMProviderOptions, typeof(DOrtUpdateROCMProviderOptions));
            OrtGetROCMProviderOptionsAsString = (DOrtGetROCMProviderOptionsAsString)Marshal.GetDelegateForFunctionPointer(api_.GetROCMProviderOptionsAsString, typeof(DOrtGetROCMProviderOptionsAsString));
            OrtReleaseROCMProviderOptions = (DOrtReleaseROCMProviderOptions)Marshal.GetDelegateForFunctionPointer(api_.ReleaseROCMProviderOptions, typeof(DOrtReleaseROCMProviderOptions));
        }

        internal class NativeLib
        {
#if __ANDROID__
            // define the library name required for android
            internal const string DllName = "libonnxruntime.so";
#elif __IOS__
            // define the library name required for iOS
            internal const string DllName = "__Internal";
#else
            internal const string DllName = "onnxruntime";
#endif
        }

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern ref OrtApiBase OrtGetApiBase();

        #region Runtime/Environment API

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateEnv(
            OrtLoggingLevel defaultLoggingLevel,
            byte[] /*const char* */ logId,
            out IntPtr /*(OrtEnv*)*/ env);

        public static DOrtCreateEnv OrtCreateEnv;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateEnvWithCustomLogger(
            IntPtr /* (OrtLoggingFunction*) */ loggingFunction,
            IntPtr /* (void*) */ loggerParam,
            OrtLoggingLevel defaultLoggingLevel,
            byte[] /* const char* */ logId,
            out IntPtr /*(OrtEnv*)*/ env);

        public static DOrtCreateEnvWithCustomLogger OrtCreateEnvWithCustomLogger;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateEnvWithGlobalThreadPools(
            OrtLoggingLevel defaultWarningLevel,
            byte[] /*const char* */ logId,
            IntPtr /*(const OrtThreadingOptions *) */ threadingOptions,
            out IntPtr /*(OrtEnv*)*/ env);

        public static DOrtCreateEnvWithGlobalThreadPools OrtCreateEnvWithGlobalThreadPools;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */ DOrtCreateEnvWithCustomLoggerAndGlobalThreadPools(
            IntPtr /* OrtLoggingFunction */ loggingFunction,
            IntPtr /* void* */loggerParam,
            OrtLoggingLevel logSeverityLevel,
            byte[] /* const char* */ logId,
            IntPtr /*(const OrtThreadingOptions *) */ threadingOptions,
            out IntPtr /*(OrtEnv*)*/ env);

        public static DOrtCreateEnvWithCustomLoggerAndGlobalThreadPools OrtCreateEnvWithCustomLoggerAndGlobalThreadPools;

        // OrtReleaseEnv should not be used
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseEnv(IntPtr /*(OrtEnv*)*/ env);
        public static DOrtReleaseEnv OrtReleaseEnv;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtEnableTelemetryEvents(IntPtr /*(OrtEnv*)*/ env);
        public static DOrtEnableTelemetryEvents OrtEnableTelemetryEvents;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtDisableTelemetryEvents(IntPtr /*(OrtEnv*)*/ env);
        public static DOrtDisableTelemetryEvents OrtDisableTelemetryEvents;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtUpdateEnvWithCustomLogLevel(IntPtr /*(OrtEnv*)*/ env, OrtLoggingLevel custom_log_level);
        public static DOrtUpdateEnvWithCustomLogLevel OrtUpdateEnvWithCustomLogLevel;

        #endregion Runtime/Environment API

        #region Provider Options API

        /// <summary>
        /// Creates native OrtTensorRTProviderOptions instance
        /// </summary>
        /// <param name="trtProviderOptionsInstance">(output) native instance of OrtTensorRTProviderOptions</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateTensorRTProviderOptions(
            out IntPtr /*(OrtTensorRTProviderOptions**)*/ trtProviderOptionsInstance);
        public static DOrtCreateTensorRTProviderOptions OrtCreateTensorRTProviderOptions;

        /// <summary>
        /// Updates native OrtTensorRTProviderOptions instance using given key/value pairs
        /// </summary>
        /// <param name="trtProviderOptionsInstance">native instance of OrtTensorRTProviderOptions</param>
        /// <param name="providerOptionsKeys">configuration keys of OrtTensorRTProviderOptions</param>
        /// <param name="providerOptionsValues">configuration values of OrtTensorRTProviderOptions</param>
        /// <param name="numKeys">number of configuration keys</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtUpdateTensorRTProviderOptions(
            IntPtr /*(OrtTensorRTProviderOptions*)*/ trtProviderOptionsInstance,
            IntPtr[] /*(const char* const *)*/ providerOptionsKeys,
            IntPtr[] /*(const char* const *)*/ providerOptionsValues,
            UIntPtr /*(size_t)*/ numKeys);
        public static DOrtUpdateTensorRTProviderOptions OrtUpdateTensorRTProviderOptions;

        /// <summary>
        /// Get native OrtTensorRTProviderOptionsV2 in serialized string
        /// </summary>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="ptr">is a UTF-8 null terminated string allocated using 'allocator'</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtGetTensorRTProviderOptionsAsString(
            IntPtr /*(OrtTensorRTProviderOptionsV2**)*/ trtProviderOptionsInstance,
            IntPtr /*(OrtAllocator*)*/ allocator,
            out IntPtr /*(char**)*/ptr);
        public static DOrtGetTensorRTProviderOptionsAsString OrtGetTensorRTProviderOptionsAsString;

        /// <summary>
        /// Releases native OrtTensorRTProviderOptions instance
        /// </summary>
        /// <param name="trtProviderOptionsInstance">native instance of OrtTensorRTProviderOptions to be released</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseTensorRTProviderOptions(IntPtr /*(OrtTensorRTProviderOptions*)*/ trtProviderOptionsInstance);
        public static DOrtReleaseTensorRTProviderOptions OrtReleaseTensorRTProviderOptions;

        /// <summary>
        /// Creates native OrtCUDAProviderOptions instance
        /// </summary>
        /// <param name="cudaProviderOptionsInstance">(output) native instance of OrtCUDAProviderOptions</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateCUDAProviderOptions(
            out IntPtr /*(OrtCUDAProviderOptions**)*/ cudaProviderOptionsInstance);
        public static DOrtCreateCUDAProviderOptions OrtCreateCUDAProviderOptions;

        /// <summary>
        /// Updates native OrtCUDAProviderOptions instance using given key/value pairs
        /// </summary>
        /// <param name="cudaProviderOptionsInstance">native instance of OrtCUDAProviderOptions</param>
        /// <param name="providerOptionsKeys">configuration keys of OrtCUDAProviderOptions</param>
        /// <param name="providerOptionsValues">configuration values of OrtCUDAProviderOptions</param>
        /// <param name="numKeys">number of configuration keys</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtUpdateCUDAProviderOptions(
            IntPtr /*(OrtCUDAProviderOptions*)*/ cudaProviderOptionsInstance,
            IntPtr[] /*(const char* const *)*/ providerOptionsKeys,
            IntPtr[] /*(const char* const *)*/ providerOptionsValues,
            UIntPtr /*(size_t)*/ numKeys);
        public static DOrtUpdateCUDAProviderOptions OrtUpdateCUDAProviderOptions;

        /// <summary>
        /// Get native OrtCUDAProviderOptionsV2 in serialized string
        /// </summary>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="ptr">is a UTF-8 null terminated string allocated using 'allocator'</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtGetCUDAProviderOptionsAsString(
            IntPtr /*(OrtCUDAProviderOptionsV2**)*/ cudaProviderOptionsInstance,
            IntPtr /*(OrtAllocator*)*/ allocator,
            out IntPtr /*(char**)*/ptr);
        public static DOrtGetCUDAProviderOptionsAsString OrtGetCUDAProviderOptionsAsString;

        /// <summary>
        /// Releases native OrtCUDAProviderOptions instance
        /// </summary>
        /// <param name="cudaProviderOptionsInstance">native instance of OrtCUDAProviderOptions to be released</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseCUDAProviderOptions(IntPtr /*(OrtCUDAProviderOptions*)*/ cudaProviderOptionsInstance);
        public static DOrtReleaseCUDAProviderOptions OrtReleaseCUDAProviderOptions;

        /// <summary>
        /// Creates native OrtROCMProviderOptions instance
        /// </summary>
        /// <param name="rocmProviderOptionsInstance">(output) native instance of OrtROCMProviderOptions</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateROCMProviderOptions(
            out IntPtr /*(OrtROCMProviderOptions**)*/ rocmProviderOptionsInstance);
        public static DOrtCreateROCMProviderOptions OrtCreateROCMProviderOptions;

        /// <summary>
        /// Updates native OrtROCMProviderOptions instance using given key/value pairs
        /// </summary>
        /// <param name="rocmProviderOptionsInstance">native instance of OrtROCMProviderOptions</param>
        /// <param name="providerOptionsKeys">configuration keys of OrtROCMProviderOptions</param>
        /// <param name="providerOptionsValues">configuration values of OrtROCMProviderOptions</param>
        /// <param name="numKeys">number of configuration keys</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtUpdateROCMProviderOptions(
            IntPtr /*(OrtROCMProviderOptions*)*/ rocmProviderOptionsInstance,
            IntPtr[] /*(const char* const *)*/ providerOptionsKeys,
            IntPtr[] /*(const char* const *)*/ providerOptionsValues,
            UIntPtr /*(size_t)*/ numKeys);
        public static DOrtUpdateROCMProviderOptions OrtUpdateROCMProviderOptions;

        /// <summary>
        /// Get native OrtROCMProviderOptions in serialized string
        /// </summary>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="ptr">is a UTF-8 null terminated string allocated using 'allocator'</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtGetROCMProviderOptionsAsString(
            IntPtr /*(OrtROCMProviderOptions**)*/ rocmProviderOptionsInstance,
            IntPtr /*(OrtAllocator*)*/ allocator,
            out IntPtr /*(char**)*/ptr);
        public static DOrtGetROCMProviderOptionsAsString OrtGetROCMProviderOptionsAsString;

        /// <summary>
        /// Releases native OrtROCMProviderOptions instance
        /// </summary>
        /// <param name="rocmProviderOptionsInstance">native instance of OrtROCMProviderOptions to be released</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseROCMProviderOptions(IntPtr /*(OrtROCMProviderOptions*)*/ rocmProviderOptionsInstance);
        public static DOrtReleaseROCMProviderOptions OrtReleaseROCMProviderOptions;

        #endregion

        #region Status API
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate ErrorCode DOrtGetErrorCode(IntPtr /*(OrtStatus*)*/status);
        public static DOrtGetErrorCode OrtGetErrorCode;

        // returns char*, need to convert to string by the caller.
        // does not free the underlying OrtStatus*
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* char* */DOrtGetErrorMessage(IntPtr /* (OrtStatus*) */status);
        public static DOrtGetErrorMessage OrtGetErrorMessage;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseStatus(IntPtr /*(OrtStatus*)*/ statusPtr);
        public static DOrtReleaseStatus OrtReleaseStatus;

        #endregion Status API

        #region InferenceSession API

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateSession(
                                                IntPtr /* (OrtEnv*) */ environment,
                                                //[MarshalAs(UnmanagedType.LPStr)]string modelPath
                                                byte[] modelPath,
                                                IntPtr /* (OrtSessionOptions*) */sessopnOptions,
                                                out IntPtr /**/ session);

        public static DOrtCreateSession OrtCreateSession;

        /// <summary>
        /// Creates an instance of OrtSession with provided parameters
        /// </summary>
        /// <param name="environment">Native OrtEnv instance</param>
        /// <param name="modelPath">UTF-8 bytes corresponding to model string path</param>
        /// <param name="sessionOptions">Native SessionOptions instance</param>
        /// <param name="prepackedWeightsContainer">Native OrtPrepackedWeightsContainer instance</param>
        /// <param name="session">(Output) Created native OrtSession instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateSessionWithPrepackedWeightsContainer(
                                        IntPtr /* (OrtEnv*) */ environment,
                                        byte[] modelPath,
                                        IntPtr /* (OrtSessionOptions*) */sessionOptions,
                                        IntPtr /* (OrtPrepackedWeightsContainer*) */prepackedWeightsContainer,
                                        out IntPtr /* (OrtSession**) */ session);

        public static DOrtCreateSessionWithPrepackedWeightsContainer OrtCreateSessionWithPrepackedWeightsContainer;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateSessionFromArray(
                                                IntPtr /* (OrtEnv*) */ environment,
                                                byte[] modelData,
                                                UIntPtr modelSize,
                                                IntPtr /* (OrtSessionOptions*) */ sessionOptions,
                                                out IntPtr /**/ session);
        public static DOrtCreateSessionFromArray OrtCreateSessionFromArray;

        /// <summary>
        /// Creates an instance of OrtSession with provided parameters
        /// </summary>
        /// <param name="environment">Native OrtEnv instance</param>
        /// <param name="modelData">Byte array correspoonding to the model</param>
        /// <param name="modelSize">Size of the model in bytes</param>
        /// <param name="sessionOptions">Native SessionOptions instance</param>
        /// <param name="prepackedWeightsContainer">Native OrtPrepackedWeightsContainer instance</param>
        /// <param name="session">(Output) Created native OrtSession instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus* */DOrtCreateSessionFromArrayWithPrepackedWeightsContainer(
                                        IntPtr /* (OrtEnv*) */ environment,
                                        byte[] /* (void*) */ modelData,
                                        UIntPtr /* (size_t) */ modelSize,
                                        IntPtr /* (OrtSessionOptions*) */ sessionOptions,
                                        IntPtr /* (OrtPrepackedWeightsContainer*) */prepackedWeightsContainer,
                                        out IntPtr /* (OrtSession**) */ session);
        public static DOrtCreateSessionFromArrayWithPrepackedWeightsContainer OrtCreateSessionFromArrayWithPrepackedWeightsContainer;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(ONNStatus*)*/ DOrtRun(
                                                IntPtr /*(OrtSession*)*/ session,
                                                IntPtr /*(OrtSessionRunOptions*)*/ runOptions,  // can be null to use the default options
                                                IntPtr[] inputNames,
                                                IntPtr[] /* (OrtValue*[])*/ inputValues,
                                                UIntPtr inputCount,
                                                IntPtr[] outputNames,
                                                UIntPtr outputCount,
                                                IntPtr[] outputValues /* An array of output value pointers. Array must be allocated by the caller */
                                                );

        public static DOrtRun OrtRun;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(ONNStatus*)*/ DOrtRunWithBinding(
                                                IntPtr /*(OrtSession*)*/ session,
                                                IntPtr /*(OrtSessionRunOptions*)*/ runOptions, // can not be null
                                                IntPtr /*(const OrtIoBinding*)*/ io_binding
                                                );

        public static DOrtRunWithBinding OrtRunWithBinding;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetInputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);

        public static DOrtSessionGetInputCount OrtSessionGetInputCount;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetOutputCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);

        public static DOrtSessionGetOutputCount OrtSessionGetOutputCount;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetOverridableInitializerCount(
                                                IntPtr /*(OrtSession*)*/ session,
                                                out UIntPtr count);

        public static DOrtSessionGetOverridableInitializerCount OrtSessionGetOverridableInitializerCount;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetInputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);

        public static DOrtSessionGetInputName OrtSessionGetInputName;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOutputName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);

        public static DOrtSessionGetOutputName OrtSessionGetOutputName;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionEndProfiling(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/profile_file);

        public static DOrtSessionEndProfiling OrtSessionEndProfiling;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOverridableInitializerName(
                                                IntPtr /*(OrtSession*)*/ session,
                                                UIntPtr index,
                                                IntPtr /*(OrtAllocator*)*/ allocator,
                                                out IntPtr /*(char**)*/name);

        public static DOrtSessionGetOverridableInitializerName OrtSessionGetOverridableInitializerName;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetInputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /*(struct OrtTypeInfo**)*/ typeInfo);

        public static DOrtSessionGetInputTypeInfo OrtSessionGetInputTypeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOutputTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);

        public static DOrtSessionGetOutputTypeInfo OrtSessionGetOutputTypeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtSessionGetOverridableInitializerTypeInfo(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                UIntPtr index,
                                                out IntPtr /* (struct OrtTypeInfo**)*/ typeInfo);

        public static DOrtSessionGetOverridableInitializerTypeInfo OrtSessionGetOverridableInitializerTypeInfo;

        // release the typeinfo using OrtReleaseTypeInfo
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseTypeInfo(IntPtr /*(OrtTypeInfo*)*/session);
        public static DOrtReleaseTypeInfo OrtReleaseTypeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseSession(IntPtr /*(OrtSession*)*/session);
        public static DOrtReleaseSession OrtReleaseSession;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSessionGetProfilingStartTimeNs(
                                                IntPtr /*(const OrtSession*)*/ session,
                                                out UIntPtr /*(ulong* out)*/ startTime);
        public static DOrtSessionGetProfilingStartTimeNs OrtSessionGetProfilingStartTimeNs;

        #endregion InferenceSession API

        #region SessionOptions API

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateSessionOptions(out IntPtr /*(OrtSessionOptions**)*/ sessionOptions);
        public static DOrtCreateSessionOptions OrtCreateSessionOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseSessionOptions(IntPtr /*(OrtSessionOptions*)*/session);
        public static DOrtReleaseSessionOptions OrtReleaseSessionOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCloneSessionOptions(IntPtr /*(OrtSessionOptions*)*/ sessionOptions, out IntPtr /*(OrtSessionOptions**)*/ output);
        public static DOrtCloneSessionOptions OrtCloneSessionOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionExecutionMode(IntPtr /*(OrtSessionOptions*)*/ options,
        ExecutionMode execution_mode);
        public static DOrtSetSessionExecutionMode OrtSetSessionExecutionMode;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetOptimizedModelFilePath(IntPtr /* OrtSessionOptions* */ options, byte[] optimizedModelFilepath);
        public static DOrtSetOptimizedModelFilePath OrtSetOptimizedModelFilePath;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableProfiling(IntPtr /* OrtSessionOptions* */ options, byte[] profilePathPrefix);
        public static DOrtEnableProfiling OrtEnableProfiling;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableProfiling(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableProfiling OrtDisableProfiling;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableMemPattern(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtEnableMemPattern OrtEnableMemPattern;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableMemPattern(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableMemPattern OrtDisableMemPattern;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtEnableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtEnableCpuMemArena OrtEnableCpuMemArena;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtDisableCpuMemArena(IntPtr /* OrtSessionOptions* */ options);
        public static DOrtDisableCpuMemArena OrtDisableCpuMemArena;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogId(IntPtr /* OrtSessionOptions* */ options, byte[] /* const char* */logId);
        public static DOrtSetSessionLogId OrtSetSessionLogId;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogVerbosityLevel(IntPtr /* OrtSessionOptions* */ options, int sessionLogVerbosityLevel);
        public static DOrtSetSessionLogVerbosityLevel OrtSetSessionLogVerbosityLevel;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionLogSeverityLevel(IntPtr /* OrtSessionOptions* */ options, OrtLoggingLevel sessionLogSeverityLevel);
        public static DOrtSetSessionLogSeverityLevel OrtSetSessionLogSeverityLevel;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetIntraOpNumThreads(IntPtr /* OrtSessionOptions* */ options, int intraOpNumThreads);
        public static DOrtSetIntraOpNumThreads OrtSetIntraOpNumThreads;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetInterOpNumThreads(IntPtr /* OrtSessionOptions* */ options, int interOpNumThreads);
        public static DOrtSetInterOpNumThreads OrtSetInterOpNumThreads;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetSessionGraphOptimizationLevel(IntPtr /* OrtSessionOptions* */ options, GraphOptimizationLevel graphOptimizationLevel);
        public static DOrtSetSessionGraphOptimizationLevel OrtSetSessionGraphOptimizationLevel;

        /// <summary>
        /// Add session config entry
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="configKey">Config key</param>
        /// <param name="configValue">Config value</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtAddSessionConfigEntry(IntPtr /* OrtSessionOptions* */ options,
                                                                          byte[] /* const char* */configKey,
                                                                          byte[] /* const char* */ configValue);
        public static DOrtAddSessionConfigEntry OrtAddSessionConfigEntry;

        //
        // The below OrtSessionOptionsAppendExecutionProvider_XYZ calls are using a publicly exported symbol from the
        // ONNX Runtime library for the EP (defined in the EP's provider factory .cc file) and not a function pointer
        // in OrtApis. This mechanism is being deprecated in favor of using OrtApis, as the latter has the ability to
        // return a graceful message if the EP is not included in the build.
        // New EPs should use OrtApis, preferably leveraging the generic SessionOptionsAppendExecutionProvider
        // entry point where optional provider configuration key/value pairs can be passed in.

        ///**
        //  * The order of invocation indicates the preference order as well. In other words call this method
        //  * on your most preferred execution provider first followed by the less preferred ones.
        //  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
        //  */
        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CPU(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);

#if __ANDROID__
        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Nnapi(IntPtr /*(OrtSessionOptions*)*/ options, uint nnapi_flags);
#endif

#if __ENABLE_COREML__
        // CoreML is available on iOS and macOS so we can't exclude based on __MOBILE__ && __IOS__
        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CoreML(IntPtr /*(OrtSessionOptions*)*/ options, uint coreml_flags);
#endif

#if !__MOBILE__
        // on non-mobile platforms any of these EPs are possible
        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Dnnl(IntPtr /*(OrtSessionOptions*) */ options, int use_arena);

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_CUDA(IntPtr /*(OrtSessionOptions*) */ options, int device_id);

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_ROCM(IntPtr /*(OrtSessionOptions*) */ options, int device_id);

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_DML(IntPtr /*(OrtSessionOptions*) */ options, int device_id);

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_OpenVINO(IntPtr /*(OrtSessionOptions*)*/ options, byte[] /*(const char*)*/ device_id);

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Tensorrt(IntPtr /*(OrtSessionOptions*)*/ options, int device_id);

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_MIGraphX(IntPtr /*(OrtSessionOptions*)*/ options, int device_id);

        [DllImport(NativeLib.DllName, CharSet = CharSet.Ansi)]
        public static extern IntPtr /*(OrtStatus*)*/ OrtSessionOptionsAppendExecutionProvider_Tvm(IntPtr /*(OrtSessionOptions*) */ options, byte[] /*(char char*)*/ settings);
#endif
        /// <summary>
        /// Append a TensorRT EP instance (configured based on given provider options) to the native OrtSessionOptions instance
        /// </summary>
        /// <param name="options">Native OrtSessionOptions instance</param>
        /// <param name="trtProviderOptions">Native OrtTensorRTProviderOptions instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DSessionOptionsAppendExecutionProvider_TensorRT(
                                               IntPtr /*(OrtSessionOptions*)*/ options,
                                               IntPtr /*(const OrtTensorRTProviderOptions*)*/ trtProviderOptions);

        public static DSessionOptionsAppendExecutionProvider_TensorRT SessionOptionsAppendExecutionProvider_TensorRT;

        /// <summary>
        /// Append a TensorRT EP instance (configured based on given provider options) to the native OrtSessionOptions instance
        /// </summary>
        /// <param name="options">Native OrtSessionOptions instance</param>
        /// <param name="trtProviderOptions">Native OrtTensorRTProviderOptionsV2 instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DSessionOptionsAppendExecutionProvider_TensorRT_V2(
                                               IntPtr /*(OrtSessionOptions*)*/ options,
                                               IntPtr /*(const OrtTensorRTProviderOptionsV2*)*/ trtProviderOptions);

        public static DSessionOptionsAppendExecutionProvider_TensorRT_V2 SessionOptionsAppendExecutionProvider_TensorRT_V2;

        /// <summary>
        /// Append a CUDA EP instance (configured based on given provider options) to the native OrtSessionOptions instance
        /// </summary>
        /// <param name="options">Native OrtSessionOptions instance</param>
        /// <param name="cudaProviderOptions">Native OrtCUDAProviderOptions instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DSessionOptionsAppendExecutionProvider_CUDA(
                                               IntPtr /*(OrtSessionOptions*)*/ options,
                                               IntPtr /*(const OrtCUDAProviderOptions*)*/ cudaProviderOptions);

        public static DSessionOptionsAppendExecutionProvider_CUDA SessionOptionsAppendExecutionProvider_CUDA;

        /// <summary>
        /// Append a CUDA EP instance (configured based on given provider options) to the native OrtSessionOptions instance
        /// </summary>
        /// <param name="options">Native OrtSessionOptions instance</param>
        /// <param name="cudaProviderOptions">Native OrtCUDAProviderOptionsV2 instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DSessionOptionsAppendExecutionProvider_CUDA_V2(
                                               IntPtr /*(OrtSessionOptions*)*/ options,
                                               IntPtr /*(const OrtCUDAProviderOptionsV2*)*/ cudaProviderOptions);

        public static DSessionOptionsAppendExecutionProvider_CUDA_V2 SessionOptionsAppendExecutionProvider_CUDA_V2;

        /// <summary>
        /// Append a ROCm EP instance (configured based on given provider options) to the native OrtSessionOptions instance
        /// </summary>
        /// <param name="options">Native OrtSessionOptions instance</param>
        /// <param name="rocmProviderOptions">Native OrtROCMProviderOptions instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DSessionOptionsAppendExecutionProvider_ROCM(
                                               IntPtr /*(OrtSessionOptions*)*/ options,
                                               IntPtr /*(const OrtROCMProviderOptions*)*/ rocmProviderOptions);

        public static DSessionOptionsAppendExecutionProvider_ROCM SessionOptionsAppendExecutionProvider_ROCM;

        /// <summary>
        /// Free Dimension override (by denotation)
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="dimDenotation">Dimension denotation</param>
        /// <param name="dimValue">Dimension value</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtAddFreeDimensionOverride(IntPtr /*(OrtSessionOptions*)*/ options,
                                                                            byte[] /*(const char*)*/ dimDenotation,
                                                                            long dimValue);

        public static DOrtAddFreeDimensionOverride OrtAddFreeDimensionOverride;

        /// <summary>
        /// Free Dimension override (by name)
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="dimName">Dimension name</param>
        /// <param name="dimValue">Dimension value</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtAddFreeDimensionOverrideByName(IntPtr /*(OrtSessionOptions*)*/ options,
                                                                                  byte[] /*(const char*)*/ dimName,
                                                                                  long dimValue);

        public static DOrtAddFreeDimensionOverrideByName OrtAddFreeDimensionOverrideByName;

        /// <summary>
        /// Register custom op library
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="libraryPath">Library path</param>
        /// <param name="libraryHandle">(out) Native library handle</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtRegisterCustomOpsLibrary(IntPtr /*(OrtSessionOptions*) */ options,
                                                                            byte[] /*(const char*)*/ libraryPath,
                                                                            out IntPtr /*(void**)*/ libraryHandle);

        public static DOrtRegisterCustomOpsLibrary OrtRegisterCustomOpsLibrary;

        /// <summary>
        /// Register custom op library. ORT will manage freeing the library.
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="libraryPath">Library path</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtRegisterCustomOpsLibrary_V2(IntPtr /*(OrtSessionOptions*) */ options,
                                                                               byte[] /*(const ORTCHAR_T*)*/ libraryPath);

        public static DOrtRegisterCustomOpsLibrary_V2 OrtRegisterCustomOpsLibrary_V2;

        /// <summary>
        /// Add initializer that is shared across Sessions using this SessionOptions (by denotation)
        /// </summary>
        /// <param name="options">Native SessionOptions instance</param>
        /// <param name="name">Name of the initializer</param>
        /// <param name="ortValue">Native OrtValue instnce</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtAddInitializer(IntPtr /*(OrtSessionOptions*)*/ options,
                                                                  byte[] /*(const char*)*/ name,
                                                                  IntPtr /*(OrtValue*)*/ ortValue);

        public static DOrtAddInitializer OrtAddInitializer;

        /// <summary>
        /// Append an execution provider instance to the native OrtSessionOptions instance.
        ///
        /// 'SNPE' and 'XNNPACK' are currently supported as providerName values.
        ///
        /// The number of providerOptionsKeys must match the number of providerOptionsValues and equal numKeys.
        /// </summary>
        /// <param name="options">Native OrtSessionOptions instance</param>
        /// <param name="providerName">Execution provider to add.</param>
        /// <param name="providerOptionsKeys">Configuration keys to add</param>
        /// <param name="providerOptionsValues">Configuration values to add</param>
        /// <param name="numKeys">Number of configuration keys</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DSessionOptionsAppendExecutionProvider(
                                               IntPtr /*(OrtSessionOptions*)*/ options,
                                               byte[] /*(const char*)*/ providerName,
                                               IntPtr[] /*(const char* const *)*/ providerOptionsKeys,
                                               IntPtr[] /*(const char* const *)*/ providerOptionsValues,
                                               UIntPtr /*(size_t)*/ numKeys);

        public static DSessionOptionsAppendExecutionProvider SessionOptionsAppendExecutionProvider;

        #endregion

        #region RunOptions API

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateRunOptions(out IntPtr /* OrtRunOptions** */ runOptions);
        public static DOrtCreateRunOptions OrtCreateRunOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseRunOptions(IntPtr /*(OrtRunOptions*)*/options);
        public static DOrtReleaseRunOptions OrtReleaseRunOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, int value);
        public static DOrtRunOptionsSetRunLogVerbosityLevel OrtRunOptionsSetRunLogVerbosityLevel;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunLogSeverityLevel(IntPtr /* OrtRunOptions* */ options, OrtLoggingLevel value);
        public static DOrtRunOptionsSetRunLogSeverityLevel OrtRunOptionsSetRunLogSeverityLevel;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetRunTag(IntPtr /* OrtRunOptions* */ options, byte[] /* const char* */ runTag);
        public static DOrtRunOptionsSetRunTag OrtRunOptionsSetRunTag;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunLogVerbosityLevel(IntPtr /* OrtRunOptions* */ options, out int verbosityLevel);
        public static DOrtRunOptionsGetRunLogVerbosityLevel OrtRunOptionsGetRunLogVerbosityLevel;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunLogSeverityLevel(IntPtr /* OrtRunOptions* */ options,
            out OrtLoggingLevel severityLevel);
        public static DOrtRunOptionsGetRunLogSeverityLevel OrtRunOptionsGetRunLogSeverityLevel;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsGetRunTag(IntPtr /* const OrtRunOptions* */options, out IntPtr /* const char** */ runtag);
        public static DOrtRunOptionsGetRunTag OrtRunOptionsGetRunTag;

        // Set a flag so that any running OrtRun* calls that are using this instance of OrtRunOptions
        // will exit as soon as possible if the flag is true.
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsSetTerminate(IntPtr /* OrtRunOptions* */ options);
        public static DOrtRunOptionsSetTerminate OrtRunOptionsSetTerminate;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtRunOptionsUnsetTerminate(IntPtr /* OrtRunOptions* */ options);
        public static DOrtRunOptionsUnsetTerminate OrtRunOptionsUnsetTerminate;


        /// <summary>
        /// Add run config entry
        /// </summary>
        /// <param name="options">Native RunOptions instance</param>
        /// <param name="configKey">Config key</param>
        /// <param name="configValue">Config value</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtAddRunConfigEntry(IntPtr /* OrtRunOptions* */ options,
                                                                      byte[] /* const char* */configKey,
                                                                      byte[] /* const char* */ configValue);
        public static DOrtAddRunConfigEntry OrtAddRunConfigEntry;

        #endregion

        #region ThreadingOptions API

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateThreadingOptions(out IntPtr /* OrtCreateThreadingOptions** */ threadingOptions);
        public static DOrtCreateThreadingOptions OrtCreateThreadingOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtReleaseThreadingOptions(IntPtr /* OrtThreadingOptions* */ threadingOptions);
        public static DOrtReleaseThreadingOptions OrtReleaseThreadingOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtThreadingOptionsSetGlobalInterOpNumThreads(IntPtr /* OrtThreadingOptions* */ threadingOptions, int numThreads);
        public static DOrtThreadingOptionsSetGlobalInterOpNumThreads OrtThreadingOptionsSetGlobalInterOpNumThreads;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtThreadingOptionsSetGlobalIntraOpNumThreads(IntPtr /* OrtThreadingOptions* */ threadingOptions, int numThreads);
        public static DOrtThreadingOptionsSetGlobalIntraOpNumThreads OrtThreadingOptionsSetGlobalIntraOpNumThreads;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtThreadingOptionsSetGlobalDenormalAsZero(IntPtr /* OrtThreadingOptions* */ threadingOptions);
        public static DOrtThreadingOptionsSetGlobalDenormalAsZero OrtThreadingOptionsSetGlobalDenormalAsZero;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtThreadingOptionsSetGlobalSpinControl(IntPtr /* OrtThreadingOptions* */ threadingOptions, int allowSpinning);
        public static DOrtThreadingOptionsSetGlobalSpinControl OrtThreadingOptionsSetGlobalSpinControl;
        #endregion

        #region Allocator/MemoryInfo API

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*)*/ DOrtCreateMemoryInfo(
                                                            byte[] /*(const char*) */name,
                                                            OrtAllocatorType allocatorType,
                                                            int identifier,
                                                            OrtMemType memType,
                                                            out IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo    // memory ownership transfered to caller
                                                       );

        public static DOrtCreateMemoryInfo OrtCreateMemoryInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*)*/ DOrtCreateCpuMemoryInfo(
                                                            OrtAllocatorType allocatorType,
                                                            OrtMemType memoryType,
                                                            out IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo
                                                        );

        public static DOrtCreateCpuMemoryInfo OrtCreateCpuMemoryInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseMemoryInfo(IntPtr /*(OrtMemoryInfo*)*/ allocatorInfo);

        public static DOrtReleaseMemoryInfo OrtReleaseMemoryInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCompareMemoryInfo(
                                               IntPtr /*(const OrtMemoryInfo*)*/ info1,
                                               IntPtr /*(const OrtMemoryInfo*)*/ info2,
                                               out int /*(int* out)*/ result);

        public static DOrtCompareMemoryInfo OrtCompareMemoryInfo;

        /**
        * Do not free the returned value
        */
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetName(IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info, out IntPtr /*(const char**)*/ name);

        public static DOrtMemoryInfoGetName OrtMemoryInfoGetName;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetId(IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info, out int /*(int* out)*/ id);

        public static DOrtMemoryInfoGetId OrtMemoryInfoGetId;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetMemType(
                                                IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info,
                                                out OrtMemType /*(OrtMemType*)*/ mem_type);

        public static DOrtMemoryInfoGetMemType OrtMemoryInfoGetMemType;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtMemoryInfoGetType(
                                                IntPtr /*(const OrtMemoryInfo* ptr)*/ mem_info,
                                                out OrtAllocatorType /*(OrtAllocatorType*)*/ alloc_type
                                                );

        public static DOrtMemoryInfoGetType OrtMemoryInfoGetType;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtGetAllocatorWithDefaultOptions(out IntPtr /*(OrtAllocator**)*/ allocator);

        public static DOrtGetAllocatorWithDefaultOptions OrtGetAllocatorWithDefaultOptions;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/DOrtAllocatorGetInfo(IntPtr /*(const OrtAllocator*)*/ ptr, out IntPtr /*(const struct OrtMemoryInfo**)*/info);

        public static DOrtAllocatorGetInfo OrtAllocatorGetInfo;

        /// <summary>
        /// Create an instance of arena configuration which will be used to create an arena based allocator
        /// See docs/C_API.md for details on what the following parameters mean and how to choose these values
        /// </summary>
        /// <param name="maxMemory">Maximum amount of memory the arena allocates</param>
        /// <param name="arenaExtendStrategy">Strategy for arena expansion</param>
        /// <param name="initialChunkSizeBytes">Size of the region that the arena allocates first</param>
        /// <param name="maxDeadBytesPerChunk">Maximum amount of fragmentation allowed per chunk</param>
        /// <returns>Pointer to a native OrtStatus instance indicating success/failure of config creation</returns>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateArenaCfg(UIntPtr /*(size_t)*/ maxMemory, int /*(int)*/ arenaExtendStrategy,
                                                                  int /*(int)*/ initialChunkSizeBytes, int /*(int)*/ maxDeadBytesPerChunk,
                                                                  out IntPtr /*(OrtArenaCfg**)*/ arenaCfg);

        public static DOrtCreateArenaCfg OrtCreateArenaCfg;

        /// <summary>
        /// Destroy an instance of an arena configuration instance
        /// </summary>
        /// <param name="arenaCfg">arena configuration instance to be destroyed</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseArenaCfg(IntPtr /*(OrtArenaCfg*)*/ arenaCfg);

        public static DOrtReleaseArenaCfg OrtReleaseArenaCfg;

        /// <summary>
        /// Create an instance of allocator according to mem_info
        /// </summary>
        /// <param name="session">Session that this allocator should be used with</param>
        /// <param name="info">memory allocator specs</param>
        /// <param name="allocator">out pointer to a new allocator instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateAllocator(IntPtr /*(const OrtSession*)*/ session, IntPtr /*(const OrtMemoryInfo*)*/ info, out IntPtr /*(OrtAllocator**)*/ allocator);

        public static DOrtCreateAllocator OrtCreateAllocator;

        /// <summary>
        /// Destroy an instance of an allocator created by OrtCreateAllocator
        /// </summary>
        /// <param name="allocator">instance to be destroyed</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseAllocator(IntPtr /*(OrtAllocator*)*/ allocator);

        public static DOrtReleaseAllocator OrtReleaseAllocator;

        /// <summary>
        /// Allocate  a chunk of native memory
        /// </summary>
        /// <param name="allocator">allocator instance</param>
        /// <param name="size">bytes to allocate</param>
        /// <param name="p">out pointer to the allocated memory. Must be freed by OrtAllocatorFree</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr DOrtAllocatorAlloc(IntPtr /*(OrtAllocator*)*/ allocator, UIntPtr /*size_t*/ size, out IntPtr /*(void**)*/ p);

        public static DOrtAllocatorAlloc OrtAllocatorAlloc;

        /// <summary>
        /// Release native memory allocated by an allocator
        /// </summary>
        /// <param name="allocator">allocator instance</param>
        /// <param name="p">pointer to native memory allocated by the allocator instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr DOrtAllocatorFree(IntPtr /*(OrtAllocator*)*/ allocator, IntPtr /*(void*)*/ p);

        public static DOrtAllocatorFree OrtAllocatorFree;

        #endregion Allocator/MemoryInfo API

        #region IoBinding API

        /// <summary>
        /// Create OrtIoBinding instance that is used to bind memory that is allocated
        /// either by a 3rd party allocator or an ORT device allocator. Such memory should be wrapped by
        /// a native OrtValue of Tensor type. By binding such named values you will direct ORT to read model inputs
        /// and write model outputs to the supplied memory.
        /// </summary>
        /// <param name="session">session to create OrtIoBinding instance</param>
        /// <param name="io_binding">out a new instance of OrtIoBinding</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtCreateIoBinding(IntPtr /*(const OrtSession*)*/ session, out IntPtr /*(OrtIoBinding)*/ io_binding);

        public static DOrtCreateIoBinding OrtCreateIoBinding;

        /// <summary>
        /// Destroy OrtIoBinding instance created by OrtCreateIoBinding
        /// </summary>
        /// <param name="io_bidning">instance of OrtIoBinding</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseIoBinding(IntPtr /*(OrtIoBinding)*/ io_binding);

        public static DOrtReleaseIoBinding OrtReleaseIoBinding;

        /// <summary>
        /// Bind OrtValue to the model input with the specified name
        /// If binding with the specified name already exists, it will be replaced
        /// </summary>
        /// <param name="io_bidning">instance of OrtIoBinding</param>
        /// <param name="name">model input name (utf-8)</param>
        /// <param name="ort_value">OrtValue that is used for input (may wrap arbitrary memory).
        ///      The param instance is copied internally so this argument may be released.
        /// </param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtBindInput(IntPtr /*(OrtIoBinding)*/ io_binding, byte[] /*(const char*)*/ name, IntPtr /*const OrtValue**/ ort_value);

        public static DOrtBindInput OrtBindInput;

        /// <summary>
        /// The API calls Sync() on all EP providers present. This blocks until the device has completed
        /// all preceding requested tasks. This is necessary when memory synchronization is required.
        /// For example, the memory bound to an input is likely to be on a different CUDA stream.
        /// For some scenarios and devices this may be a no-op, use
        /// your best judgment.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        /// <returns>An instance of OrtStatus or null</returns>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtSynchronizeBoundInputs(IntPtr /*(OrtIoBinding)*/ io_binding);

        public static DOrtSynchronizeBoundInputs OrtSynchronizeBoundInputs;

        /// <summary>
        /// Bind OrtValue to the model output with the specified name
        /// If binding with the specified name already exists, it will be replaced
        /// </summary>
        /// <param name="io_bidning">instance of OrtIoBinding</param>
        /// <param name="name">model output name (utf-8)</param>
        /// <param name="ort_value">OrtValue that is used for output (may wrap arbitrary memory).
        ///      The param instance is copied internally so this argument may be released.
        /// </param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtBindOutput(IntPtr /*(OrtIoBinding)*/ io_binding, byte[] /*(const char*) */ name, IntPtr /*const OrtValue**/ ort_value);

        public static DOrtBindOutput OrtBindOutput;

        /// <summary>
        /// Bind a device to the model output with the specified name
        /// This is useful when the OrtValue can not be allocated ahead of time
        /// due to unknown dimensions.
        /// </summary>
        /// <param name="io_binding">Instance of OrtIoBinding</param>
        /// <param name="name">UTF-8 zero terminated name</param>
        /// <param name="mem_info">OrtMemoryInfo instance that contains device id. May be obtained from the device specific allocator instance</param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtBindOutputToDevice(IntPtr /*(OrtIoBinding)*/ io_binding, byte[] /*(const char*) */ name, IntPtr /* const OrtMemoryInfo */ mem_info);

        public static DOrtBindOutputToDevice OrtBindOutputToDevice;

        /// <summary>
        /// The API calls Sync() on all EP providers present. This blocks until the device has completed
        /// all preceding requested tasks. This is necessary when memory synchronization is required.
        /// For some scenarios and devices this may be a no-op, use your best judgment.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        /// <returns>An instance of OrtStatus or null</returns>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtSynchronizeBoundOutputs(IntPtr /*(OrtIoBinding)*/ io_binding);

        public static DOrtSynchronizeBoundOutputs OrtSynchronizeBoundOutputs;

        /// <summary>
        /// The function will return all bound output names in the order they were bound.
        /// It is the same order that the output values will be returned after RunWithBinding() is used.
        /// The function will allocate two native allocations  using the allocator supplied.
        /// The caller is responsible for deallocating both of the buffers using the same allocator.
        /// You may use OrtMemoryAllocation disposable class to wrap those allocations.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        /// <param name="allocator">allocator to use for memory allocation</param>
        /// <param name="buffer">a continuous buffer that contains all output names.
        /// Names are not zero terminated use lengths to extract strings. This needs to be deallocated.</param>
        /// <param name="lengths">A buffer that contains lengths (size_t) for each of the returned strings in order.
        /// The buffer must be deallocated.</param>
        /// <param name="count">this contains the count of names returned which is the number of elements in lengths.</param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtGetBoundOutputNames(IntPtr /* (const OrtIoBinding*) */ io_binding, IntPtr /* OrtAllocator* */ allocator,
                                                                      out IntPtr /* char** */ buffer, out IntPtr /* size_t** */ lengths, out UIntPtr count);

        public static DOrtGetBoundOutputNames OrtGetBoundOutputNames;

        /// <summary>
        /// The function returns output values after the model has been run with RunWithBinding()
        /// It returns a natively allocated buffer of OrtValue pointers. All of the OrtValues must be individually
        /// released after no longer needed. You may use OrtValue disposable class to wrap the native handle and properly dispose it
        /// in connection with DisposableList<T>. All values are returned in the same order as they were bound.
        /// The buffer that contains OrtValues must deallocated using the same allocator that was specified as an argument.
        /// You may use an instance OrtMemoryAllocation to properly dispose of the native memory.
        /// </summary>
        /// <param name="io_binding">instance of OrtIOBinding</param>
        /// <param name="allocator">allocator to use to allocate output buffer</param>
        /// <param name="ortvalues">allocated buffer that contains pointers (IntPtr) to individual OrtValue instances</param>
        /// <param name="count">count of OrtValues returned</param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus*/ DOrtGetBoundOutputValues(IntPtr /* (const OrtIoBinding*) */ io_binding, IntPtr /* OrtAllocator* */ allocator,
                                                                       out IntPtr /* OrtValue** */ ortvalues, out UIntPtr count);

        public static DOrtGetBoundOutputValues OrtGetBoundOutputValues;

        /// <summary>
        /// Clears Input bindings. This is a convenience method.
        /// Releasing OrtIoBinding instance would clear all bound inputs.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtClearBoundInputs(IntPtr /*(OrtIoBinding)*/ io_binding);

        public static DOrtClearBoundInputs OrtClearBoundInputs;

        /// <summary>
        /// Clears Output bindings. This is a convenience method.
        /// Releasing OrtIoBinding instance would clear all bound outputs.
        /// </summary>
        /// <param name="io_binding">instance of OrtIoBinding</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtClearBoundOutputs(IntPtr /*(OrtIoBinding)*/ io_binding);

        public static DOrtClearBoundOutputs OrtClearBoundOutputs;

        /// <summary>
        /// Provides element-level access into a tensor.
        /// </summary>
        /// <param name="location_values">a pointer to an array of index values that specify an element's location in the tensor data blob</param>
        /// <param name="location_values_count">length of location_values</param>
        /// <param name="out">a pointer to the element specified by location_values</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtTensorAt(IntPtr /*(OrtIoBinding)*/ io_binding);

        public static DOrtTensorAt OrtTensorAt;

        /// <summary>
        /// Creates an allocator instance and registers it with the env to enable
        /// sharing between multiple sessions that use the same env instance.
        /// Lifetime of the created allocator will be valid for the duration of the environment.
        /// Returns an error if an allocator with the same OrtMemoryInfo is already registered.
        /// <param name="env">Native OrtEnv instance</param>
        /// <param name="memInfo">Native OrtMemoryInfo instance</param>
        /// <param name="arenaCfg">Native OrtArenaCfg instance</param>
        /// <retruns>A pointer to native ortStatus indicating success/failure</retruns>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateAndRegisterAllocator(IntPtr /*(OrtEnv*)*/ env,
                                                                               IntPtr /*(const OrtMemoryInfo*)*/ memInfo,
                                                                               IntPtr/*(const OrtArenaCfg*)*/ arenaCfg);

        public static DOrtCreateAndRegisterAllocator OrtCreateAndRegisterAllocator;

        /// <summary>
        /// Set the language projection for collecting telemetry data when Env is created
        /// </summary>
        /// <param name="projection">the source projected language</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtSetLanguageProjection(IntPtr /* (OrtEnv*) */ environment,
            int projection);

        public static DOrtSetLanguageProjection OrtSetLanguageProjection;

        #endregion IoBinding API

        #region ModelMetadata API

        /// <summary>
        /// Gets the ModelMetadata associated with an InferenceSession
        /// </summary>
        /// <param name="session">instance of OrtSession</param>
        /// <param name="modelMetadata">(output) instance of OrtModelMetadata</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtSessionGetModelMetadata(IntPtr /* (const OrtSession*) */ session, out IntPtr /* (OrtModelMetadata**) */ modelMetadata);

        public static DOrtSessionGetModelMetadata OrtSessionGetModelMetadata;

        /// <summary>
        /// Gets the producer name associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) producer name from the ModelMetadata instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetProducerName(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);

        public static DOrtModelMetadataGetProducerName OrtModelMetadataGetProducerName;

        /// <summary>
        /// Gets the graph name associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) graph name from the ModelMetadata instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetGraphName(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);

        public static DOrtModelMetadataGetGraphName OrtModelMetadataGetGraphName;

        /// <summary>
        /// Gets the domain associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) domain from the ModelMetadata instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetDomain(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);

        public static DOrtModelMetadataGetDomain OrtModelMetadataGetDomain;

        /// <summary>
        /// Gets the description associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) description from the ModelMetadata instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetDescription(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);

        public static DOrtModelMetadataGetDescription OrtModelMetadataGetDescription;

        /// <summary>
        /// Gets the description associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="value">(output) graph description from the ModelMetadata instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetGraphDescription(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char**) */ value);

        public static DOrtModelMetadataGetGraphDescription OrtModelMetadataGetGraphDescription;

        /// <summary>
        /// Gets the version associated with a ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="value">(output) version from the ModelMetadata instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetVersion(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
                                                                              out long /* (int64_t*) */ value);

        public static DOrtModelMetadataGetVersion OrtModelMetadataGetVersion;

        /// <summary>
        /// Gets all the keys in the custom metadata map in the ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="keys">(output) all keys in the custom metadata map</param>
        /// <param name="numKeys">(output) number of keys in the custom metadata map</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataGetCustomMetadataMapKeys(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
            IntPtr /* (OrtAllocator*) */ allocator, out IntPtr /* (char***) */ keys, out long /* (int64_t*) */ numKeys);

        public static DOrtModelMetadataGetCustomMetadataMapKeys OrtModelMetadataGetCustomMetadataMapKeys;

        /// <summary>
        /// Gets the value associated with the given key in custom metadata map in the ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        /// <param name="allocator">instance of OrtAllocator</param>
        /// <param name="key">key in the custom metadata map</param>
        /// <param name="value">(output) value for the key in the custom metadata map</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtModelMetadataLookupCustomMetadataMap(IntPtr /* (const OrtModelMetadata*) */ modelMetadata,
            IntPtr /* (OrtAllocator*) */ allocator, IntPtr /* (const char*) */ key, out IntPtr /* (char**) */ value);

        public static DOrtModelMetadataLookupCustomMetadataMap OrtModelMetadataLookupCustomMetadataMap;

        /// <summary>
        /// Frees ModelMetadata instance
        /// </summary>
        /// <param name="modelMetadata">instance of OrtModelMetadata</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseModelMetadata(IntPtr /*(OrtModelMetadata*)*/ modelMetadata);

        public static DOrtReleaseModelMetadata OrtReleaseModelMetadata;

        #endregion ModelMetadata API

        #region OrtValue API

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtHasValue(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(int*)*/ hasValue);

        public static DOrtHasValue OrtHasValue;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValue(IntPtr /*(OrtValue*)*/ value,
                                                                 int index,
                                                                 IntPtr /*(OrtAllocator*)*/ allocator,
                                                                 out IntPtr /*(OrtValue**)*/ outputValue);

        public static DOrtGetValue OrtGetValue;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValueType(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(OnnxValueType*)*/ onnxtype);

        public static DOrtGetValueType OrtGetValueType;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetOnnxTypeFromTypeInfo(IntPtr /*(OrtTypeInfo*)*/ typeinfo, out IntPtr /*(OnnxValueType*)*/ onnxtype);

        public static DOrtGetOnnxTypeFromTypeInfo OrtGetOnnxTypeFromTypeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetValueCount(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(size_t*)*/ count);

        public static DOrtGetValueCount OrtGetValueCount;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr/*(OrtStatus*)*/ DOrtCreateValue(IntPtr[] /* const OrtValue* const* in */ values,
            UIntPtr /* size_t */ num_values, IntPtr /* (OnnxValueType */ onnxValueType, out IntPtr /* OrtValue** */ ortValue);

        public static DOrtCreateValue OrtCreateValue;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTypeInfo(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(OrtValue**)*/ typeInfo);

        public static DOrtGetTypeInfo OrtGetTypeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreateTensorAsOrtValue(
                        IntPtr /*_Inout_ OrtAllocator* */ allocator,
                        long[] /*_In_ const int64_t* */ shape,
                        UIntPtr /*size_t*/ shape_len,
                        Tensors.TensorElementType type,
                        out IntPtr /* OrtValue** */ outputValue);

        public static DOrtCreateTensorAsOrtValue OrtCreateTensorAsOrtValue;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus */ DOrtCreateTensorWithDataAsOrtValue(
                                                        IntPtr /* (const OrtMemoryInfo*) */ allocatorInfo,
                                                        IntPtr /* (void*) */dataBufferHandle,
                                                        UIntPtr dataLength,
                                                        long[] shape,
                                                        UIntPtr shapeLength,
                                                        Tensors.TensorElementType type,
                                                        out IntPtr /* OrtValue** */ outputValue);

        public static DOrtCreateTensorWithDataAsOrtValue OrtCreateTensorWithDataAsOrtValue;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus */ DOrtValueIsTensor(IntPtr /*(OrtValue*)*/ ortValue, out IntPtr val);

        public static DOrtValueIsTensor OrtValueIsTensor;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* OrtStatus */ DOrtValueIsSparseTensor(IntPtr /*(OrtValue*)*/ ortValue, out IntPtr val);

        public static DOrtValueIsSparseTensor OrtValueIsSparseTensor;

        /// This function doesn't work with string tensor
        /// this is a no-copy method whose pointer is only valid until the backing OrtValue* is free'd.
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorMutableData(IntPtr /*(OrtValue*)*/ value, out IntPtr /* (void**)*/ dataBufferHandle);

        public static DOrtGetTensorMutableData OrtGetTensorMutableData;

        /// \param value A tensor created from OrtCreateTensor... function.
        /// \param len total data length, not including the trailing '\0' chars.
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtFillStringTensor(
                                                        IntPtr /* OrtValue */ value,
                                                        IntPtr[] /* const char* const* */s,
                                                        UIntPtr /* size_t */ s_len);

        public static DOrtFillStringTensor OrtFillStringTensor;

        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetResizedStringTensorElementBuffer(
                IntPtr /* OrtValue */ value,
                UIntPtr /* size_t */ index,
                UIntPtr /* size_t */ length_in_bytes,
                out IntPtr /* char** */ buffer
            );

        public static DOrtGetResizedStringTensorElementBuffer OrtGetResizedStringTensorElementBuffer;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorContent(
                                                        IntPtr /*(OrtValue*)*/ value,
                                                        byte[] /*(void*)*/  dst_buffer,
                                                        UIntPtr dst_buffer_len,
                                                        UIntPtr[] offsets,
                                                        UIntPtr offsets_len);

        public static DOrtGetStringTensorContent OrtGetStringTensorContent;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorDataLength(IntPtr /*(OrtValue*)*/ value,
                                                        out UIntPtr /*(size_t*)*/ len);

        public static DOrtGetStringTensorDataLength OrtGetStringTensorDataLength;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorElementLength(IntPtr /*(OrtValue*)*/ value,
                                                        UIntPtr /*(size_t)*/ index,
                                                        out UIntPtr /*(size_t*)*/ len);

        public static DOrtGetStringTensorElementLength OrtGetStringTensorElementLength;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetStringTensorElement(IntPtr /*(OrtValue*)*/ value,
                                                UIntPtr /*(size_t)*/ bufferLength,
                                                UIntPtr /*(size_t)*/ elementIndex,
                                                byte[] buffer);

        public static DOrtGetStringTensorElement OrtGetStringTensorElement;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/
                                DOrtCastTypeInfoToTensorInfo(IntPtr /*(struct OrtTypeInfo*)*/ typeInfo, out IntPtr /*(const struct OrtTensorTypeAndShapeInfo**)*/ typeAndShapeInfo);

        public static DOrtCastTypeInfoToTensorInfo OrtCastTypeInfoToTensorInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorTypeAndShape(IntPtr /*(OrtValue*)*/ value, out IntPtr /*(struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo);

        public static DOrtGetTensorTypeAndShape OrtGetTensorTypeAndShape;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseTensorTypeAndShapeInfo(IntPtr /*(OrtTensorTypeAndShapeInfo*)*/ value);

        public static DOrtReleaseTensorTypeAndShapeInfo OrtReleaseTensorTypeAndShapeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorElementType(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, out IntPtr /*(TensorElementType*)*/ output);

        public static DOrtGetTensorElementType OrtGetTensorElementType;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetDimensionsCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo, out UIntPtr output);

        public static DOrtGetDimensionsCount OrtGetDimensionsCount;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
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
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
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
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorShapeElementCount(IntPtr /*(const struct OrtTensorTypeAndShapeInfo*)*/ typeAndShapeInfo,
            out UIntPtr /* size_t */ output);

        public static DOrtGetTensorShapeElementCount OrtGetTensorShapeElementCount;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        // The out ortMemoryInfo must not be destroyed/deallocated. The pointer points to an object owned by
        // the contained Tensor/SparseTensor.
        public delegate IntPtr /*(OrtStatus*)*/ DOrtGetTensorMemoryInfo(IntPtr /* const OrtValue* */ ortValue,
            out IntPtr /* const OrtMemoryInfo** */ ortMemoryInfo);

        public static DOrtGetTensorMemoryInfo OrtGetTensorMemoryInfo;

        ///  Map Type API
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DCastTypeInfoToMapTypeInfo(IntPtr /*(const struct OrtTypeInfo*)*/ typeInfo, out IntPtr /*const OrtMapTypeInfo** */ mapTypeInfo);

        public static DCastTypeInfoToMapTypeInfo OrtCastTypeInfoToMapTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DGetMapKeyType(IntPtr /*const OrtMapTypeInfo* */ mapTypeInfo, out IntPtr /*(TensorElementType*)*/ tensorElementType);

        public static DGetMapKeyType OrtGetMapKeyType;

        public delegate IntPtr /*(OrtStatus*)*/ DGetMapValueType(IntPtr /* const OrtMapTypeInfo* */ map_type_info, out IntPtr /* OrtTypeInfo** */ type_info);

        public static DGetMapValueType OrtGetMapValueType;

        // Sequence TypeInfo
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DCastTypeInfoToSequenceTypeInfo(IntPtr /*(struct OrtTypeInfo*)*/ typeInfo, out IntPtr /* const OrtSequenceTypeInfo** */ sequenceTypeInfo);

        public static DCastTypeInfoToSequenceTypeInfo OrtCastTypeInfoToSequenceTypeInfo;

        public delegate IntPtr /*(OrtStatus*)*/ DGetSequenceElementType(IntPtr /* const OrtSequenceTypeInfo* */ sequenceTypeInfo, out IntPtr /* OrtTypeInfo** */ elementTypeInfo);

        public static DGetSequenceElementType OrtGetSequenceElementType;

        // OptionalTypeInfo
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/  DOrtCastTypeInfoToOptionalTypeInfo(IntPtr /*(struct OrtTypeInfo*)*/ typeInfo, out IntPtr /* const struct OrtOptionalTypeInfo** */  optionalTypeInfo);

        public static DOrtCastTypeInfoToOptionalTypeInfo OrtCastTypeInfoToOptionalTypeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DGetOptionalContainedTypeInfo(IntPtr /* const struct OrtOptionalTypeInfo*/ optTypeInfo, out IntPtr /* struct OrtTypeInfo** */ containedTypeInfo);

        public static DGetOptionalContainedTypeInfo OrtGetOptionalContainedTypeInfo;

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleaseValue(IntPtr /*(OrtValue*)*/ value);

        public static DOrtReleaseValue OrtReleaseValue;

        #endregion


        #region Misc API

        /// <summary>
        /// Queries all the execution providers supported in the native onnxruntime shared library
        /// </summary>
        /// <param name="providers">(output) all execution providers (strings) supported in the native onnxruntime shared library</param>
        /// <param name="numProviders">(output) number of execution providers (strings)</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtGetAvailableProviders(out IntPtr /* (char***) */ providers, out int /* (int*) */ numProviders);

        public static DOrtGetAvailableProviders OrtGetAvailableProviders;

        /// <summary>
        /// Releases all execution provider strings allocated and returned by OrtGetAvailableProviders
        /// </summary>
        /// <param name="providers">all execution providers (strings) returned by OrtGetAvailableProviders</param>
        /// <param name="numProviders">number of execution providers (strings)</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /* (OrtStatus*) */ DOrtReleaseAvailableProviders(IntPtr /* (char**) */ providers, int /* (int) */ numProviders);

        public static DOrtReleaseAvailableProviders OrtReleaseAvailableProviders;

        /// <summary>
        /// Create an instance of PrepackedWeightsContainer
        /// </summary>
        /// <param name="prepackedWeightsContainer">(output) Created native OrtPrepackedWeightsContainer instance</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr /*(OrtStatus*)*/ DOrtCreatePrepackedWeightsContainer(out IntPtr /*(OrtPrepackedWeightsContainer**)*/ prepackedWeightsContainer);

        public static DOrtCreatePrepackedWeightsContainer OrtCreatePrepackedWeightsContainer;

        /// <summary>
        /// Destroy an instance of PrepackedWeightsContainer
        /// </summary>
        /// <param name="prepackedWeightsContainer">Native OrtPrepackedWeightsContainer instance to be destroyed</param>
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void DOrtReleasePrepackedWeightsContainer(IntPtr /*(OrtPrepackedWeightsContainer*)*/ prepackedWeightsContainer);

        public static DOrtReleasePrepackedWeightsContainer OrtReleasePrepackedWeightsContainer;

        #endregion
    } //class NativeMethods

    // onnxruntime-extensions helpers to make usage simpler.
    // The onnxruntime-extensions nuget package containing the native library can be optionally added to the app.
    // If added, SessionOptions.RegisterOrtExtensions can be called to add the custom ops to the session options.
    // We handle the DllImport and platform specific aspects so the user code doesn't require that.
    // adjust the library name based on platform.
    internal static class OrtExtensionsNativeMethods
    {
#if __ANDROID__
        internal const string ExtensionsDllName = "libortextensions.so";
#elif __IOS__
        internal const string ExtensionsDllName = "__Internal";
#else
        internal const string ExtensionsDllName = "ortextensions";
#endif

        [DllImport(ExtensionsDllName, CharSet = CharSet.Ansi,
                   CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OrtStatus* */ RegisterCustomOps(IntPtr /* OrtSessionOptions* */ sessionOptions,
                                                                       ref OrtApiBase /* OrtApiBase* */ ortApiBase);


    }
} //namespace
