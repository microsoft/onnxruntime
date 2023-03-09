// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"

//
// Stub out the API with default implementations that do nothing.
//
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"

namespace {
const NnApi LoadNnApi() {
  NnApi nnapi = {};
  nnapi.android_sdk_version = ORT_NNAPI_MAX_SUPPORTED_API_LEVEL;
  nnapi.nnapi_runtime_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_7;

  nnapi.ANeuralNetworksMemory_createFromFd =
      [](size_t, int, int, size_t, ANeuralNetworksMemory**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksMemory_free = [](ANeuralNetworksMemory*) {};
  nnapi.ANeuralNetworksModel_create =
      [](ANeuralNetworksModel**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksModel_free = [](ANeuralNetworksModel*) {};
  nnapi.ANeuralNetworksModel_finish = [](ANeuralNetworksModel*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksModel_addOperand = [](ANeuralNetworksModel*, const ANeuralNetworksOperandType*) {
    return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
  };
  nnapi.ANeuralNetworksModel_setOperandValue = [](ANeuralNetworksModel*, int32_t, const void*, size_t) {
    return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
  };
  nnapi.ANeuralNetworksModel_setOperandSymmPerChannelQuantParams =
      [](ANeuralNetworksModel*, int32_t, const ANeuralNetworksSymmPerChannelQuantParams*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksModel_setOperandValueFromMemory =
      [](ANeuralNetworksModel*, int32_t, const ANeuralNetworksMemory*, size_t, size_t) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksModel_addOperation =
      [](ANeuralNetworksModel*, ANeuralNetworksOperationType, uint32_t, const uint32_t*, uint32_t, const uint32_t*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksModel_identifyInputsAndOutputs =
      [](ANeuralNetworksModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
      [](ANeuralNetworksModel*, bool) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksCompilation_create =
      [](ANeuralNetworksModel*, ANeuralNetworksCompilation**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksCompilation_free = [](ANeuralNetworksCompilation*) {};
  nnapi.ANeuralNetworksCompilation_setPreference =
      [](ANeuralNetworksCompilation*, int32_t) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksCompilation_finish =
      [](ANeuralNetworksCompilation*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_create =
      [](ANeuralNetworksCompilation*, ANeuralNetworksExecution**) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksExecution_free = [](ANeuralNetworksExecution*) {};
  nnapi.ANeuralNetworksExecution_setInput =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*, const void*, size_t) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksExecution_setInputFromMemory =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*, const ANeuralNetworksMemory*, size_t,
         size_t) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksExecution_setOutput =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*, void*, size_t) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksExecution_setOutputFromMemory =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*, const ANeuralNetworksMemory*, size_t,
         size_t) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksExecution_startCompute =
      [](ANeuralNetworksExecution*, ANeuralNetworksEvent**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksEvent_wait = [](ANeuralNetworksEvent*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksEvent_free = [](ANeuralNetworksEvent*) {};
  nnapi.ASharedMemory_create = [](const char*, size_t) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworks_getDeviceCount = [](uint32_t*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworks_getDevice =
      [](uint32_t, ANeuralNetworksDevice**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice*, const char**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksDevice_getVersion =
      [](const ANeuralNetworksDevice*, const char**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice*, int64_t* feature_level) {
        *feature_level = ANEURALNETWORKS_FEATURE_LEVEL_7;
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksDevice_getType =
      [](const ANeuralNetworksDevice*, int32_t*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel*, const ANeuralNetworksDevice* const*, uint32_t, bool*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksCompilation_createForDevices =
      [](ANeuralNetworksModel*, const ANeuralNetworksDevice* const*, uint32_t, ANeuralNetworksCompilation**) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksCompilation_setCaching =
      [](ANeuralNetworksCompilation*, const char*, const uint8_t*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksCompilation_setTimeout =
      [](ANeuralNetworksCompilation*, uint64_t) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksCompilation_setPriority =
      [](ANeuralNetworksCompilation*, int) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_compute =
      [](ANeuralNetworksExecution*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_setTimeout =
      [](ANeuralNetworksExecution*, uint64_t) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_setLoopTimeout =
      [](ANeuralNetworksExecution*, uint64_t) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_getOutputOperandRank =
      [](ANeuralNetworksExecution*, int32_t, uint32_t*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_getOutputOperandDimensions =
      [](ANeuralNetworksExecution*, int32_t, uint32_t*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksBurst_create =
      [](ANeuralNetworksCompilation*, ANeuralNetworksBurst**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksBurst_free = [](ANeuralNetworksBurst*) {};
  nnapi.ANeuralNetworksExecution_burstCompute =
      [](ANeuralNetworksExecution*, ANeuralNetworksBurst*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksMemory_createFromAHardwareBuffer =
      [](const AHardwareBuffer*, ANeuralNetworksMemory**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_setMeasureTiming =
      [](ANeuralNetworksExecution*, bool) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_getDuration =
      [](const ANeuralNetworksExecution*, int32_t, uint64_t*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksDevice_getExtensionSupport =
      [](const ANeuralNetworksDevice*, const char*, bool*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksModel_getExtensionOperandType =
      [](ANeuralNetworksModel*, const char*, uint16_t, int32_t*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksModel_getExtensionOperationType =
      [](ANeuralNetworksModel*, const char*, uint16_t, ANeuralNetworksOperationType*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksModel_setOperandExtensionData =
      [](ANeuralNetworksModel*, int32_t, const void*, size_t) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksMemoryDesc_create =
      [](ANeuralNetworksMemoryDesc**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksMemoryDesc_free = [](ANeuralNetworksMemoryDesc*) {};
  nnapi.ANeuralNetworksMemoryDesc_addInputRole =
      [](ANeuralNetworksMemoryDesc*, const ANeuralNetworksCompilation*, uint32_t, float) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksMemoryDesc_addOutputRole =
      [](ANeuralNetworksMemoryDesc*, const ANeuralNetworksCompilation*, uint32_t, float) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksMemoryDesc_setDimensions =
      [](ANeuralNetworksMemoryDesc*, uint32_t, const uint32_t*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksMemoryDesc_finish =
      [](ANeuralNetworksMemoryDesc*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksMemory_createFromDesc =
      [](const ANeuralNetworksMemoryDesc*, ANeuralNetworksMemory**) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksMemory_copy =
      [](const ANeuralNetworksMemory*, const ANeuralNetworksMemory*) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksEvent_createFromSyncFenceFd =
      [](int, ANeuralNetworksEvent**) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksEvent_getSyncFenceFd =
      [](const ANeuralNetworksEvent*, int*) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_startComputeWithDependencies =
      [](ANeuralNetworksExecution*, const ANeuralNetworksEvent* const*, uint32_t, uint64_t, ANeuralNetworksEvent**) {
        return static_cast<int>(ANEURALNETWORKS_NO_ERROR);
      };
  nnapi.ANeuralNetworksExecution_enableInputAndOutputPadding =
      [](ANeuralNetworksExecution*, bool) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworksExecution_setReusable =
      [](ANeuralNetworksExecution*, bool) { return static_cast<int>(ANEURALNETWORKS_NO_ERROR); };
  nnapi.ANeuralNetworks_getRuntimeFeatureLevel = []() { return int64_t(ANEURALNETWORKS_FEATURE_LEVEL_7); };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> int32_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> int64_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> const uint8_t* { return nullptr; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> const char* { return ""; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> int32_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> ANeuralNetworksDiagnosticDataClass { return {}; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> ANeuralNetworksDiagnosticDataClass { return {}; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> uint64_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> bool { return false; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> bool { return false; };
  nnapi.SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> bool { return false; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> int32_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> int64_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> const uint8_t* { return nullptr; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> const char* { return ""; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> ANeuralNetworksDiagnosticExecutionMode { return {}; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> ANeuralNetworksDiagnosticDataClass { return {}; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> ANeuralNetworksDiagnosticDataClass { return {}; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint32_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint64_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint64_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint64_t { return 0; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> bool { return false; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> bool { return false; };
  nnapi.SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> bool { return false; };
  nnapi.SL_ANeuralNetworksDiagnostic_registerCallbacks =
      [](ANeuralNetworksDiagnosticCompilationFinishedCallback,
         ANeuralNetworksDiagnosticExecutionFinishedCallback,
         void*) -> void {};

  return nnapi;
}
}  // namespace

const NnApi* NnApiImplementation() {
  static const NnApi nnapi = LoadNnApi();
  return &nnapi;
}
