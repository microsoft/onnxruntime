// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include "winrt/windows.foundation.collections.h"

/**
 * All APIs exported by winml_adapter_c_api.h are part of the private interface dedicated to supporting the WinML API.
 * This contract is subject to change based on the needs of the WinML API and is not intended for direct use by callers 
 * of the onnxruntime c-api and usage of APIs in this header are *not* supported by the onnxruntime product.
  */

ORT_RUNTIME_CLASS(Model);
ORT_RUNTIME_CLASS(ExecutionProvider);

struct WinmlAdapterApi;
typedef struct WinmlAdapterApi WinmlAdapterApi;

struct ID3D12Resource;
struct ID3D12Device;
struct ID3D12CommandQueue;
struct IMLOperatorRegistry;

// TODO: Must match onnxruntime::profiling::EventRecord
enum OrtProfilerEventCategory {
  SESSION_EVENT = 0,
  NODE_EVENT,
  EVENT_CATEGORY_MAX
};

struct OrtProfilerEventRecord {
  OrtProfilerEventCategory category_;
  const char* category_name_;
  int64_t duration_;
  int64_t time_span_;
  const char* event_name_;
  int32_t process_id_;
  int32_t thread_id_;
  const char* op_name_;
  const char* execution_provider_;
};

typedef void(ORT_API_CALL* OrtProfilingFunction)(const OrtProfilerEventRecord* event_record);

struct WinmlAdapterApi {
  /**
    * OverrideSchema
	 * This api is used to override schema inference functions for a variety of ops across opsets.
	 * This exists because certain ops were failing to infer schemas and caused performance
	 * issues for DML as it was forced to create resources during evaluation.
	 * This can be removed when schema inference functions have been updated.
    */
  OrtStatus*(ORT_API_CALL* OverrideSchema)() NO_EXCEPTION;

  /**
     * EnvConfigureCustomLoggerAndProfiler
	 * This api is used to add a custom logger and profiler to the ors environment.
	 * This exists because existing methods on the c-abi to create the environment only support a custom logger.
	 * Since WinML hooks the profiler events, we expose the profiler and an associated profiling function.
    */
  OrtStatus*(ORT_API_CALL* EnvConfigureCustomLoggerAndProfiler)(_In_ OrtEnv* env, OrtLoggingFunction logging_function, OrtProfilingFunction profiling_function, _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid, _Outptr_ OrtEnv** out)NO_EXCEPTION;

  // OrtModel methods

  /**
    * CreateModelFromPath
	 * This api creates an OrtModel based on a specified model path.
	 * There is no inferencing or evaluation setup performed. Only ONNX load is done to reflect on the model's inputs/outputs and other properties.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* CreateModelFromPath)(_In_ const char* model_path, _In_ size_t size, _Outptr_ OrtModel** out)NO_EXCEPTION;

  /**
    * CreateModelFromData
	 * This api creates an OrtModel from a buffer.
	 * There is no inferencing or evaluation setup performed. Only ONNX load is done to reflect on the model's inputs/outputs and other properties.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* CreateModelFromData)(_In_ void* data, _In_ size_t size, _Outptr_ OrtModel** out)NO_EXCEPTION;

  /**
    * CloneModel
	 * This api copies the OrtModel along with its internal proto buffer and cached metadata.
	 * The OrtSession type expects to own the model proto buffer.
	 * WinML uses this to yield copies of the model proto held by OrtModel to OrtSession.
    */
  OrtStatus*(ORT_API_CALL* CloneModel)(_In_ const OrtModel* in, _Outptr_ OrtModel** out)NO_EXCEPTION;

  /**
    * ModelGetAuthor
	 * This api gets the model author from the OrtModel.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetAuthor)(_In_ const OrtModel* model, _Out_ const char** const author, _Out_ size_t* len)NO_EXCEPTION;

  /**
    * ModelGetName
	 * This api gets the model name from the OrtModel.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetName)(_In_ const OrtModel* model, _Out_ const char** const name, _Out_ size_t* len)NO_EXCEPTION;

  /**
    * ModelGetDomain
	 * This api gets the model domain from the OrtModel.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetDomain)(_In_ const OrtModel* model, _Out_ const char** const domain, _Out_ size_t* len)NO_EXCEPTION;

  /**
    * ModelGetDescription
	 * This api gets the model description from the OrtModel.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetDescription)(_In_ const OrtModel* model, _Out_ const char** const description, _Out_ size_t* len)NO_EXCEPTION;

  /**
    * ModelGetVersion
	 * This api gets the model version from the OrtModel.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetVersion)(_In_ const OrtModel* model, _Out_ int64_t* version)NO_EXCEPTION;

  /**
    * ModelGetInputCount
	 * This api gets the number of inputs from the OrtModel. It closely matches the API of a similar name similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetInputCount)(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;

  /**
    * ModelGetOutputCount
	 * This api gets the number of outputs from the OrtModel. It closely matches the API of a similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetOutputCount)(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;

  /**
    * ModelGetInputName
	 * This api gets the input name from the OrtModel given an index. It closely matches the API of a similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetInputName)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** input_name, _Out_ size_t* count)NO_EXCEPTION;

  /**
    * ModelGetOutputName
	 * This api gets the output name from the OrtModel given an index. It closely matches the API of a similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetOutputName)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** output_name, _Out_ size_t* count)NO_EXCEPTION;

  /**
    * ModelGetInputDescription
	 * This api gets the input description from the OrtModel given an index. It closely matches the API of a similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetInputDescription)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** input_description, _Out_ size_t* count)NO_EXCEPTION;

  /**
    * ModelGetOutputDescription
	 * This api gets the output description from the OrtModel given an index. It closely matches the API of a similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetOutputDescription)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** output_description, _Out_ size_t* count)NO_EXCEPTION;

  /**
    * ModelGetInputTypeInfo
	 * This api gets the input OrtTypeInfo from the OrtModel given an index. It closely matches the API of a similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetInputTypeInfo)(_In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;

  /**
    * ModelGetOutputTypeInfo
	 * This api gets the output OrtTypeInfo from the OrtModel given an index. It closely matches the API of a similar name for retrieving model metadata from OrtSession.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetOutputTypeInfo)(_In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;

  /**
    * ModelGetMetadataCount
	 * This api gets the number of metadata entries from the OrtModel.
	 * This is used by WinML to support model reflection APIs.
    */
  OrtStatus*(ORT_API_CALL* ModelGetMetadataCount)(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;

  /**
    * ModelGetMetadata
	 * This api gets the model metadata from the OrtModel.
	 * This is used by WinML to deduce whether model input and output formats are supported by the WinML tensorization code paths.
    */
  OrtStatus*(ORT_API_CALL* ModelGetMetadata)(_In_ const OrtModel* model, _Out_ size_t count, _Out_ const char** const key, _Out_ size_t* key_len, _Out_ const char** const value, _Out_ size_t* value_len)NO_EXCEPTION;

  /**
    * ModelEnsureNoFloat16
	 * This api checks whether the model requires float 16 support.
	 * This is used by WinML to fail gracefully when float 16 support is not available on the device.
    * 
    * Can this API be moved into the EP during session initialization. Currently we do an early fp16 check to avoid initialization when it is not supported.
    */
  OrtStatus*(ORT_API_CALL* ModelEnsureNoFloat16)(_In_ const OrtModel* model)NO_EXCEPTION;

  // OrtSessionOptions methods

  /**
    * OrtSessionOptionsAppendExecutionProvider_CPU
	 * This api is used to add the cpu EP to OrtSessionOptions so that WinML Gpu session are configures with CPU fallback.
    */
  OrtStatus*(ORT_API_CALL* OrtSessionOptionsAppendExecutionProvider_CPU)(_In_ OrtSessionOptions* options, int use_arena)NO_EXCEPTION;

  /**
    * OrtSessionOptionsAppendExecutionProvider_DML
	 * This api is used to add the DML EP to OrtSessionOptions.
    */
  OrtStatus*(ORT_API_CALL* OrtSessionOptionsAppendExecutionProvider_DML)(_In_ OrtSessionOptions* options, ID3D12Device* device, ID3D12CommandQueue* queue, bool metacommands_enabled)NO_EXCEPTION;

  // OrtSession methods

  /**
    * CreateSessionWithoutModel
	 * This api is used to create a Session that is completely uninitialized. While there are other Session creation APIs in the
    * c-abi, WinML uses this so that it can perform optimizations prior to loading the model, and initializing.
    * Moreover, WinML needs a new api to support the OrtModel type, and prevent the parsing model protobufs again on session creation. 
    */
  OrtStatus*(ORT_API_CALL* CreateSessionWithoutModel)(_In_ OrtEnv* env, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session)NO_EXCEPTION;

  /**
    * SessionGetExecutionProvider
	 * This api is used to get a handle to an OrtExecutionProvider.
    * Currently WinML uses this to talk directly to the DML EP and configure settings on it.
    */
  OrtStatus*(ORT_API_CALL* SessionGetExecutionProvider)(_In_ OrtSession* session, _In_ size_t index, _Out_ OrtExecutionProvider** provider)NO_EXCEPTION;

  /**
    * SessionInitialize
	 * This api is used to initialize an OrtSession. This is one component of creating a usable OrtSession, and is a part of CreateSession in the c-abi.
    * Currently WinML uses this to finalize session creation, after configuring a variety of properties on the OrtSession.
    */
  OrtStatus*(ORT_API_CALL* SessionInitialize)(_In_ OrtSession* session)NO_EXCEPTION;

  /**
    * SessionRegisterGraphTransformers
	 * This api is used to enable DML specific graph transformations on an OrtSession.
    * 
    * Ideally these transformations should be configured by the contract between the runtime and the EP and not overridden by WinML. 
    */
  OrtStatus*(ORT_API_CALL* SessionRegisterGraphTransformers)(_In_ OrtSession* session)NO_EXCEPTION;

  /**
    * SessionRegisterCustomRegistry
	 * This api is used to support custom operators as they were shipped in WinML RS5.
    */
  OrtStatus*(ORT_API_CALL* SessionRegisterCustomRegistry)(_In_ OrtSession* session, _In_ IMLOperatorRegistry* registry)NO_EXCEPTION;

  /**
    * SessionLoadAndPurloinModel
	 * This api is used to load an OrtModel into an OrtSession.
    * 
 	 * Don't free the 'out' value as this API will defunct and release the OrtModel internally.
    */
  OrtStatus*(ORT_API_CALL* SessionLoadAndPurloinModel)(_In_ OrtSession* session, _In_ OrtModel* model)NO_EXCEPTION;

  /**
    * SessionStartProfiling
	 * This api is used to start profiling OrtSession. The existing mechanism only allows configuring profiling at session creation.
    * 
 	 * WinML uses this to toggle profilling on and off based on if a telemetry providers are being listened to.
    */
  OrtStatus*(ORT_API_CALL* SessionStartProfiling)(_In_ OrtEnv* env, _In_ OrtSession* session)NO_EXCEPTION;

  /**
    * SessionEndProfiling
	 * This api is used to end profiling OrtSession. The existing mechanism only allows configuring profiling at session creation.
    * 
 	 * WinML uses this to toggle profilling on and off based on if a telemetry providers are being listened to.
    */
  OrtStatus*(ORT_API_CALL* SessionEndProfiling)(_In_ OrtSession* session)NO_EXCEPTION;

  /**
    * SessionCopyOneInputAcrossDevices
	 * This api is used to copy and create an OrtValue input to prepare the input on the correct device.
    * 
 	 * WinML uses this to copy gpu device OrtValues to the CPU and vice-versa.
    */
  OrtStatus*(ORT_API_CALL* SessionCopyOneInputAcrossDevices)(_In_ OrtSession* session, _In_ const char* const input_name, _In_ OrtValue* orig_value, _Outptr_ OrtValue** new_value)NO_EXCEPTION;

  // Dml methods (TODO need to figure out how these need to move to session somehow...)

    /**
    * SessionGetNumberOfIntraOpThreads
     * This api returns the number of intra operator threads set on the OrtSession.
    *
    * WinML uses this to determine that the correct number of threads was set correctly through OrtSessionOptions.
    */
  OrtStatus*(ORT_API_CALL* SessionGetNumberOfIntraOpThreads)(_In_ OrtSession* session, _Out_ uint32_t* num_threads)NO_EXCEPTION;

      /**
    * SessionGetNamedDimensionsOverrides
     * This api returns the named dimension overrides that are specified for this session
    *
    * WinML uses this to determine that named dimension overrides were set correctly through OrtSessionOptions.
    */
  OrtStatus*(ORT_API_CALL* SessionGetNamedDimensionsOverrides)(_In_ OrtSession* session, _Out_ winrt::Windows::Foundation::Collections::IMapView<winrt::hstring, uint32_t>& overrides)NO_EXCEPTION;

  /**
    * DmlExecutionProviderSetDefaultRoundingMode
	 * This api is used to configure the DML EP to turn on/off rounding.
    * 
 	 * WinML uses this to disable rounding during session initialization and then enables it again post initialization.
    */
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderSetDefaultRoundingMode)(_In_ OrtExecutionProvider* dml_provider, _In_ bool is_enabled)NO_EXCEPTION;

  /**
    * DmlExecutionProviderFlushContext
	 * This api is used to flush the DML EP.
    * 
    * WinML communicates directly with DML to perform this as an optimization.
    */
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderFlushContext)(_In_ OrtExecutionProvider* dml_provider)NO_EXCEPTION;

  /**
    * DmlExecutionProviderReleaseCompletedReferences
	 * This api is used to release completed references after first run the DML EP.
    * 
    * WinML communicates directly with DML to perform this as an optimization.
    */
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderReleaseCompletedReferences)(_In_ OrtExecutionProvider* dml_provider)NO_EXCEPTION;

  /**
    * DmlCreateGPUAllocationFromD3DResource
	 * This api is used to create a DML EP input based on a user specified d3d12 resource.
    * 
    * WinML uses this as part of its Tensor apis to allow callers to specify their own D3D12 resources as inputs/outputs.
    */
  OrtStatus*(ORT_API_CALL* DmlCreateGPUAllocationFromD3DResource)(_In_ ID3D12Resource* pResource, _Out_ void** dml_resource)NO_EXCEPTION;

  /**
    * DmlFreeGPUAllocation
	 * This api is used free the DML EP input created by DmlCreateGPUAllocationFromD3DResource.
    * 
    * WinML uses this as part of its Tensor apis to allow callers to specify their own D3D12 resources as inputs/outputs.
    */
  OrtStatus*(ORT_API_CALL* DmlFreeGPUAllocation)(_In_ void* ptr)NO_EXCEPTION;

  /**
    * DmlGetD3D12ResourceFromAllocation
	 * This api is used to get the D3D12 resource when a OrtValue has been allocated by the DML EP and accessed via GetMutableTensorData.
    * 
    * WinML uses this in the image feature path to get the d3d resource and perform and tensorization on inputs directly into the allocated d3d12 resource.
    */
  OrtStatus*(ORT_API_CALL* DmlGetD3D12ResourceFromAllocation)(_In_ OrtExecutionProvider* provider, _In_ void* allocation, _Out_ ID3D12Resource** resource)NO_EXCEPTION;

  /**
    * DmlCopyTensor
	 * This api is used copy a tensor allocated by the DML EP Allocator to the CPU.
    * 
    * WinML uses this when graphs are evaluated with DML, and their outputs remain on the GPU but need to be copied back to the CPU.
    */
  OrtStatus*(ORT_API_CALL* DmlCopyTensor)(_In_ OrtExecutionProvider* provider, _In_ OrtValue* src, _In_ OrtValue* dst)NO_EXCEPTION;

  /**
    * GetProviderMemoryInfo
	 * This api gets the memory info object associated with an EP.
    * 
    * WinML uses this to manage caller specified D3D12 inputs/outputs. It uses the memory info here to call DmlCreateGPUAllocationFromD3DResource.
    */
  OrtStatus*(ORT_API_CALL* GetProviderMemoryInfo)(_In_ OrtExecutionProvider* provider, OrtMemoryInfo** memory_info)NO_EXCEPTION;

  /**
    * GetProviderAllocator
	 * This api gets associated allocator used by a provider.
    * 
    * WinML uses this to create tensors, and needs to hold onto the allocator for the duration of the associated value's lifetime.
    */
  OrtStatus*(ORT_API_CALL* GetProviderAllocator)(_In_ OrtExecutionProvider* provider, OrtAllocator** allocator)NO_EXCEPTION;

  /**
    * FreeProviderAllocator
	 * This api frees an allocator.
    * 
    * WinML uses this to free the associated allocator for an ortvalue when creating tensors.
    * Internally this derefs a shared_ptr.
    */
  OrtStatus*(ORT_API_CALL* FreeProviderAllocator)(_In_ OrtAllocator* allocator)NO_EXCEPTION;

  /**
    * GetValueMemoryInfo
	 * This api gets the memory info of an OrtValue.
    * 
    * WinML uses this to determine if an OrtValue is allocated on the Cpu or elsewhere.
    */
  OrtStatus*(ORT_API_CALL* GetValueMemoryInfo)(const OrtValue* value, OrtMemoryInfo** memory_info)NO_EXCEPTION;

  /**
    * ExecutionProviderSync
	 * This api syncs the EP.
    * 
    * WinML uses this to sync EP inputs/outputs directly.
    */
  OrtStatus*(ORT_API_CALL* ExecutionProviderSync)(_In_ OrtExecutionProvider* provider)NO_EXCEPTION;

  /**
    * CreateCustomRegistry
	 * This api creates a custom registry that callers can populate with custom ops.
    *
    * WinML uses this to support custom ops.
    */
  OrtStatus*(ORT_API_CALL* CreateCustomRegistry)(_Out_ IMLOperatorRegistry** registry)NO_EXCEPTION;

  /**
    * ValueGetDeviceId
	 * This api returns the device id of the OrtValue.
    * 
    * WinML uses this to determine if an OrtValue is created on the needed device.
    */
  OrtStatus*(ORT_API_CALL* ValueGetDeviceId)(_In_ OrtValue* ort_value, _Out_ int16_t* device_id)NO_EXCEPTION;

  /**
    * SessionGetInputRequiredDeviceId
	 * This api returns the required device id for a model input.
    * 
    * WinML uses this to determine if an OrtValue is created on the needed device.
    */
  OrtStatus*(ORT_API_CALL* SessionGetInputRequiredDeviceId)(_In_ OrtSession* session, _In_ const char* const input_name, _Out_ int16_t* device_id)NO_EXCEPTION;

  ORT_CLASS_RELEASE(Model);
};
