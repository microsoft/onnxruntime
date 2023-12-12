// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/provider_options.h"
#include "core/providers/provider_factory_creators.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

using namespace onnxruntime;

namespace {

OrtStatus* ParseProviderOptions(_In_reads_(num_keys) const char* const* provider_options_keys,
                                _In_reads_(num_keys) const char* const* provider_options_values,
                                _In_ size_t num_keys,
                                ProviderOptions& provider_options) {
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Provider options key/value cannot be empty");
    }

    // arbitrary length to validate the key/value. adjust if/when needed.
    // TODO: are any other input validation checks required here (and in the other functions that process
    // provider options)?
    if (strlen(provider_options_keys[i]) > 1024 || strlen(provider_options_values[i]) > 1024) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Maximum string length for a provider options key/value is 1024.");
    }

    provider_options[provider_options_keys[i]] = provider_options_values[i];
  }

  return nullptr;
}
}  // namespace
/**
 * Implementation of OrtApis functions for provider registration.
 *
 * EPs that use the provider bridge are handled in provider_bridge_ort.cc
 */

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* provider_name,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    _In_ size_t num_keys) {
  API_IMPL_BEGIN
  ProviderOptions provider_options;
  OrtStatus* status = ParseProviderOptions(provider_options_keys,
                                           provider_options_values,
                                           num_keys,
                                           provider_options);
  if (status != nullptr) {
    return status;
  }

  auto create_not_supported_status = [&provider_name]() {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 (std::string(provider_name) + " execution provider is not supported in this build. ").c_str());
  };

  if (strcmp(provider_name, "DML") == 0) {
#if defined(USE_DML)
    options->provider_factories.push_back(DMLProviderFactoryCreator::CreateFromProviderOptions(provider_options));
#else
    status = create_not_supported_status();
#endif
  } else if (strcmp(provider_name, "QNN") == 0) {
#if defined(USE_QNN)
    options->provider_factories.push_back(QNNProviderFactoryCreator::Create(provider_options, &(options->value)));
#else
    status = create_not_supported_status();
#endif
  } else if (strcmp(provider_name, "OpenVINO") == 0) {
#if defined(USE_OPENVINO)
    options->provider_factories.push_back(OpenVINOProviderFactoryCreator::Create(&provider_options));
#else
    status = create_not_supported_status();
#endif

  } else if (strcmp(provider_name, "SNPE") == 0) {
#if defined(USE_SNPE)
    options->provider_factories.push_back(SNPEProviderFactoryCreator::Create(provider_options));
#else
    status = create_not_supported_status();
#endif
  } else if (strcmp(provider_name, "XNNPACK") == 0) {
#if defined(USE_XNNPACK)
    options->provider_factories.push_back(XnnpackProviderFactoryCreator::Create(provider_options, &(options->value)));
#else
    status = create_not_supported_status();
#endif
  } else if (strcmp(provider_name, "WEBNN") == 0) {
#if defined(USE_WEBNN)
    std::string deviceType = options->value.config_options.GetConfigOrDefault("deviceType", "cpu");
    std::string numThreads = options->value.config_options.GetConfigOrDefault("numThreads", "0");
    std::string powerPreference = options->value.config_options.GetConfigOrDefault("powerPreference", "default");
    provider_options["deviceType"] = deviceType;
    provider_options["numThreads"] = numThreads;
    provider_options["powerPreference"] = powerPreference;
    options->provider_factories.push_back(WebNNProviderFactoryCreator::Create(provider_options));
#else
    status = create_not_supported_status();
#endif
  } else if (strcmp(provider_name, "AZURE") == 0) {
#if defined(USE_AZURE)
    options->provider_factories.push_back(AzureProviderFactoryCreator::Create(provider_options));
#else
    status = create_not_supported_status();
#endif
  } else if (strcmp(provider_name, "JS") == 0) {
#if defined(USE_JSEP)
    std::string preferred_layout;
    if (options->value.config_options.TryGetConfigEntry("preferredLayout", preferred_layout)) {
      provider_options["preferred_layout"] = preferred_layout;
    }
    options->provider_factories.push_back(JsProviderFactoryCreator::Create(provider_options));
#else
    status = create_not_supported_status();
#endif
  } else if (strcmp(provider_name, "VitisAI") == 0) {
#if defined(USE_VITISAI)
    options->provider_factories.push_back(VitisAIProviderFactoryCreator::Create(provider_options));
#else
    status = create_not_supported_status();
#endif
  } else {
    ORT_UNUSED_PARAMETER(options);
    status = OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Unknown provider name. Currently supported values are 'OPENVINO', 'SNPE', 'XNNPACK', 'QNN', 'WEBNN' and 'AZURE'");
  }

  return status;
  API_IMPL_END
}

#if defined(__APPLE__) || defined(ORT_MINIMAL_BUILD)
static OrtStatus* CreateNotEnabledStatus(const std::string& ep) {
  return OrtApis::CreateStatus(ORT_FAIL, (ep + " execution provider is not enabled in this build. ").c_str());
}
#endif

/**
 * Stubs for the publicly exported static registration functions for EPs that are referenced in the C# bindings
 * and are not implemented in provider_bridge_ort.cc.
 *
 * NOTE: The nuget packages that the C# bindings will use are all for full builds, so we don't need to allow for
 * provider_bridge_ort.cc being excluded in a minimal build.
 *
 * These are required when building an iOS app using Xamarin as all external symbols must be defined at compile time.
 * In that case a static ORT library is used and the symbol needs to exist but doesn't need to be publicly exported.
 * TODO: Not sure if we need to purely limit to iOS builds, so limit to __APPLE__ for now
 */
#ifdef __APPLE__
#ifdef __cplusplus
extern "C" {
#endif

#ifndef USE_DML
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("DML");
}
#endif

#ifndef USE_NNAPI
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi,
                    _In_ OrtSessionOptions* options, uint32_t nnapi_flags) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(nnapi_flags);
  return CreateNotEnabledStatus("NNAPI");
}
#endif

#ifndef USE_TVM
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tvm,
                    _In_ OrtSessionOptions* options, _In_ const char* settings) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(settings);
  return CreateNotEnabledStatus("Tvm");
}
#endif

#ifdef __cplusplus
}
#endif

#endif  // __APPLE__

/**
 * Stubs for EP functions from OrtApis that are implemented in provider_bridge_ort.cc in a full build.
 * That file is not included in a minimal build.
 */
#if defined(ORT_MINIMAL_BUILD)
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA_V2,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptionsV2* cuda_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(cuda_options);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::CreateCUDAProviderOptions, _Outptr_ OrtCUDAProviderOptionsV2** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCUDAProviderOptions,
                    _Inout_ OrtCUDAProviderOptionsV2* cuda_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::GetCUDAProviderOptionsAsString, _In_ const OrtCUDAProviderOptionsV2* cuda_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled in this build.");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCUDAProviderOptionsWithValue,
                    _Inout_ OrtCUDAProviderOptionsV2* cuda_options,
                    _In_ const char* key,
                    _In_ void* value) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(value);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::GetCUDAProviderOptionsByName,
                    _In_ const OrtCUDAProviderOptionsV2* cuda_options,
                    _In_ const char* key,
                    _Outptr_ void** ptr) {
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API(void, OrtApis::ReleaseCUDAProviderOptions, _Frees_ptr_opt_ OrtCUDAProviderOptionsV2* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::GetCurrentGpuDeviceId, _In_ int* device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SetCurrentGpuDeviceId, _In_ int device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_ROCM,
                    _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("OpenVINO");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(tensorrt_options);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT_V2,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(tensorrt_options);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptions,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsAsString,
                    _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptionsWithValue,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_ const char* key,
                    _In_ void* value) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(value);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsByName,
                    _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_ const char* key,
                    _Outptr_ void** ptr) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API(void, OrtApis::ReleaseTensorRTProviderOptions, _Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_MIGraphX,
                    _In_ OrtSessionOptions* options, _In_ const OrtMIGraphXProviderOptions* migraphx_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(migraphx_options);
  return CreateNotEnabledStatus("MIGraphX");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CANN,
                    _In_ OrtSessionOptions* options, _In_ const OrtCANNProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("CANN");
}

ORT_API_STATUS_IMPL(OrtApis::CreateCANNProviderOptions, _Outptr_ OrtCANNProviderOptions** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("CANN");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCANNProviderOptions,
                    _Inout_ OrtCANNProviderOptions* cann_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(cann_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("CANN");
}

ORT_API_STATUS_IMPL(OrtApis::GetCANNProviderOptionsAsString,
                    _In_ const OrtCANNProviderOptions* cann_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(cann_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "CANN execution provider is not enabled in this build.");
}

ORT_API(void, OrtApis::ReleaseCANNProviderOptions, _Frees_ptr_opt_ OrtCANNProviderOptions* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_Dnnl,
                    _In_ OrtSessionOptions* options, _In_ const OrtDnnlProviderOptions* dnnl_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(dnnl_options);
  return CreateNotEnabledStatus("Dnnl");
}

ORT_API_STATUS_IMPL(OrtApis::CreateDnnlProviderOptions, _Outptr_ OrtDnnlProviderOptions** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("Dnnl");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateDnnlProviderOptions,
                    _Inout_ OrtDnnlProviderOptions* dnnl_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(dnnl_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("Dnnl");
}

ORT_API_STATUS_IMPL(OrtApis::GetDnnlProviderOptionsAsString,
                    _In_ const OrtDnnlProviderOptions* dnnl_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(dnnl_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "Dnnl execution provider is not enabled in this build.");
}

ORT_API(void, OrtApis::ReleaseDnnlProviderOptions, _Frees_ptr_opt_ OrtDnnlProviderOptions* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

ORT_API_STATUS_IMPL(OrtApis::CreateROCMProviderOptions, _Outptr_ OrtROCMProviderOptions** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateROCMProviderOptions,
                    _Inout_ OrtROCMProviderOptions* rocm_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(rocm_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API_STATUS_IMPL(OrtApis::GetROCMProviderOptionsAsString,
                    _In_ const OrtROCMProviderOptions* rocm_options,
                    _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(rocm_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("ROCM");
}

ORT_API(void, OrtApis::ReleaseROCMProviderOptions, _Frees_ptr_opt_ OrtROCMProviderOptions* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}
#endif
