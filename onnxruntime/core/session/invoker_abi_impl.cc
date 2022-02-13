// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <string>
#include <iostream>

#include "core/framework/error_code_helper.h"
#include "core/framework/ort_value.h"
#include "core/framework/execution_provider.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/eager/ort_kernel_invoker.h"
#include "abi_session_options_impl.h"
#include "core/session/ort_env.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {
struct OrtInvokerImpl {
  IOnnxRuntimeOpSchemaRegistryList custom_schemas;
  std::unique_ptr<onnxruntime::ORTInvoker> invoker;
};
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtApis::CreateInvoker,
                    _In_ const OrtEnv* env_,
                    _In_ const OrtSessionOptions* options,
                    size_t provider_index,
                    _Outptr_ OrtInvoker** invoker_) {
  API_IMPL_BEGIN
  // Create the logger
  const onnxruntime::logging::Logger& logger = env_->GetLoggingManager()->DefaultLogger();

  // Create the provider
  ORT_ENFORCE(provider_index < options->provider_factories.size(), "provider_index (" + std::to_string(provider_index) + ") must be less than the provider list size (" + std::to_string(options->provider_factories.size()) + ").");
  std::shared_ptr<onnxruntime::IExecutionProvider> provider = options->provider_factories.at(provider_index)->CreateProvider();

  // Create the invoker
  onnxruntime::OrtInvokerImpl* invoker = new onnxruntime::OrtInvokerImpl;
  invoker->invoker = std::make_unique<onnxruntime::ORTInvoker>(provider, logger, invoker->custom_schemas);
  *invoker_ = reinterpret_cast<OrtInvoker*>(invoker);

  return onnxruntime::ToOrtStatus(onnxruntime::common::Status::OK());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Invoker_Invoke,
                    _Inout_ OrtInvoker* invoker_,
                    const char* op_name_,
                    _In_reads_(input_len) const OrtValue* const* inputs_,
                    size_t inputs_len,
                    _Inout_updates_all_(output_names_len) OrtValue** outputs_,
                    size_t outputs_len,
                    const OrtNodeAttributes* attributes_,
                    const char* domain_,
                    int version) {
  API_IMPL_BEGIN
  auto invoker = reinterpret_cast<onnxruntime::OrtInvokerImpl*>(invoker_);

  std::vector<OrtValue> inputs;
  for (size_t i = 0; i < inputs_len; i++) {
    inputs.push_back(*inputs_[i]);
  }
  std::vector<OrtValue> outputs;
  for (size_t i = 0; i < outputs_len; i++) {
    outputs.push_back(*outputs_[i]);
  }
  const std::string op_name{op_name_};
  const std::string domain{domain_};

  const onnxruntime::NodeAttributes* attributes;
  if (attributes_) {
    attributes = reinterpret_cast<const onnxruntime::NodeAttributes*>(attributes_);
  } else {
    attributes = new onnxruntime::NodeAttributes;
  }
  auto status = invoker->invoker->Invoke(op_name, inputs, outputs, attributes, domain, version);
  return onnxruntime::ToOrtStatus(status);

  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseInvoker, _Frees_ptr_opt_ OrtInvoker* value) {
  delete reinterpret_cast<onnxruntime::OrtInvokerImpl*>(value);
}

ORT_API(void, OrtApis::ReleaseNodeAttributes, _Frees_ptr_opt_ OrtNodeAttributes* value) {
  delete reinterpret_cast<onnxruntime::NodeAttributes*>(value);
}

ORT_API_STATUS_IMPL(NodeAttributes_SetInt64, _Inout_ OrtNodeAttributes* attributes, int value);
ORT_API_STATUS_IMPL(NodeAttributes_SetString, _Inout_ OrtNodeAttributes* attributes, const char* value);
ORT_API_STATUS_IMPL(NodeAttributes_SetFloats, _Inout_ OrtNodeAttributes* attributes, const float* values, size_t values_len);
ORT_API_STATUS_IMPL(NodeAttributes_SetInts, _Inout_ OrtNodeAttributes* attributes, const int* values, size_t values_len);
ORT_API_STATUS_IMPL(NodeAttributes_SetStrings, _Inout_ OrtNodeAttributes* attributes, const char* const* values, size_t values_len);
