// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opwrapper/opwrapper_provider_factory.h"

#include <sstream>
#include "core/providers/opwrapper/ort_opwrapper_apis.h"
#include "core/providers/opwrapper/opwrapper_provider_factory_creator.h"
#include "core/providers/opwrapper/opwrapper_execution_provider.h"
#include "core/session/ort_apis.h"
#include "core/session/abi_session_options_impl.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/utils.h"

namespace onnxruntime {

struct OpWrapperProviderFactory : IExecutionProviderFactory {
  explicit OpWrapperProviderFactory(const ProviderOptionsMap& provider_options_map)
    : provider_options_map_(provider_options_map) {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<OpWrapperExecutionProvider>(provider_options_map_);
  }
 private:
  ProviderOptionsMap provider_options_map_;
};

std::shared_ptr<IExecutionProviderFactory>
OpWrapperProviderFactoryCreator::Create(const ProviderOptionsMap& provider_options_map) {
  return std::make_shared<onnxruntime::OpWrapperProviderFactory>(provider_options_map);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtOpWrapperApis::SessionOptionsAppendExecutionProvider,
                    _In_ OrtSessionOptions* session_options,
                    _In_reads_(num_ops) const char* const* op_names,
                    _In_reads_(num_ops) const OrtOpWrapperProviderOptions* const* provider_options,
                    _In_ size_t num_ops) {
  API_IMPL_BEGIN
  onnxruntime::ProviderOptionsMap options_map;
  options_map.reserve(num_ops);

  for (size_t i = 0; i < num_ops; ++i) {
    const auto* op_options = reinterpret_cast<const onnxruntime::ProviderOptions*>(provider_options[i]);
    options_map[op_names[i]] = *op_options;
  }

  auto provider_factory = onnxruntime::OpWrapperProviderFactoryCreator::Create(options_map);
  session_options->provider_factories.push_back(provider_factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtOpWrapperApis::CreateProviderOptions, _Outptr_ OrtOpWrapperProviderOptions** out) {
  API_IMPL_BEGIN
  *out = reinterpret_cast<OrtOpWrapperProviderOptions*>(new onnxruntime::ProviderOptions);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtOpWrapperApis::ProviderOptions_Update, _Inout_ OrtOpWrapperProviderOptions* provider_options,
                    _In_reads_(num_options) const char* const* options_keys,
                    _In_reads_(num_options) const char* const* options_values,
                    _In_ size_t num_options) {
  API_IMPL_BEGIN
  auto& options = *reinterpret_cast<onnxruntime::ProviderOptions*>(provider_options);

  for (size_t i = 0; i < num_options; ++i) {
    options[options_keys[i]] = options_values[i];
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtOpWrapperApis::ProviderOptions_HasOption,
                    _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ const char* key, _Out_ int* out) {
  API_IMPL_BEGIN
  const auto* options = reinterpret_cast<const onnxruntime::ProviderOptions*>(provider_options);
  auto it = options->find(key);
  *out = it != options->end();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtOpWrapperApis::ProviderOptions_GetOption,
                    _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ const char* key, _Out_ const char** value, _Out_opt_ size_t* length) {
  API_IMPL_BEGIN
  const auto* options = reinterpret_cast<const onnxruntime::ProviderOptions*>(provider_options);
  auto it = options->find(key);

  if (it == options->end()) {
    std::ostringstream err_msg;
    err_msg << "Provider option '"<< key << "' was not found.";
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, err_msg.str().c_str());
  }

  *value = it->second.c_str();

  if (length != nullptr) {
    *length = it->second.length();
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtOpWrapperApis::ProviderOptions_Serialize,
                    _In_ const OrtOpWrapperProviderOptions* provider_options,
                    _In_ OrtAllocator* allocator, _Outptr_ const char*** keys, _Outptr_ size_t** key_lengths,
                    _Outptr_ const char*** values, _Outptr_ size_t** value_lengths, _Out_ size_t* num_options) {
  API_IMPL_BEGIN
  const auto* options = reinterpret_cast<const onnxruntime::ProviderOptions*>(provider_options);

  *num_options = options->size();

  if (*num_options == 0) {
    *keys = nullptr;
    *key_lengths = nullptr;
    *values = nullptr;
    *value_lengths = nullptr;
    return nullptr;
  }

  const size_t strs_array_size = *num_options * sizeof(const char*);
  const size_t lens_array_size = *num_options * sizeof(size_t);

  *keys = reinterpret_cast<const char**>(allocator->Alloc(allocator, strs_array_size));
  *values = reinterpret_cast<const char**>(allocator->Alloc(allocator, strs_array_size));
  *key_lengths = reinterpret_cast<size_t*>(allocator->Alloc(allocator, lens_array_size));
  *value_lengths = reinterpret_cast<size_t*>(allocator->Alloc(allocator, lens_array_size));

  size_t index = 0;

  for (const auto& it : *options) {
    const std::string& key = it.first;
    const std::string& val = it.second;

    (*keys)[index] = key.c_str();
    (*key_lengths)[index] = key.length();

    (*values)[index] = val.c_str();
    (*value_lengths)[index] = val.length();

    ++index;
  }

  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtOpWrapperApis::ReleaseOpWrapperProviderOptions,
        _Frees_ptr_opt_ OrtOpWrapperProviderOptions* provider_options) {
  if (provider_options) {
    delete reinterpret_cast<onnxruntime::ProviderOptions*>(provider_options);
  }
}

ORT_API_STATUS_IMPL(OrtOpWrapperApis::KernelInfo_GetProviderOptions, _In_ const OrtKernelInfo* info,
                    _In_ const char* op_name, _Outptr_ OrtOpWrapperProviderOptions** provider_options) {
  API_IMPL_BEGIN
  const auto* op_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  const auto* ep = op_info->GetExecutionProvider();
  const std::string& ep_type = ep->Type();

  if (ep_type.compare(onnxruntime::kOpWrapperExecutionProvider) != 0) {
    std::ostringstream err_msg;
    err_msg << "Expected provider of type '" << onnxruntime::kOpWrapperExecutionProvider
            << "' but got type '" << ep_type << "'.";
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, err_msg.str().c_str());
  }

  const auto* opwrapper_ep = reinterpret_cast<const onnxruntime::OpWrapperExecutionProvider*>(ep);
  auto* options = new onnxruntime::ProviderOptions(opwrapper_ep->GetOpProviderOptions(op_name));
  *provider_options = reinterpret_cast<OrtOpWrapperProviderOptions*>(options);
  return nullptr;
  API_IMPL_END
}

static constexpr OrtOpWrapperApi ort_opwrapper_api_13_to_x = {
  &OrtOpWrapperApis::SessionOptionsAppendExecutionProvider,
  &OrtOpWrapperApis::CreateProviderOptions,
  &OrtOpWrapperApis::ProviderOptions_Update,
  &OrtOpWrapperApis::ProviderOptions_HasOption,
  &OrtOpWrapperApis::ProviderOptions_GetOption,
  &OrtOpWrapperApis::ProviderOptions_Serialize,
  &OrtOpWrapperApis::ReleaseOpWrapperProviderOptions,
  &OrtOpWrapperApis::KernelInfo_GetProviderOptions,
};

ORT_API(const OrtOpWrapperApi*, GetOrtOpWrapperApi, uint32_t version) {
#ifdef USE_OPWRAPPER
  if (version >= 13 && version <= ORT_API_VERSION) {
    return &ort_opwrapper_api_13_to_x;
  }
#endif
  return nullptr;
}
