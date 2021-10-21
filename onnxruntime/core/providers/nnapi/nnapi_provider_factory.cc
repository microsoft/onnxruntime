// Copyright 2019 JD.com Inc. JD AI

#include "core/providers/nnapi/nnapi_provider_factory.h"

#include "core/common/optional.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

namespace {
struct NnapiProviderFactory : IExecutionProviderFactory {
  NnapiProviderFactory(uint32_t nnapi_flags,
                       const optional<std::string>& partitioning_stop_ops_list)
      : nnapi_flags_(nnapi_flags),
        partitioning_stop_ops_list_(partitioning_stop_ops_list) {}

  ~NnapiProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  const uint32_t nnapi_flags_;
  const optional<std::string> partitioning_stop_ops_list_;
};

std::unique_ptr<IExecutionProvider> NnapiProviderFactory::CreateProvider() {
  return std::make_unique<NnapiExecutionProvider>(nnapi_flags_, partitioning_stop_ops_list_);
}
}  // namespace

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(
    uint32_t nnapi_flags, const optional<std::string>& partitioning_stop_ops_list) {
  return std::make_shared<NnapiProviderFactory>(nnapi_flags, partitioning_stop_ops_list);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options, uint32_t nnapi_flags) {
  const auto partitioning_stop_ops_list = options->value.config_options.GetConfigEntry(
      kOrtSessionOptionsConfigNnapiEpPartitioningStopOps);
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_Nnapi(nnapi_flags, partitioning_stop_ops_list));
  return nullptr;
}
