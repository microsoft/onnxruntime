// Copyright 2019 JD.com Inc. JD AI

#include "core/providers/nnapi/nnapi_provider_factory.h"

#include "core/common/optional.h"
#include "core/common/string_utils.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

namespace {
struct NnapiProviderFactory : IExecutionProviderFactory {
  NnapiProviderFactory(uint32_t nnapi_flags,
                       const optional<std::unordered_set<std::string>>& partitioning_stop_ops)
      : nnapi_flags_(nnapi_flags),
        partitioning_stop_ops_(partitioning_stop_ops) {}

  ~NnapiProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  const uint32_t nnapi_flags_;
  const optional<std::unordered_set<std::string>> partitioning_stop_ops_;
};

std::unique_ptr<IExecutionProvider> NnapiProviderFactory::CreateProvider() {
  return std::make_unique<NnapiExecutionProvider>(nnapi_flags_, partitioning_stop_ops_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi_Internal(
    uint32_t nnapi_flags, const optional<std::unordered_set<std::string>>& partitioning_stop_ops) {
  return std::make_shared<NnapiProviderFactory>(nnapi_flags, partitioning_stop_ops);
}
}  // namespace

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(uint32_t nnapi_flags) {
  return CreateExecutionProviderFactory_Nnapi_Internal(nnapi_flags, nullopt);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options, uint32_t nnapi_flags) {
  const auto partitioning_stop_ops = [&]() -> onnxruntime::optional<std::unordered_set<std::string>> {
    if (std::string partitioning_stop_ops_value{};
        options->value.config_options.TryGetConfigEntry(kOrtSessionOptionsConfigNnapiEpPartitioningStopOps,
                                                        partitioning_stop_ops_value)) {
      const auto partitioning_stop_ops_list = onnxruntime::utils::SplitString(partitioning_stop_ops_value, ",");
      return std::unordered_set<std::string>(partitioning_stop_ops_list.begin(), partitioning_stop_ops_list.end());
    }
    return onnxruntime::nullopt;
  }();

  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_Nnapi_Internal(nnapi_flags, partitioning_stop_ops));
  return nullptr;
}
