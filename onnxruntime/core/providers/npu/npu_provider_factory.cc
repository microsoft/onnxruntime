

#include "core/providers/npu/npu_provider_factory.h"

#include <memory>

#include "core/providers/npu/npu_execution_provider.h"
#include "core/providers/npu/npu_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct NpuProviderFactory : IExecutionProviderFactory {
  NpuProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~NpuProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> NpuProviderFactory::CreateProvider() {
  NPUExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<NPUExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> NPUProviderFactoryCreator::Create(int use_arena) {
  return std::make_shared<onnxruntime::NpuProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_NPU, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::NPUProviderFactoryCreator::Create(use_arena));
  return nullptr;
}
