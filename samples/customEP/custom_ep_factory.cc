#include "custom_ep_factory.h"
#include "custom_ep.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include <cmath>

namespace onnxruntime {

void KernelTwo(const Ort::Custom::Tensor<float>& X,
               Ort::Custom::Tensor<int32_t>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t i = 0; i < total; i++) {
    Y_raw[i] = static_cast<int32_t>(round(X_raw[i]));
  }
}

struct CustomProviderFactory : IExecutionProviderFactory {
  CustomProviderFactory(const CustomEpInfo& info) : info_{info} {}
  ~CustomProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

private:
  CustomEpInfo info_;
};

std::unique_ptr<IExecutionProvider> CustomProviderFactory::CreateProvider() {
  std::cout<<"in CustomProviderFactory::CreateProvider()\n";
  return std::make_unique<CustomEp>(info_);
}

CustomProviderFactory custom_provider_factory(CustomEpInfo{0, ""});
}

extern "C" {

ORT_API(onnxruntime::IExecutionProviderFactory*, GetEPFactory) {
  return &onnxruntime::custom_provider_factory;
}

ORT_API_STATUS_IMPL(RegisterCustomOp, _In_ OrtSessionOptions* options) {
  using LiteOp = Ort::Custom::OrtLiteCustomOp;
  static const std::unique_ptr<LiteOp> c_CustomOpTwo{Ort::Custom::CreateLiteCustomOp("CustomOpTwo", onnxruntime::custom_ep_type, onnxruntime::KernelTwo)};
  Ort::CustomOpDomain domain{"test.customop"};
  domain.Add(c_CustomOpTwo.get());

  Ort::UnownedSessionOptions session_options(options);
  session_options.Add(domain);
  return nullptr;
}

}
