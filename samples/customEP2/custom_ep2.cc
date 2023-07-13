#include <memory>
#include <cmath>
#include "custom_ep2.h"
#include "core/session/onnxruntime_lite_custom_op.h"

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

CustomEp2::CustomEp2() : type_{"customEp2"} {
    custom_ops_.push_back(Ort::Custom::CreateLiteCustomOp("CustomOpTwo", type_.c_str(), KernelTwo));  // TODO: should use smart pointer for vector custom_ops_
}

CustomEp2 custom_ep2;
}

#ifdef __cplusplus
extern "C" {
#endif

ORT_API(onnxruntime::CustomEp2*, GetExternalProvider) {
    return &onnxruntime::custom_ep2;
}

#ifdef __cplusplus
}
#endif
