// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace contrib {

constexpr const char* ACTIVATION_NAME_PREFIX = "activation_";
constexpr size_t ACTIVATION_NAME_PREFIX_LEN = 11;

template <typename T>
class FusedGemm final : public Gemm<T> {
 public:
  FusedGemm(const OpKernelInfo& info) : Gemm<T>(info) {
    std::string activation = info.GetAttrOrDefault<std::string>("activation", "");
    NodeAttributes attrs;
    for (const auto& p : info.node().GetAttributes()) {
      if (p.first.size() > ACTIVATION_NAME_PREFIX_LEN && p.first.compare(0, ACTIVATION_NAME_PREFIX_LEN, ACTIVATION_NAME_PREFIX) == 0) {
        attrs[p.first.substr(ACTIVATION_NAME_PREFIX_LEN)] = p.second;
      }
    }
    ORT_THROW_IF_ERROR(functors::ElementWiseRangedTransform<T>::Create(activation, attrs, this->activation_));
  }
};

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    FusedGemm,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedGemm<float>);

}  // namespace contrib
}  // namespace onnxruntime
