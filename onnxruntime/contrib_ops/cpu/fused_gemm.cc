// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class FusedGemm final : public Gemm<T> {
 public:
  FusedGemm(const OpKernelInfo& info) : Gemm<T>(info) {
    std::string activation = info.GetAttrOrDefault<std::string>("activation", "");
    NodeAttributes attrs;
    for (const auto& p : info.node().GetAttributes()) {
      if (p.first.size() >= 12 && p.first.compare(0, 11, "activation_") == 0) {
        attrs[p.first.substr(11)] = p.second;
      }
    }
    functors::ElementWiseRangedTransform<T>* p;
    ORT_THROW_IF_ERROR(functors::ElementWiseRangedTransform<float>::Create(activation, attrs, &p));
    this->activation_.reset(p);
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
