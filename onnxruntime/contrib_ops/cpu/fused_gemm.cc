// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class FusedGemm final : public Gemm<T> {
 public:
  FusedGemm(const OpKernelInfo& info) : Gemm<T>(info) {
    Gemm<T>::activation_ = info.GetAttrOrDefault<std::string>("activation", "");
    Gemm<T>::leaky_relu_alpha_ = info.GetAttrOrDefault("leaky_relu_alpha", 0.01f);
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
