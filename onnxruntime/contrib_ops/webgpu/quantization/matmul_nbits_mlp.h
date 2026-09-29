// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/common/status.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

// Gate activation applied between the gate and up MatMulNBits projections.
// Currently only SiLU is supported; future activations (e.g. GELU for Gemma-style
// gated MLPs) can be added here and threaded through the kernel and shader template.
enum class MlpActivationKind : uint32_t {
  Silu = 0,
};

// Parses the `activation` attribute string into MlpActivationKind. Returns a non-OK
// Status for unsupported activations so the kernel rejects unknown values up front.
Status ParseMlpActivation(std::string_view name, MlpActivationKind* out);

class MatMulNBitsMlp final : public WebGpuKernel {
 public:
  explicit MatMulNBitsMlp(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    block_size_ = info.GetAttr<int64_t>("block_size");
    bits_ = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
    std::string activation;
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &activation).IsOK(),
                "MatMulNBitsMlp requires the 'activation' attribute.");
    ORT_ENFORCE(ParseMlpActivation(activation, &activation_kind_).IsOK(),
                "MatMulNBitsMlp: unsupported activation '", activation, "'.");
    ORT_ENFORCE(bits_ == 4 || bits_ == 8 || bits_ == 2,
                "Only 4b/8b/2b quantization is supported for MatMulNBitsMlp op.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t accuracy_level_;
  int64_t bits_;
  float epsilon_;
  MlpActivationKind activation_kind_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
