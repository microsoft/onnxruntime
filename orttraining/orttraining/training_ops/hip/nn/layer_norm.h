#pragma once
#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {
template <typename T, typename U>
class LayerNorm final : public HipKernel {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  double epsilon_;
};

template <typename T, typename U>
class LayerNormGrad final : public HipKernel {
 public:
  LayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace hip
}  // namespace onnxruntime