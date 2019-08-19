#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {
template <typename T, typename U>
class LayerNorm final : public CudaKernel {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  double epsilon_;
};

template <typename T, typename U>
class LayerNormGrad final : public CudaKernel {
 public:
  LayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime