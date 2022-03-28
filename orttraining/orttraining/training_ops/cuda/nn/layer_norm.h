#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {
template <typename T, typename U, typename V, bool simplified>
class LayerNormGrad final : public CudaKernel {
 public:
  LayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

template <typename T, typename U, typename V>
class InvertibleLayerNormGrad final : public CudaKernel {
 public:
  InvertibleLayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime