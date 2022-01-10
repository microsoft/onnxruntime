#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {
template <typename T, typename T1, typename U>
class LayerNorm final : public CudaKernel {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  double epsilon_;
};

template <typename T, typename T1, typename U, bool simplified>
class LayerNormGrad final : public CudaKernel {
 public:
  LayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

template <typename T, typename T1, typename U>
class InvertibleLayerNormGrad final : public CudaKernel {
 public:
  InvertibleLayerNormGrad(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime