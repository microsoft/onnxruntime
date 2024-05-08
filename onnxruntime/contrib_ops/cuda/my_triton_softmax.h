#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class MyTritonSoftmax final : public onnxruntime::cuda::CudaKernel {
 public:
  MyTritonSoftmax(const OpKernelInfo& info) : CudaKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
