#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class MyTritonKernel final : public onnxruntime::cuda::CudaKernel {
  public:
    MyTritonKernel(const OpKernelInfo& info);
    Status ComputeInternal(OpKernelContext* context) const override;
  private:
    int64_t input_size;
    int64_t block_size;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
