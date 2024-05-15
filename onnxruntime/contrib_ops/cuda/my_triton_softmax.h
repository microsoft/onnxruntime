#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class MyTritonSoftmax final : public onnxruntime::cuda::CudaKernel {
  public:
    MyTritonSoftmax(const OpKernelInfo& info);
    Status ComputeInternal(OpKernelContext* context) const override;
  private:
    int64_t input_step_size;  // actual row size
    int64_t output_step_size; // unused
    int64_t mask_size;        // amount of row to change
    int64_t batch_size;       // number of blocks to run (grid size)
    int64_t block_size;       // width of block (amount of row to change)
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
