// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/tensor/dynamic_time_warping.h"
#include "contrib_ops/cuda/tensor/dynamic_time_warping_impl.h"
#include "core/providers/cpu/tensor/utils.h"

#include <vector>
#include <numeric>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    DynamicTimeWarping,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int32_t>()),
    DynamicTimeWarping);

Status DynamicTimeWarping::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& input_tensor = *ctx->Input<Tensor>(0);
  const auto& input_dims = input_tensor.Shape().GetDims();
  int rank = SafeInt<int>(input_dims.size());
  ORT_ENFORCE(rank == 2 || (rank == 3 && input_dims[0] == 1), "Currently input rank must be 2, or (3 with first dim equal to 1), but got:", rank);

  const size_t rows = SafeInt<size_t>(input_dims[rank == 3 ? 1 : 0]);
  const size_t cols = SafeInt<size_t>(input_dims[rank == 3 ? 2 : 1]);
  size_t max_index_len = 0;

  size_t buffer_size_in_bytes = GetDynamicTimeWarpingBufferSize(1, rows, cols, max_index_len);
  IAllocatorUniquePtr<int8_t> buffer = GetScratchBuffer<int8_t>(buffer_size_in_bytes, ctx->GetComputeStream());

  size_t result_len = 0;
  ORT_RETURN_IF_ERROR(LaunchDynamicTimeWarping(
      this->Stream(ctx), this->GetDeviceProp(), 1, rows, cols,
      input_tensor.Data<float>(), buffer.get(), result_len));

  Tensor* output_tensor = ctx->Output(0, TensorShape{2LL, SafeInt<int64_t>(result_len)});

  return CUDA_CALL(cudaMemcpy2DAsync(
      output_tensor->MutableData<int32_t>(), result_len * sizeof(int32_t),
      buffer.get() + ((max_index_len - result_len) * sizeof(int32_t)), max_index_len * sizeof(int32_t),
      result_len * sizeof(int32_t), 2,
      cudaMemcpyDeviceToDevice, this->Stream(ctx)));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
