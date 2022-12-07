// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_last_token.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

Status GatherLastToken::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);

  const auto& input_shape = input->Shape();

  if (input_shape.NumDimensions() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherLastToken: Input must be 3 dimensional");
  }

  int64_t batch_size = input_shape[0];
  int64_t sequence_length = input_shape[1];
  int64_t hidden_dim = input_shape[2];

  auto output_shape = input_shape;
  output_shape[1] = 1;  // Middle dim is always 1

  Tensor* output = ctx->Output(0, output_shape);

  // Bail out early if the output is going to be empty
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  auto element_size = input->DataType()->Size();

  const void* input_ptr = input->DataRaw();
  void* output_ptr = output->MutableDataRaw();

  auto stream = Stream();
  // This means that the provided batched sequence already
  // has just one token per-sequence in the batch.
  if (input_shape == output_shape) {
    // The allocation planner has deemed that it is safe to re-use
    // the input buffer if the 2 data pointers are the same.
    // In that case, there is nothing to do.
    // Otherwise, do a memcpy.
    if (input_ptr != output_ptr) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_ptr, input_ptr,
                                           input_shape.Size() * element_size, cudaMemcpyDeviceToDevice, stream));
    }

    return Status::OK();
  }

  // TODO (hasesh): Explore perf tuning the following section
  // We are slicing using cudaMemcpy's in a loop, possibly using the
  // Slice kernel is better for some batch sizes.
  const int8_t* src = reinterpret_cast<const int8_t*>(input_ptr);
  int8_t* dst = reinterpret_cast<int8_t*>(output_ptr);

  // Move the src pointer to the last token in the first sequence in the batch
  src += (sequence_length - 1) * hidden_dim * element_size;
  size_t copy_size_in_bytes = hidden_dim * element_size;

  for (int i = 0; i < batch_size; ++i) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst, src, copy_size_in_bytes, cudaMemcpyDeviceToDevice, stream));
    src += sequence_length * hidden_dim * element_size;
    dst += hidden_dim * element_size;
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    GatherLastToken,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .MayInplace(0, 0),
    GatherLastToken);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime