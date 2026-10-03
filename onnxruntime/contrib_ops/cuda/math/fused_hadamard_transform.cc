// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/fused_hadamard_transform.h"

#include <limits>

#include "contrib_ops/cuda/math/fused_hadamard_transform_impl.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    FusedHadamardTransform,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16>()),
    FusedHadamardTransform);

Status FusedHadamardTransform::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* sign = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input != nullptr && sign != nullptr, "X and sign are required.");

  const TensorShape& input_shape = input->Shape();
  const TensorShape& sign_shape = sign->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() >= 1, "X must have at least one dimension.");
  ORT_RETURN_IF_NOT(sign_shape.NumDimensions() == 1, "sign must be a 1-D tensor.");
  ORT_RETURN_IF_NOT(block_size_ > 0 && block_size_ <= 1024 && (block_size_ & (block_size_ - 1)) == 0,
                    "block_size must be a power of two no greater than 1024.");

  const int64_t last_dimension = input_shape.GetDims().back();
  ORT_RETURN_IF_NOT(sign_shape.Size() == last_dimension,
                    "sign length must equal the last dimension of X.");
  ORT_RETURN_IF_NOT(last_dimension > 0 && last_dimension % block_size_ == 0,
                    "The last dimension of X must be a positive multiple of block_size.");

  Tensor* output = context->Output(0, input_shape);
  ORT_RETURN_IF_NOT(output != nullptr, "Failed to allocate output.");
  const int64_t element_count = input_shape.Size();
  if (element_count == 0) {
    return Status::OK();
  }

  const int64_t block_count = element_count / block_size_;
  ORT_RETURN_IF_NOT(block_count <= std::numeric_limits<uint32_t>::max(),
                    "The input contains too many Hadamard blocks for a CUDA launch.");

  CUDA_RETURN_IF_ERROR(LaunchFusedHadamardTransform(
      Stream(context),
      reinterpret_cast<const half*>(input->Data<MLFloat16>()),
      reinterpret_cast<const half*>(sign->Data<MLFloat16>()),
      reinterpret_cast<half*>(output->MutableData<MLFloat16>()),
      block_count,
      last_dimension,
      static_cast<int>(block_size_)));
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime