// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/trilu.h"
#include "core/providers/cuda/tensor/trilu_impl.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Trilu,
    kOnnxDomain,
    14,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Trilu);

Status Trilu::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_ptr = ctx->Input<Tensor>(0);
  const auto* k = ctx->Input<Tensor>(1);

  int64_t k_val = 0;
  if (k) {
    ORT_ENFORCE(IsScalarOr1ElementVector(k), "k should be a 1-D or 0-D tensor.");
    k_val = *(k->Data<int64_t>());
  }
  if (input_ptr == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *input_ptr;
  const auto& shape = input.Shape();
  const auto& input_dims = shape.GetDims();
  int32_t rank = gsl::narrow_cast<int32_t>(input_dims.size());
  if (rank < 2) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor should have a rank of at least 2");
  }
  Tensor* output = ctx->Output(0, shape);
  int64_t matrix_size = input_dims[rank - 1] * input_dims[rank - 2];
  if (matrix_size == 0) {
    return Status::OK();
  }
  const fast_divmod row_col_divmod_indices(gsl::narrow_cast<int>(input_dims[rank - 1]));
  const fast_divmod batch_divmod_indices(gsl::narrow_cast<int>(matrix_size));

  size_t element_size = input.DataType()->Size();
  return TriluImpl(
      this->Stream(ctx),
      upper_,
      element_size,
      k_val,
      input.DataRaw(),
      output->MutableDataRaw(),
      gsl::narrow<int>(shape.Size()),
      batch_divmod_indices,
      row_col_divmod_indices);
}

}  // namespace cuda
}  // namespace onnxruntime
