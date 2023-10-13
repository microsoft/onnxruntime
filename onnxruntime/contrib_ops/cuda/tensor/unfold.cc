// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/tensor/unfold.h"
#include "contrib_ops/cuda/tensor/unfold_impl.h"
#include "core/providers/cpu/tensor/utils.h"

#include <vector>
#include <numeric>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    UnfoldTensor,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    UnfoldTensor);

Status UnfoldTensor::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& input_tensor = *ctx->Input<Tensor>(0);
  const auto& input_dims = input_tensor.Shape().GetDims();
  int rank = SafeInt<int>(input_dims.size());

  int dim = SafeInt<int>(HandleNegativeAxis(dim_, rank));
  ORT_ENFORCE(dim < rank, "input rank:", rank, " is not bigger than attribut specified dim: ", dim);
  ORT_ENFORCE(input_dims[dim] >= size_, "dimsize:", input_dims[dim], " is less than unfold size:", size_);

  int64_t leading_dims = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1LL, std::multiplies<int64_t>());
  int64_t tailing_dims = std::accumulate(input_dims.begin() + (dim + 1), input_dims.end(), 1LL, std::multiplies<int64_t>());

  std::vector<int64_t> output_dims(rank + 1, 0);
  std::copy(input_dims.begin(), input_dims.end(), output_dims.begin());
  output_dims[dim] = (input_dims[dim] - size_) / step_ + 1;
  output_dims.back() = size_;
  TensorShape output_shape(output_dims);
  Tensor* output_tensor = ctx->Output(0, output_shape);

  cudaStream_t stream = this->Stream(ctx);
  const cudaDeviceProp& device_prop = this->GetDeviceProp();
  size_t element_size = input_tensor.DataType()->Size();
  return LaunchUnfoldTensor(
      stream, device_prop, element_size, input_tensor.DataRaw(), output_tensor->MutableDataRaw(),
      leading_dims, input_dims[dim], tailing_dims, size_, step_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
