// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/tensor/unfold.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

#include <vector>
#include <numeric>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    UnfoldTensor,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    UnfoldTensor);

template <typename T>
Status LaunchUnfoldTensor(const T* input,
                          T* output,
                          int64_t leading_dims_size,
                          int64_t unfold_dim_size,
                          int64_t tailing_dims_size,
                          int64_t unfold_size,
                          int64_t step_size) {
  int64_t unfold_dim_size_dst = (unfold_dim_size - unfold_size) / step_size + 1;
  int64_t N = leading_dims_size * unfold_dim_size_dst * tailing_dims_size * unfold_size;

  int64_t stride_leading_dst = unfold_size * tailing_dims_size * unfold_dim_size_dst;
  int64_t stride_fold_dim_src = tailing_dims_size * step_size;
  int64_t stride_leading_src = tailing_dims_size * unfold_dim_size;

  for (int64_t idx = 0; idx < N; ++idx) {
    const int64_t idx_leading = idx / stride_leading_dst;
    int64_t n = idx % stride_leading_dst;
    const int64_t stride_fold_dim_dst = tailing_dims_size * unfold_size;
    const int64_t idx_fold = n / stride_fold_dim_dst;
    n %= stride_fold_dim_dst;
    const int64_t idx_tailing = n / unfold_size;
    const int64_t idx_append = n % unfold_size;

    int64_t idx_src = idx_leading * stride_leading_src + idx_fold * stride_fold_dim_src + idx_tailing + idx_append * tailing_dims_size;
    output[idx] = input[idx_src];
  }

  return Status::OK();
}

Status UnfoldTensor::Compute(OpKernelContext* ctx) const {
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

  Status status;
  if (input_tensor.IsDataType<float>()) {
    status = LaunchUnfoldTensor<float>(input_tensor.Data<float>(), output_tensor->MutableData<float>(),
                                       leading_dims, input_dims[dim], tailing_dims, size_, step_);
  } else if (input_tensor.IsDataType<double>()) {
    status = LaunchUnfoldTensor<double>(input_tensor.Data<double>(), output_tensor->MutableData<double>(),
                                        leading_dims, input_dims[dim], tailing_dims, size_, step_);
  } else if (input_tensor.IsDataType<int32_t>()) {
    status = LaunchUnfoldTensor<int32_t>(input_tensor.Data<int32_t>(), output_tensor->MutableData<int32_t>(),
                                         leading_dims, input_dims[dim], tailing_dims, size_, step_);
  } else if (input_tensor.IsDataType<int64_t>()) {
    status = LaunchUnfoldTensor<int64_t>(input_tensor.Data<int64_t>(), output_tensor->MutableData<int64_t>(),
                                         leading_dims, input_dims[dim], tailing_dims, size_, step_);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported data type: ", input_tensor.DataType());
  }

  return status;
}

}  // namespace contrib
}  // namespace onnxruntime
