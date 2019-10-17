// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/cuda/cuda_common.h"
#include "range.h"
#include "range_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Range,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<int16_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>(),
                                            DataTypeImpl::GetTensorType<int64_t>()}),
    Range);

template <typename T>
static Status ComputeRange(OpKernelContext* ctx) {
  const auto& start_tensor = *ctx->Input<Tensor>(0);
  const auto& limit_tensor = *ctx->Input<Tensor>(1);
  const auto* delta_tensor_ptr = ctx->Input<Tensor>(2);

  if (!start_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "start in Range operator should be scalar like tensor, yet got shape:",
                           start_tensor.Shape());
  }
  if (!limit_tensor.Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "limit in Range operator should be scalar like tensor, yet got shape:",
                           limit_tensor.Shape());
  }
  if (delta_tensor_ptr != nullptr && !delta_tensor_ptr->Shape().IsScalar()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "delta in Range operator should be scalar like tensor, yet got shape:",
                           delta_tensor_ptr->Shape());
  }

  // Start, Limit and Delta are stored in GPU. So we need copy it to CPU to read.
  // It is better to store these tensors in pinned memory or CPU for better performance.
  T start;
  CUDA_RETURN_IF_ERROR(cudaMemcpy(&start, start_tensor.template Data<T>(), sizeof(T), cudaMemcpyDeviceToHost));

  T limit;
  CUDA_RETURN_IF_ERROR(cudaMemcpy(&limit, limit_tensor.template Data<T>(), sizeof(T), cudaMemcpyDeviceToHost));

  T delta = T(1);
  if (delta_tensor_ptr != nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemcpy(&delta, delta_tensor_ptr->template Data<T>(), sizeof(T), cudaMemcpyDeviceToHost));
  }

  if (delta == T(0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }

  int count = static_cast<int>(ceil(1.0 * (limit - start) / delta));
  if (count <= 0)
    count = 0;
  TensorShape shape = {static_cast<int64_t>(count)};
  T* y = ctx->Output(0, shape)->template MutableData<T>();

  if (count > 0) {
    if (!RangeImpl(start, delta, count, y)) {
      CUDA_CALL(cudaGetLastError());
      return Status(common::ONNXRUNTIME, common::FAIL);
    }
  }

  return Status::OK();
}

Status Range::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  }

  auto data_type = input_tensor->DataType();
  if (data_type == DataTypeImpl::GetType<int32_t>()) {
    return ComputeRange<int32_t>(ctx);
  } else if (data_type == DataTypeImpl::GetType<float>()) {
    return ComputeRange<float>(ctx);
  } else if (data_type == DataTypeImpl::GetType<int64_t>()) {
    return ComputeRange<int64_t>(ctx);
  } else if (data_type == DataTypeImpl::GetType<double>()) {
    return ComputeRange<double>(ctx);
  } else if (data_type == DataTypeImpl::GetType<int16_t>()) {
    return ComputeRange<int16_t>(ctx);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Range op: Unsupported tensor data type:", data_type);
}

}  // namespace cuda
}  // namespace onnxruntime
