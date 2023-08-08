// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
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
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)  // start
        .InputMemoryType(OrtMemTypeCPUInput, 1)  // limit
        .InputMemoryType(OrtMemTypeCPUInput, 2)  // delta
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int16_t>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()}),
    Range);

template <typename T>
static Status ComputeRange(cudaStream_t stream, OpKernelContext* ctx) {
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

  // Start, Limit and Delta are stored in CPU.
  T start = *(start_tensor.Data<T>());
  T limit = *(limit_tensor.Data<T>());

  T delta = T(1);
  if (delta_tensor_ptr != nullptr) {
    delta = *(delta_tensor_ptr->Data<T>());
  }

  if (delta == T(0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "delta in Range operator can not be zero!");
  }

  double num = (static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta);
  int count = static_cast<int>(ceil(num));
  if (count <= 0)
    count = 0;
  TensorShape shape = {static_cast<int64_t>(count)};
  T* y = ctx->Output(0, shape)->MutableData<T>();

  if (count > 0) {
    ORT_RETURN_IF_ERROR(RangeImpl(stream, start, delta, count, y));
  }

  return Status::OK();
}

namespace cuda_range_internal {

template <class T>
struct CallCudaRangeImpl {
  Status operator()(cudaStream_t stream, OpKernelContext* ctx) const {
    return ComputeRange<T>(stream, ctx);
  }
};

}  // namespace cuda_range_internal

Status Range::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  }

  utils::MLTypeCallDispatcher<int32_t, float, int64_t, double, int16_t>
      t_disp(input_tensor->GetElementType());
  return t_disp.InvokeRet<Status, cuda_range_internal::CallCudaRangeImpl>(Stream(ctx), ctx);
}

}  // namespace cuda
}  // namespace onnxruntime
