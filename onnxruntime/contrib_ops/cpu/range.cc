#include "range.h"
#include "onnx/defs/schema.h"

#include <cmath>

namespace onnxruntime {
namespace contrib {

template <typename T>
static Status ComputeRange(OpKernelContext* ctx) {
  auto& start_tensor = *ctx->Input<Tensor>(0);
  auto& limit_tensor = *ctx->Input<Tensor>(1);
  auto  delta_tensor_ptr = ctx->Input<Tensor>(2);

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

  T start = *start_tensor.template Data<T>();
  T limit = *limit_tensor.template Data<T>();
  T delta = (delta_tensor_ptr == nullptr) ? T{1} : *(delta_tensor_ptr->template Data<T>());
  
  if (delta == T{0}) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  "delta in Range operator can not be zero!");
  }
  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
  if (n <= 0) n = 0;
  TensorShape shape = {n};
  T* y = ctx->Output(0, shape)->template MutableData<T>();
  for (int64_t i = 0; i < n; ++i) {
      *y++ = start;
      start += delta;
  }

  return Status::OK();
}

Status Range::Compute(OpKernelContext* ctx) const {
  auto input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  auto data_type = input_tensor->DataType();
  if (data_type == DataTypeImpl::GetType<int32_t>()) {
      return ComputeRange<int32_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int16_t>()) {
      return ComputeRange<int16_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int64_t>()) {
      return ComputeRange<int64_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<float>()) {
      return ComputeRange<float>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<double>()) {
      return ComputeRange<double>(ctx);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
                                 "Unsupportted tensor data type:",
                                 data_type);
}

/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
    Range,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {
        DataTypeImpl::GetTensorType<float>(), 
        DataTypeImpl::GetTensorType<double>(),
        DataTypeImpl::GetTensorType<int16_t>(), 
        DataTypeImpl::GetTensorType<int32_t>(), 
        DataTypeImpl::GetTensorType<int64_t>()}),
    Range);


}  // namespace contrib
}  // namespace onnxruntime
