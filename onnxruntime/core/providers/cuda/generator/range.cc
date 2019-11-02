// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
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

namespace cuda_range_internal {

// Workaround GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47226
// can not specify lambda directly
template <class T>
inline int invoke_cuda_range_callable(size_t& called, Status& s, int32_t dt_type, OpKernelContext* ctx) {
  auto fn = [&] {
    if (utils::ToTensorDataType<T>() == dt_type) {
      s = ComputeRange<T>(ctx);
      ++called;
    }
    return 0;
  };
  return fn();
}

template <typename... Types>
inline Status CallDispatcher(int32_t dt_type, OpKernelContext* ctx) {
  Status s;
  size_t called = 0;

  int results[] = {0, invoke_cuda_range_callable<Types>(called, s, dt_type, ctx)...};

  ORT_UNUSED_PARAMETER(results);
  ORT_ENFORCE(called < 2, "Range CallDispatcher broken. Check for duplicate type.");
  if (called == 0) {
    s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Range op: Unsupported tensor data type:", dt_type);
  }

  return s;
}
}  // namespace cuda_range_internal


Status Range::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_tensor = ctx->Input<Tensor>(0);
  if (input_tensor == nullptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  }

  auto data_type = input_tensor->DataType()->AsPrimitiveDataType();
  ORT_RETURN_IF_NOT(data_type != nullptr, "Range op: Unsupported tensor data type:", data_type);
  return cuda_range_internal::CallDispatcher<int32_t, float, int64_t, double, int16_t>(data_type->GetTensorElementType(), ctx);
}

}  // namespace cuda
}  // namespace onnxruntime
