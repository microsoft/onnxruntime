// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_common.h"
#include "range.h"
#include "range_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Range,
    kOnnxDomain,
    11,
    26,
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

// Opset 27 added float16/bfloat16 to the type constraint and a stash_type attribute.
// This kernel continues to natively support the common numeric types only; a native
// float16/bfloat16 CUDA kernel (range_impl.cu specialization) is a follow-up enhancement.
// Note that float16/bfloat16 Range models still execute correctly today: Range-27 carries
// an ONNX function body that ORT expands into primitive ops at graph-partition time, so the
// follow-up is about adding an efficient native kernel, not about fixing broken functionality.
ONNX_OPERATOR_KERNEL_EX(
    Range,
    kOnnxDomain,
    27,
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

  // Compute the element count in double, mirroring the CPU kernel's guard structure and error
  // messages (core/providers/cpu/generator/range.cc ComputeRange) and the shape-inference path
  // (core/graph/contrib_ops/range_schema_defs.cc CalcRangeDim). The operands are
  // promoted to double before the subtraction so integral inputs cannot overflow in T.
  double num = ceil((static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta));
  if (!std::isfinite(num)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Range: the computed number of elements is not a finite value.");
  }
  // Empty or backward ranges clamp to 0; handle the non-positive case before the cast so a
  // large-magnitude negative count can never reach (and overflow) the int64 conversion.
  int64_t count = 0;
  if (num > 0) {
    // static_cast<double>(INT64_MAX) rounds up to 2^63 (9223372036854775808.0), which is not
    // representable as int64_t, so reject any count at or above that boundary before the cast.
    if (num >= 9223372036854775808.0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Range: the computed number of elements exceeds the supported range.");
    }
    count = static_cast<int64_t>(num);
  }
  TensorShape shape = {count};
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
