// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
#include "core/providers/cpu/tensor/gather.h"
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/platform/threadpool.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

using DefaultIndexTypes = TypeList<int32_t, int64_t>;

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Gather, Input, 1, DefaultIndexTypes);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// enable all types for layout transformation
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Gather, Input, 1, DefaultIndexTypes);
#else
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Gather, Input, 1, int32_t, int64_t);
#endif
}  // namespace op_kernel_type_control

using EnabledIndexTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                         Gather, Input, 1);
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gather,
    1,
    10,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndexTypes>()),
    Gather);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gather,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndexTypes>()),
    Gather);

ONNX_CPU_OPERATOR_KERNEL(
    Gather,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndexTypes>()),
    Gather);

Status GatherBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  p.input_tensor = context->Input<Tensor>(0);
  const TensorShape& input_data_shape = p.input_tensor->Shape();
  p.indices_tensor = context->Input<Tensor>(1);
  const TensorShape& indices_shape = p.indices_tensor->Shape();

  const auto input_rank = input_data_shape.NumDimensions();
  p.axis = HandleNegativeAxis(axis_, narrow<int64_t>(input_rank));

  std::vector<int64_t> shape;
  shape.reserve(input_rank - 1 + indices_shape.NumDimensions());

  // replace the dimension for p.axis with the shape from the indices
  for (int64_t i = 0; i < p.axis; ++i)
    shape.push_back(input_data_shape[narrow<size_t>(i)]);

  for (const auto dim : indices_shape.GetDims())
    shape.push_back(dim);

  for (int64_t i = p.axis + 1; i < static_cast<int64_t>(input_rank); ++i)
    shape.push_back(input_data_shape[narrow<size_t>(i)]);

  p.output_tensor = context->Output(0, TensorShape(std::move(shape)));

  return Status::OK();
}

template <typename Tin>
Status GatherCopyData(const Tensor* indices_tensor, const uint8_t* src_base, uint8_t* dst_base, bool is_string_type,
                      const size_t element_bytes, const int64_t block_size, const int64_t M,
                      const int64_t N, const int64_t data_batch_bytes, const int64_t gathered_batch_bytes,
                      const TensorShape& input_data_shape, const int64_t axis, concurrency::ThreadPool* tp) {
  const Tin* indices_data = indices_tensor->Data<Tin>();

  // Check the indices first in case there's a out of bound index.
  auto axis_dim_limit = input_data_shape[narrow<size_t>(axis)];

  for (int64_t i = 0; i < N; ++i) {
    Tin idx = indices_data[i];
    if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "indices element out of data bounds, idx=", idx,
                             " must be within the inclusive range [", -axis_dim_limit, ",", axis_dim_limit - 1, "]");
    }
  }

  auto lambda = [&](int64_t index) {
    int64_t batch = index / N;
    int64_t i = index % N;

    const int64_t src_offset_batch = batch * data_batch_bytes;
    const int64_t dst_offset_batch = batch * gathered_batch_bytes;
    Tin idx = indices_data[i];
    idx = idx < 0 ? idx + static_cast<Tin>(axis_dim_limit) : idx;
    const int64_t src_offset = src_offset_batch + idx * block_size;
    const int64_t dst_offset = dst_offset_batch + i * block_size;

    if (is_string_type) {
      reinterpret_cast<std::string*>(dst_base)[dst_offset / element_bytes] =
          reinterpret_cast<const std::string*>(src_base)[src_offset / element_bytes];
    } else {
      memcpy(dst_base + dst_offset, src_base + src_offset, narrow<size_t>(block_size));
    }
  };
  concurrency::ThreadPool::TryParallelFor(tp, SafeInt<ptrdiff_t>(M) * N, static_cast<double>(block_size),
                                          [&lambda](ptrdiff_t first, ptrdiff_t last) {
                                            for (int index = static_cast<int>(first), end = static_cast<int>(last); index < end; ++index) {
                                              lambda(index);
                                            }
                                          });

  return Status::OK();
}

Status Gather::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_data_shape = p.input_tensor->Shape();

  bool is_string_type = p.input_tensor->IsDataTypeString();

  const size_t element_bytes = p.input_tensor->DataType()->Size();
  const int64_t block = input_data_shape.SizeFromDimension(SafeInt<size_t>(p.axis) + 1);
  const int64_t block_size = SafeInt<int64_t>(element_bytes) * block ;
  const int64_t M = input_data_shape.SizeToDimension(narrow<size_t>(p.axis));
  const int64_t N = p.indices_tensor->Shape().Size();
  const int64_t data_batch_bytes = input_data_shape.SizeFromDimension(narrow<size_t>(p.axis)) * element_bytes;
  const int64_t gathered_batch_bytes = N * block * SafeInt<int64_t>(element_bytes);

  const auto* src_base = static_cast<const uint8_t*>(p.input_tensor->DataRaw());
  auto* dst_base = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  if (utils::HasType<EnabledIndexTypes, int32_t>() &&
      p.indices_tensor->IsDataType<int32_t>()) {
    return GatherCopyData<int32_t>(p.indices_tensor, src_base, dst_base, is_string_type, element_bytes,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis, tp);
  }
  if (utils::HasType<EnabledIndexTypes, int64_t>() &&
      p.indices_tensor->IsDataType<int64_t>()) {
    return GatherCopyData<int64_t>(p.indices_tensor, src_base, dst_base, is_string_type, element_bytes,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis, tp);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Gather Tind type not supported in this build.");
}

}  // namespace onnxruntime
