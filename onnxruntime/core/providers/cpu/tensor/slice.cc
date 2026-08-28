// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"

#include <limits>
#include <unordered_map>

#include "core/common/narrow.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

namespace op_kernel_type_control {
// we're using one set of types for all opsets
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 0,
    element_type_lists::All);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 0, int32_t, int64_t);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 1, int32_t, int64_t);
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Slice, Input, 1, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace {
using EnabledDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                        Slice, Input, 0);
using EnabledIndicesTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                           Slice, Input, 1);
}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    1, 9,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>()),
    Slice1);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    10, 10,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Slice,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

ONNX_CPU_OPERATOR_KERNEL(
    Slice,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<EnabledDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndicesTypes>()),
    Slice10);

template <typename T>
static Status SliceImpl(OpKernelContext* ctx,
                        const Tensor& input_tensor,
                        SliceOp::PrepareForComputeMetadata& compute_metadata) {
  TensorShape output_shape(compute_metadata.output_dims_);
  auto& output_tensor = *ctx->Output(0, output_shape);

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  // use MutableDataRaw as actual data type in tensor may not match as we templatize on data size
  T* output = reinterpret_cast<T*>(output_tensor.MutableDataRaw());
  const auto* output_end = output + output_tensor.Shape().Size();

  auto create_output = [&output, &output_end](SliceIterator<T>& slice_input_iterator) {
    while (output < output_end) {
      output = slice_input_iterator.CopyContiguousInnermostAxes(output);
    }

    ORT_ENFORCE(output == output_end);
  };

  if (compute_metadata.p_flattened_input_dims_) {
    // If we were able to coalesce the input and output shapes, use the new shapes.
    auto input_iterator2 =
        SliceIterator<T>(input_tensor, TensorShape(compute_metadata.flattened_input_dims_), compute_metadata.starts_,
                         compute_metadata.flattened_output_dims_, compute_metadata.steps_);
    create_output(input_iterator2);
  } else {
    auto input_iterator2 = SliceIterator<T>(input_tensor, compute_metadata.starts_, compute_metadata.output_dims_,
                                            compute_metadata.steps_);
    create_output(input_iterator2);
  }

  return Status::OK();
}

template <typename EnabledTypes, typename T>
static inline bool CallSliceImplIfEnabled(OpKernelContext* ctx,
                                          const Tensor& input_tensor,
                                          SliceOp::PrepareForComputeMetadata& compute_metadata,
                                          Status& status) {
  constexpr bool enabled = utils::HasTypeWithSameSize<EnabledTypes, T>();
  if constexpr (enabled) {
    status = SliceImpl<T>(ctx, input_tensor, compute_metadata);
  }

  return enabled;
}

Status SliceBase::Compute(OpKernelContext* ctx) const {
  const auto* input_tensor_ptr = ctx->Input<Tensor>(0);
  const auto& input_tensor = *input_tensor_ptr;
  const auto input_dimensions = input_tensor.Shape().GetDims();

  if (input_dimensions.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot slice scalars");
  }

  SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);

  // Slice V10 & DynamicSlice
  if (dynamic_) {
    TensorShapeVector input_starts;
    TensorShapeVector input_ends;
    TensorShapeVector input_axes;
    TensorShapeVector input_steps;
    ORT_RETURN_IF_ERROR(FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2),
                                             ctx->Input<Tensor>(3), ctx->Input<Tensor>(4),
                                             input_starts, input_ends,
                                             input_axes, input_steps));

    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  }
  // Slice V1-9
  else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_, compute_metadata));
  }

  Status status = Status::OK();

  bool supported = false;
  if (input_tensor.IsDataTypeString()) {
    if (utils::HasType<EnabledDataTypes, std::string>()) {
      supported = true;
      status = SliceImpl<std::string>(ctx, input_tensor, compute_metadata);
    }
  } else {
    const auto element_size = input_tensor.DataType()->Size();
    // call SliceImpl
    switch (element_size) {
      case sizeof(uint32_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint32_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint64_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint64_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint16_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint16_t>(ctx, input_tensor, compute_metadata, status);
        break;
      case sizeof(uint8_t):
        supported = CallSliceImplIfEnabled<EnabledDataTypes, uint8_t>(ctx, input_tensor, compute_metadata, status);
        break;
      default:
        // leave 'supported' as false
        break;
    }
  }

  if (!supported) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input data type of ", input_tensor.DataType());
  }

  return status;
}

}  // namespace onnxruntime
