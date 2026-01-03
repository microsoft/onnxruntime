// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/concat.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/copy.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Concat,
    4,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Concat,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// Opset 13 .
ONNX_CPU_OPERATOR_KERNEL(
    Concat,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

namespace op_kernel_type_control {
// we're using one set of types for all opsets
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Concat, Input, 0,
    element_type_lists::All);

// Concat can be used with dimensions or indices so require int32_t and int64_t to be supported
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Concat, Input, 0, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace {
using EnabledDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                        Concat, Input, 0);
}  // namespace

// this method will be shared between 'Concat' (CPU and GPU) and
// 'ConcatFromSequence' ('concat' and 'stack' modes) to validate inputs
Status ConcatBase::PrepareForCompute(OpKernelContext* ctx,
                                     const InlinedTensorsVector& input_tensors,
                                     Prepare& p) const {
  return PrepareForComputeImpl(ctx, input_tensors, p);
}

namespace {
TensorShapeVector StridesForStack(const TensorShapeVector& full_strides, uint64_t axis) {
  // if we are stacking, skip the dimension that will be stacked along in the output strides
  // (the striding for that dimension is handled by the initial_output_offset)
  const auto num_dims = full_strides.size();

  TensorShapeVector strides;
  strides.reserve(num_dims - 1);

  for (size_t i = 0; i < num_dims - 1; i++) {
    auto read_i = (i >= axis) ? i + 1 : i;
    strides.push_back(full_strides[read_i]);
  }
  return strides;
}
}  // namespace

// This method computes the output tensor for Concat/ConcatFromSequence ops
Status ConcatBase::ComputeImpl(Prepare& p, OpKernelContext* ctx) const {
  int input_count = static_cast<int>(p.inputs.size());
  int64_t initial_output_offset = 0;  // initial offset for each input

  auto output_strides_full = StridesForTensor(*p.output_tensor);
  // Note that output_strides_full is only used later when is_stack_ is true, so it's safe to move
  auto output_strides_for_copy = is_stack_ ? StridesForStack(output_strides_full, p.axis) : std::move(output_strides_full);

  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];

    // no data in this tensor - so skip it
    if (prep.num_elements == 0)
      continue;

    // parallel copy the data across
    auto status = DispatchStridedCopy<EnabledDataTypes>(ctx->GetOperatorThreadPool(),
                                                        *p.output_tensor,
                                                        onnxruntime::narrow<ptrdiff_t>(initial_output_offset),
                                                        output_strides_for_copy,
                                                        prep.tensor->Shape(),
                                                        *prep.tensor,
                                                        0,  // src_offset
                                                        StridesForTensor(*prep.tensor));
    ORT_RETURN_IF_ERROR(status);

    // advance along the axis that we are concatenating on (by the size of the axis of the tensor that we just copied)
    if (is_stack_) {
      initial_output_offset += output_strides_full[onnxruntime::narrow<size_t>(p.axis)];
    } else {
      initial_output_offset += prep.tensor->Shape()[onnxruntime::narrow<size_t>(p.axis)] * output_strides_for_copy[onnxruntime::narrow<size_t>(p.axis)];
    }
  }

  return Status::OK();
}

// core Compute() method for the 'Concat' kernel
Status Concat::Compute(OpKernelContext* ctx) const {
  // Number of input tensors to concatenate
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  // Compute values to be placed in the output tensor
  return ComputeImpl(p, ctx);
}

}  // namespace onnxruntime
