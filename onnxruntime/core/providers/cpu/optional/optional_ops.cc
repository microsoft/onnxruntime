// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optional_ops.h"
#include "core/framework/ort_value.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(Optional,
                         15,
                         KernelDefBuilder()
                             .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                             .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                             // We may be able to re-use the input for the output as is unless the output
                             // is a graph output. We provide this hint to the allocation planner
                             // to make the re-use call.
                             .Alias(0, 0),
                         Optional);

ONNX_CPU_OPERATOR_KERNEL(OptionalHasElement,
                         15,
                         KernelDefBuilder()
                             .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                             .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
                         OptionalHasElement);

ONNX_CPU_OPERATOR_KERNEL(OptionalGetElement,
                         15,
                         KernelDefBuilder()
                             .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                             .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                             // We may be able to re-use the input for the output as is unless the output
                             // is a graph output. We provide this hint to the allocation planner
                             // to make the re-use call.
                             .Alias(0, 0),
                         OptionalGetElement);

static void CopySequenceTensor(AllocatorPtr alloc,
                               const TensorSeq* src,
                               TensorSeq* tgt) {
  // The static allocation planner has deemed that the input can be re-used as the output
  // Analogy: Checking if data pointers for the input and output Tensors are the same
  // before proceeding to make the copy.
  if (src == tgt) {
    return;
  }

  tgt->SetType(src->DataType());

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(src->Size());

  auto in_tensor = src->begin();
  for (; in_tensor != src->end(); ++in_tensor) {
    Tensor tmp(in_tensor->DataType(), onnxruntime::TensorShape(in_tensor->Shape()), alloc);
    CopyCpuTensor(&*in_tensor, &tmp);
    output_tensors.push_back(std::move(tmp));
  }

  tgt->SetElements(std::move(output_tensors));
}

static Status PropagateInputOrtValueToFirstOutput(const OrtValue* input_ort_value,
                                                  OpKernelContext* ctx) {
  if (input_ort_value->IsTensor()) {
    const auto* input_tensor = &input_ort_value->Get<Tensor>();
    auto* output_tensor = ctx->Output(0, input_tensor->Shape());

    // If the allocation planner had deemed that we re-use the input OrtValue
    // as the output OrtValue, the data pointers in the input_tensor and the
    // output_tensor will be the same and the copy is a no-op.
    // CopyCpuTensor() already has such copy optimizations - so
    // just re-use it.
    CopyCpuTensor(input_tensor, output_tensor);

  } else if (input_ort_value->IsTensorSequence()) {
    const auto* input_tensor_sequence = &input_ort_value->Get<TensorSeq>();
    auto* output_tensor_sequence = ctx->Output<TensorSeq>(0);

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

    // If the allocation planner had deemed that we re-use the input OrtValue
    // as the output OrtValue, the pointers of the source TensorSeq and the
    // target TensorSeq will be the same and the copy is a no-op.
    // CopySequenceTensor() already has such copy optimizations
    CopySequenceTensor(alloc, input_tensor_sequence, output_tensor_sequence);

  } else {
    // Will not reach here
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Only Optional type OrtValues containing Tensors "
                           "and Sequence Tensors are acceptable");
  }

  return Status::OK();
}

static bool CheckValidTypeProto(const onnx::TypeProto& tp) {
  // Optional types can currently be Tensors or SequenceTensors only
  // Ensure that the TypeProto holds those types

  // TODO: Should we also ensure that element type for tensor is set ?
  return (tp.has_tensor_type()) ||
         (tp.has_sequence_type() &&
          tp.sequence_type().elem_type().has_tensor_type());
}

Status Optional::Compute(OpKernelContext* ctx) const {
  const auto* input_ort_value = ctx->GetInputOrtValue(0);

  if (input_ort_value != nullptr) {
    // An input was provided by the user - so just propagate it to the output
    ORT_RETURN_IF_ERROR(PropagateInputOrtValueToFirstOutput(input_ort_value, ctx));

  } else {  // No input was provided - we use the type proto to construct the output OrtValue

    CheckValidTypeProto(*type_proto_);

    // type is either Tensor or TensorSeq (we have validated this already in CheckValidTypeProto())
    if (type_proto_->has_tensor_type()) {
      ctx->OutputOptionalWithoutData<Tensor>(0);
    } else {
      ctx->OutputOptionalWithoutData<TensorSeq>(0);
    }
  }

  return Status::OK();
}

Status OptionalHasElement::Compute(OpKernelContext* ctx) const {
  const auto* input_ort_value = ctx->GetInputOrtValue(0);

  // Output is a scalar
  auto* output_tensor = ctx->Output(0, {});
  output_tensor->MutableData<bool>()[0] = input_ort_value->IsAllocated();

  return Status::OK();
}

Status OptionalGetElement::Compute(OpKernelContext* ctx) const {
  const auto* input_ort_value = ctx->GetInputOrtValue(0);

  if (!input_ort_value->IsAllocated()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Trying to use OptionalGetElement on an optional type "
                           "OrtValue which contains no data");
  }

  // Propagate input to the output
  ORT_RETURN_IF_ERROR(PropagateInputOrtValueToFirstOutput(input_ort_value, ctx));

  return Status::OK();
}

}  // namespace onnxruntime
