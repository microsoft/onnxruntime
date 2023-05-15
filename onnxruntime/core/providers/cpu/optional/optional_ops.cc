// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_OPTIONAL_TYPE)

#include "optional_ops.h"
#include "core/framework/ort_value.h"
#include "core/framework/TensorSeq.h"
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

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(OptionalHasElement,
                                   15,
                                   17,
                                   KernelDefBuilder()
                                       .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                                       .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
                                   OptionalHasElement);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(OptionalGetElement,
                                   15,
                                   17,
                                   KernelDefBuilder()
                                       .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
                                       .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                                       // We may be able to re-use the input for the output as is unless the output
                                       // is a graph output. We provide this hint to the allocation planner
                                       // to make the re-use call.
                                       .Alias(0, 0),
                                   OptionalGetElement);

ONNX_CPU_OPERATOR_KERNEL(OptionalHasElement,
                         18,
                         KernelDefBuilder()
                             .TypeConstraint("O", DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypes())
                             .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
                         OptionalHasElement);

ONNX_CPU_OPERATOR_KERNEL(OptionalGetElement,
                         18,
                         KernelDefBuilder()
                             .TypeConstraint("O", DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypes())
                             .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                             // We may be able to re-use the input for the output as is unless the output
                             // is a graph output. We provide this hint to the allocation planner
                             // to make the re-use call.
                             .Alias(0, 0),
                         OptionalGetElement);

static void CopySequenceTensor(AllocatorPtr alloc,
                               const TensorSeq* src,
                               TensorSeq* tgt,
                               const DataTransferManager& data_transfer_mgr) {
  // The static allocation planner has deemed that the input can be re-used as the output
  // Analogy: Checking if data pointers for the input and output Tensors are the same
  // before proceeding to make the copy.
  if (src == tgt) {
    return;
  }

  tgt->SetType(src->DataType());
  tgt->Reserve(src->Size());

  auto in_tensor = src->begin();
  for (; in_tensor != src->end(); ++in_tensor) {
    auto& tensor = in_tensor->Get<Tensor>();
    Tensor tmp(tensor.DataType(), tensor.Shape(), alloc);
    // Using DataTransferManager here allows other non-CPU EPs to use this implementation of the sequence ops
    (void)data_transfer_mgr.CopyTensor(tensor, tmp);

    tgt->Add(std::move(tmp));
  }
}

static Status PropagateInputOrtValueToFirstOutput(const OrtValue* input_ort_value,
                                                  OpKernelContext* ctx,
                                                  const DataTransferManager& data_transfer_mgr) {
  if (input_ort_value->IsTensor()) {
    const auto* input_tensor = &input_ort_value->Get<Tensor>();
    auto* output_tensor = ctx->Output(0, input_tensor->Shape());

    // If the allocation planner had deemed that we re-use the input OrtValue
    // as the output OrtValue, the data pointers in the input_tensor and the
    // output_tensor will be the same and the copy is a no-op.
    // DataTransferManager.CopyTensor() already has such copy optimizations - so
    // just re-use it.
    ORT_RETURN_IF_ERROR(data_transfer_mgr.CopyTensor(*input_tensor, *output_tensor));
  } else if (input_ort_value->IsTensorSequence()) {
    const auto* input_tensor_sequence = &input_ort_value->Get<TensorSeq>();
    auto* output_tensor_sequence = ctx->Output<TensorSeq>(0);

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

    // If the allocation planner had deemed that we re-use the input OrtValue
    // as the output OrtValue, the pointers of the source TensorSeq and the
    // target TensorSeq will be the same and the copy is a no-op.
    // CopySequenceTensor() already has such copy optimizations
    CopySequenceTensor(alloc, input_tensor_sequence, output_tensor_sequence, data_transfer_mgr);

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
    const DataTransferManager& data_transfer_mgr = Info().GetDataTransferManager();
    ORT_RETURN_IF_ERROR(PropagateInputOrtValueToFirstOutput(input_ort_value, ctx, data_transfer_mgr));

  } else {  // No input was provided - we use the type proto to construct the output OrtValue

    if (!CheckValidTypeProto(*type_proto_)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The TypeProto attribute in the Optional op ",
                             "can only be of type(tensor) or (seq(tensor))");
    }

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
  if (input_ort_value)
    output_tensor->MutableData<bool>()[0] = input_ort_value->IsAllocated();
  else
    output_tensor->MutableData<bool>()[0] = false;

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
  const DataTransferManager& data_transfer_mgr = Info().GetDataTransferManager();
  ORT_RETURN_IF_ERROR(PropagateInputOrtValueToFirstOutput(input_ort_value, ctx, data_transfer_mgr));

  return Status::OK();
}

}  // namespace onnxruntime

#endif
