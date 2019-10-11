// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/sequence/concat_from_sequence.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/TensorSeq.h"

using namespace onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    ConcatFromSequence,
    11,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes()),
    ConcatFromSequence);

Status StackInputs(const std::vector<Tensor>& inputs, int64_t axis) {
  // TODO: Implement logic to stack tensors
  return Status::OK();
}

// core Compute() method for the 'ConcatFromSequence' kernel
Status ConcatFromSequence::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<TensorSeq>(0);
  ORT_ENFORCE(X != nullptr, "Got nullptr for sequence input.");

  // number of input tensors in the Sequence to concatenate
  int input_count = static_cast<int>(X->tensors.size());

  // validate inputs and compute output (Concat mode)
  if (!stack_tensors) {
    return ValidateAndConcatenateInputs(context, input_count);
  } else {
    return StackInputs(X->tensors, axis_);
  }
}

}  // namespace onnxruntime
