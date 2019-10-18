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

// core Compute() method for the 'ConcatFromSequence' kernel
Status ConcatFromSequence::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<TensorSeq>(0);
  ORT_ENFORCE(X != nullptr, "Got nullptr for sequence input.");

  // number of input tensors in the Sequence to concatenate
  int input_count = static_cast<int>(X->tensors.size());

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  const auto& input_tensors = X->tensors;

  std::vector<const Tensor*> input_tensor_pointers;
  input_tensor_pointers.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensor_pointers.push_back(&input_tensors[i]);
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensor_pointers, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  // Compute values to be placed in the output tensor
  return ComputeImpl(p);
}

}  // namespace onnxruntime
