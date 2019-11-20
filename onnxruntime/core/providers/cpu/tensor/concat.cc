// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/concat.h"
#include "core/providers/common.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Concat,
    4,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_KERNEL(
    Concat,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

// this method will be shared between 'Concat' (CPU and GPU) and 
// 'ConcatFromSequence' ('concat' and 'stack' modes) to validate inputs
Status ConcatBase::PrepareForCompute(OpKernelContext* ctx,
                                     const std::vector<const Tensor*>& input_tensors,
                                     Prepare& p) const {
  int input_count = static_cast<int>(input_tensors.size());
  // Must have atleast one input to concat
  ORT_RETURN_IF_NOT(input_count >= 1, "Must have 1 or more inputs");

  const Tensor* tensor_pointer = input_tensors[0];
  if (tensor_pointer == nullptr)
    return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

  const Tensor& inputs_0 = *tensor_pointer;
  const auto& inputs_0_dims = inputs_0.Shape().GetDims();
  const size_t inputs_0_rank = inputs_0_dims.size();

  // Cannot concatenate scalars (but they can be stacked)
  if (!is_stack_)
    ORT_RETURN_IF_NOT(inputs_0_rank > 0, "Cannot concatenate scalars");

  // Handle and fix negative axis
  // In 'stack' mode, the accepted range depends on the output rank (which is one more than the input rank)
  p.axis = static_cast<uint64_t>(HandleNegativeAxis(axis_, !is_stack_ ? inputs_0_rank : inputs_0_rank + 1));

  // Note if input tensor is empty for later use (it's expensive to call Size() on TensorShape)
  std::vector<int64_t> input_tensor_sizes(input_count);
  // Assign the number of values in the first input tensor
  input_tensor_sizes[0] = inputs_0.Shape().Size();

  // Ensure all of the non concatenated axes match each other
  for (int index = 1; index < input_count; index++) {
    tensor_pointer = input_tensors[index];
    if (tensor_pointer == nullptr)
      return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    auto& inputs_n = *tensor_pointer;
    const auto& inputs_n_dims = inputs_n.Shape().GetDims();
    const size_t inputs_n_rank = inputs_n_dims.size();
    ORT_ENFORCE(inputs_n_rank == inputs_0_rank,
                "Ranks of input data are different, cannot concatenate them. expected rank: ",
                inputs_0_rank, " got: ", inputs_n_rank);
    // Ensure all the other (non-concat) axes match
    int64_t tensor_size = 1;
    for (size_t axis_index = 0; axis_index < inputs_0_rank; ++axis_index) {
      auto dim_value = inputs_n_dims[axis_index];
      tensor_size *= dim_value;

      // In 'concat' mode, the axis to be concatenated may be different
      // But in 'stack' mode, all input shapes must be the same and must be validated
      if (!is_stack_ && axis_index == p.axis)
        continue;

      ORT_RETURN_IF_NOT(dim_value == inputs_0_dims[axis_index],
                        "Non concat axis dimensions must match: Axis ",
                        axis_index, " has mismatched dimensions of ", dim_value,
                        " and ", inputs_0_dims[axis_index]);
    }

    input_tensor_sizes[index] = tensor_size;  //assign the computed size of the input tensor
  }

  // Calculate the shape of the output tensor
  std::vector<int64_t> output_dims = inputs_0_dims;
  // 'Concat' mode
  if (!is_stack_) {
    // While concating, the rank of the output is the same as the input rank(s)

    // Calculate the size of the concatenated axis
    size_t concat_axis_size = 0;
    for (int64_t index = 0; index < input_count; index++) {
      concat_axis_size += input_tensors[index]->Shape()[static_cast<int>(p.axis)];
    }

    output_dims[p.axis] = concat_axis_size;
  } else {  // 'Stack' mode
    // While stacking, the rank of the output is one more than the input rank(s).
    // Stacking may be thought of as adding an unit dimension (of value 1) in the input tensors,
    // and concatenating them on thie new axis.
    // The value in the corresponding axis of the output will be the number of inputs that are being stacked.
    output_dims.insert(output_dims.begin() + p.axis, static_cast<int64_t>(input_count));
  }

  TensorShape output_shape(output_dims);

  // Create output tensor
  p.output_tensor = &(*ctx->Output(0, output_shape));

  // Make note if output tensor is going to be empty
  p.output_num_elements = output_shape.Size();

  // No need to proceed further if output is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  // The output_axis_pitch is the number of elements to add to move to the next split axis in the output.
  // Can handle stacking as well.
  p.output_axis_pitch = 1;
  auto output_rank = !is_stack_ ? inputs_0_rank : inputs_0_rank + 1;
  for (size_t i = output_rank; i-- > p.axis;) {
    p.output_axis_pitch *= output_dims[i];
  }

  // Fill the 'Prepare' struct with available information
  p.inputs.reserve(input_count);
  for (int input_index = 0; input_index < input_count; input_index++) {
    const Tensor* data_n_ptr = input_tensors[input_index];
    auto& data_n = *data_n_ptr;

    // Type sanity check (Make sure we are working on homogeneous types)
    ORT_RETURN_IF_NOT(data_n.DataType() == p.output_tensor->DataType());

    // The input_axis_pitch is the number of elements to add to move to the next split axis in the input
    // Can handle stacking as well (as the "new dummy dimension" in the input is of unit value).
    // TODO: Minor Optimization possibility: This input_axis_patch will be common across all inputs
    // in 'ConcatFromSequence' (stack mode). They have to be computed for each input only while concatenating.
    int64_t input_axis_pitch = 1;
    const auto& data_dims = data_n.Shape().GetDims();
    for (size_t i = inputs_0_rank; i-- > p.axis;) {
      input_axis_pitch *= data_dims[i];
    }

    p.inputs.push_back({&data_n, input_axis_pitch, input_tensor_sizes[input_index]});
  }

  // Make note if the input Tensors of type 'string'
  p.is_string_type = p.inputs[0].tensor->IsDataTypeString();

  return Status::OK();
}

// This method computes the output tensor for Concat/ConcatFromSequence ops
Status ConcatBase::ComputeImpl(Prepare& p) const {
  int input_count = static_cast<int>(p.inputs.size());
  int64_t initial_output_offset = 0;  // initial offset for each input
  auto element_bytes = p.output_tensor->DataType()->Size();
  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];

    // no data in this tensor - so skip it
    if (prep.num_elements == 0)
      continue;

    auto input_axis_pitch = prep.axis_pitch;
    const uint8_t* input = static_cast<const uint8_t*>(prep.tensor->DataRaw());

    auto input_size = prep.num_elements;

    // Copy the data across. For every 'input_axis_pitch' values copied, we move over by the 'output_axis_pitch'
    // TODO: Optimization possibility: There are cases where we simply need to "merge" raw buffers and this 
    // could be done without the pointer house-keeping as below. Some scenarios whether this is possible are:
    // 1) Concatenating on input axis = 0
    // 2) Stacking on output axis = 0
    // 3) Stacking scalars
    uint8_t* output = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());
    int64_t cur_out_offset = 0;
    int64_t cur_in_offset = 0;
    for (size_t idx_copy = 0, end = input_size / input_axis_pitch; idx_copy < end; ++idx_copy) {
      if (p.is_string_type) {
        size_t out = initial_output_offset + cur_out_offset;
        for (int idx_item = 0; idx_item < input_axis_pitch; ++idx_item) {
          reinterpret_cast<std::string*>(output)[out + idx_item] =
              reinterpret_cast<const std::string*>(input)[cur_in_offset + idx_item];
        }
      } else {
        memcpy(
            output + (initial_output_offset + cur_out_offset) * element_bytes,
            input + cur_in_offset * element_bytes,
            input_axis_pitch * element_bytes);
      }

      cur_out_offset += p.output_axis_pitch;
      cur_in_offset += input_axis_pitch;
    }

    initial_output_offset += input_axis_pitch;
  }

  return Status::OK();
}

// core Compute() method for the 'Concat' kernel
Status Concat::Compute(OpKernelContext* ctx) const {
  // Number of input tensors to concatenate
  auto input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  std::vector<const Tensor*> input_tensors;
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
  return ComputeImpl(p);
}

}  // namespace onnxruntime
