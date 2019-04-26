// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/concat.h"
#include "core/providers/common.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Concat,
    4,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

Status ConcatBase::PrepareForCompute(OpKernelContext* ctx, int input_count, Prepare& p) const {
  ORT_RETURN_IF_NOT(input_count >= 1, "Must have 1 or more inputs");
  const Tensor* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& inputs_0 = *tensor_pointer;
  const auto& inputs_0_dims = inputs_0.Shape().GetDims();
  const size_t inputs_0_rank = inputs_0_dims.size();
  ORT_RETURN_IF_NOT(inputs_0_rank > 0, "Cannot concatenate scalars");

  auto axis = HandleNegativeAxis(axis_, inputs_0.Shape().NumDimensions());

  // cache num of elements in tensor for later use
  // as it's expensive to call Size() on TensorShape over and over
  std::vector<size_t> tensor_num_elements(input_count);
  // Ensure all of the non concatenated axes match each other
  for (int index = 1; index < input_count; index++) {
    size_t num_elements = 1;
    tensor_pointer = ctx->Input<Tensor>(index);
    if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    auto& inputs_n = *tensor_pointer;
    const auto& inputs_n_dims = inputs_n.Shape().GetDims();
    const size_t inputs_n_rank = inputs_n_dims.size();
    ORT_ENFORCE(inputs_n_rank == inputs_0_rank, "Ranks of input data are different, cannot concatenate them, "
                "expected rank: ", std::to_string(inputs_0_rank), " got: ", std::to_string(inputs_n_rank));
    // Ensure all the other (non-concat) axes match
    for (int axis_index = 0; axis_index < inputs_0_rank; ++axis_index) {
      num_elements *= inputs_n_dims[axis_index];
      if (axis_index == axis)
        continue;
      ORT_RETURN_IF_NOT(inputs_n_dims[axis_index] == inputs_0_dims[axis_index],
                        "Non concat axis dimensions must match: Axis ", 
                        axis_index, " has mismatched dimensions of ", inputs_n_dims[axis_index], 
                        " and ", inputs_0_dims[axis_index]);
    }
    tensor_num_elements[index] = num_elements;
  }

  // Calculate the size of the concatenated axis, and verify all other dimensions match
  size_t concat_axis_size = 0;
  for (int index = 0; index < input_count; index++) {
    tensor_pointer = ctx->Input<Tensor>(index);
    concat_axis_size += tensor_pointer->Shape()[int(axis)];
  }

  // Calculate the shape of the output tensor
  std::vector<int64_t> dims(inputs_0_rank);
  size_t num_elements = 1; // cache size of the first input along the way
  for (int dimension_index = 0; dimension_index < inputs_0_rank; dimension_index++) {
    dims[dimension_index] = inputs_0_dims[dimension_index];
    num_elements *= inputs_0_dims[dimension_index];
  }
  tensor_num_elements[0] = num_elements;
  dims[axis] = concat_axis_size;
  TensorShape output_shape(dims);
 
  auto& concat_result = *ctx->Output(0, output_shape);
  p.output_tensor = &concat_result;
  p.output_num_elements = output_shape.Size();

  // if the output tensor is not going to hold any elements,
  // there is no need to proceed further
  if (p.output_num_elements == 0)
    return Status::OK();
    
  // The output_axis_pitch is the number of elements to add to move to the next split axis in the output
  p.output_axis_pitch = 1;
  for (auto i = int64_t(inputs_0_rank); i-- > axis;)
    p.output_axis_pitch *= dims[i];

  for (int input_index = 0; input_index < input_count; input_index++) {
    const Tensor* data_n_ptr = ctx->Input<Tensor>(input_index);
    auto& data_n = *data_n_ptr;

    ORT_RETURN_IF_NOT(data_n.DataType() == concat_result.DataType());

    // The input_axis_pitch is the number of elements to add to move to the next split axis in the input
    int64_t input_axis_pitch = 1;
    const auto& data_dims = data_n.Shape().GetDims();
    for (int i = static_cast<int>(inputs_0_rank); i-- > axis;)
      input_axis_pitch *= data_dims[i];

    p.inputs.push_back({&data_n, tensor_num_elements[input_index], input_axis_pitch});
  }

  return Status::OK();
}

Status Concat::Compute(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();

  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, input_count, p));

  // return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  auto is_string_type = ctx->Input<Tensor>(0)->DataType() == DataTypeImpl::GetType<std::string>();

  int64_t output_offset = 0;
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
    uint8_t* output = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());
    for (int idxCopy = 0; idxCopy < input_size / input_axis_pitch; ++idxCopy) {
      if (is_string_type) {
        for (int idxItem = 0; idxItem < input_axis_pitch; ++idxItem)
          reinterpret_cast<std::string*>(output)[output_offset + idxCopy * p.output_axis_pitch + idxItem] =
              reinterpret_cast<const std::string*>(input)[idxCopy * input_axis_pitch + idxItem];
      } else
        memcpy(
            output + (output_offset + idxCopy * p.output_axis_pitch) * element_bytes,
            input + idxCopy * input_axis_pitch * element_bytes,
            input_axis_pitch * element_bytes);
    }
    output_offset += input_axis_pitch;
  }
  return Status::OK();
}

}  // namespace onnxruntime
