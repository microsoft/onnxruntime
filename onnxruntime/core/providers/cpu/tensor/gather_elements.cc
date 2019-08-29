// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_elements.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    GatherElements,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElements);

// Some helpers needed for GatherElements op -

// The following method computes the offset in the flattened array
// using every axis except the inner dimension (as the offset is just 1)
// and the axis that 'GatherElements' is processing for as that requires the corresponding
// 'indices' value
// This prevents the need to compute this offset for every element within the same 'inner_dimension' chunk
// as this value is constant for the entire chunk
int64_t compute_base_offset(const std::vector<int64_t>& shape, const TensorPitches& pitches, int64_t skip_axis) {
  // in this context rank can never be < 1, so saving checking overhead
  auto loop_size = static_cast<int64_t>(shape.size()) - 1;

  int64_t base_offset = 0;

  for (int64_t i = 0; i < loop_size; ++i) {
    if (i != skip_axis)
      base_offset += (shape[i] * pitches[i]);
  }

  return base_offset;
}

// This method computes the number of 'inner_dimensions'
// Example: input = [2, 3]     output = 2
//          input = [3, 2, 4]  output = 3 * 2 = 6
//          input  = [2]       output = 1
int64_t calculate_num_inner_dim(const std::vector<int64_t>& dims) {
  // in this context rank can never be < 1, so saving checking overhead
  if (dims.size() == 1)
    return 1;

  int64_t num_inner_dims = 1;

  for (int64_t i = 0; i < static_cast<int64_t>(dims.size()) - 1; ++i) {
    num_inner_dims *= dims[i];
  }

  return num_inner_dims;
}

// This method computes increments over an 'inner_dimension'
// Example: current_dims = [0, 0, 0] tensor_dims = [1, 2, 2], then current_dims = [0, 1, 0]
//          current_dims = [0, 1, 0] tensor_dims = [1, 2, 2], then current_dims = [0, 0, 0]
void increment_over_inner_dim(std::vector<int64_t>& current_dims, std::vector<int64_t> tensor_dims) {
  // in this context rank can never be < 1, so saving checking overhead
  int64_t rank = static_cast<int64_t>(current_dims.size());

  // 'reset' innermost dimension value
  current_dims[0] = 0;

  // nothing to increment over
  if (rank == 1) {
    return;
  }

  int64_t current_axis = rank - 2;

  while (current_axis >= 0) {
    if (++current_dims[current_axis] != tensor_dims[current_axis])
      return;

    current_dims[current_axis] = 0;
    --current_axis;
  }
}

// parse indices_tensor and along the way validate its shape and contents
const std::vector<int64_t> parse_and_validate_indices_tensor(const Tensor* indices_tensor,
                                                             int64_t axis, const std::vector<int64_t>& input_shape) {
  const auto& indices_shape = indices_tensor->Shape().GetDims();
  int64_t indices_rank = static_cast<int64_t>(indices_shape.size());
  for (int64_t i = 0; i < indices_rank; ++i) {
    // for all axes except the axis of interest,
    // make sure that the corresponding 'indices' shape
    // value if within bounds of the corresponding 'data' shape
    if (i != axis) {
      if (indices_shape[i] < 0 || indices_shape[i] > indices_shape[i])
        ORT_THROW("GatherElements op: 'indices' shape should have values within bounds of 'data' shape");
    }
  }

  // first parse 'indices' data
  auto num_elements = indices_tensor->Shape().Size();
  std::vector<int64_t> indices_data;
  // reserving memory ahead as we know the size of the container
  indices_data.reserve(num_elements);
  if (indices_tensor->DataType() == DataTypeImpl::GetType<int32_t>()) {
    const auto* data = indices_tensor->Data<int32_t>();
    for (int64_t i = 0; i < num_elements; ++i)
      indices_data.push_back(data[i]);
  } else if (indices_tensor->DataType() == DataTypeImpl::GetType<int64_t>()) {
    const auto* data = indices_tensor->Data<int64_t>();
    for (int64_t i = 0; i < num_elements; ++i)
      indices_data.push_back(data[i]);
  } else {
    ORT_THROW("GatherElements op: Data type for 'indices' tensor must be 'int32_t' and 'int64_t'");
  }

  // validate 'indices' data
  // along the way 'fix' negative index values if within bounds
  int64_t lower_index_limit = -input_shape[axis];
  int64_t upper_index_limit = input_shape[axis] - 1;

  for (int64_t i = 0; i < num_elements; ++i) {
    if (indices_data[i] < lower_index_limit || indices_data[i] > upper_index_limit)
      ORT_THROW("GatherElements op: Value in indices must be within bounds [",
                lower_index_limit, " , ", upper_index_limit, "]");

    if (indices_data[i] < 0)
      indices_data[i] += lower_index_limit;
  }

  return indices_data;
}

Status GatherElements::CoreImplString(const Tensor* input_tensor, const Tensor* indices_tensor,
                                      Tensor* output_tensor, int64_t axis) const {
  const std::string* input_data = input_tensor->Data<std::string>();
  const std::vector<int64_t>& input_shape = input_tensor->Shape().GetDims();
  const int64_t input_rank = static_cast<int64_t>(input_tensor->Shape().NumDimensions());

  const std::vector<int64_t>& indices_data = parse_and_validate_indices_tensor(indices_tensor, axis, input_shape);
  const std::vector<int64_t>& indices_shape = indices_tensor->Shape().GetDims();

  std::string* output_data = output_tensor->MutableData<std::string>();

  const TensorPitches input_shape_pitches(*input_tensor);

  int64_t num_inner_dim = calculate_num_inner_dim(indices_shape);
  int64_t inner_dim_size = indices_shape[input_rank - 1];
  bool processing_inner_dim = (axis == input_rank - 1) ? true : false;

  int64_t base_offset = 0;
  int64_t indices_counter = -1;
  std::vector<int64_t> process_dims(input_rank, 0);

  int64_t output_counter = -1;

  if (!processing_inner_dim) {
    while (num_inner_dim-- != 0) {
      base_offset = compute_base_offset(process_dims, input_shape_pitches, axis);

      // process 1 chunk of 'inner dimension' length
      for (int64_t i = 0; i < inner_dim_size; ++i) {
        output_data[++output_counter] = input_data[base_offset + (indices_data[++indices_counter] * input_shape_pitches[axis]) + i];
      }

      increment_over_inner_dim(process_dims, indices_shape);
    }
  } else {
    while (num_inner_dim-- != 0) {
      base_offset = compute_base_offset(process_dims, input_shape_pitches, axis);

      // process 1 chunk of 'inner dimension' length
      for (int64_t i = 0; i < inner_dim_size; ++i) {
        output_data[++output_counter] = input_data[base_offset + indices_data[++indices_counter]];
      }

      increment_over_inner_dim(process_dims, indices_shape);
    }
  }

  return Status::OK();
}

Status GatherElements::CoreImpl(const Tensor* input_tensor, const Tensor* indices_tensor,
                                Tensor* output_tensor, int64_t axis) const {
  const char* input_data = reinterpret_cast<const char*>(input_tensor->DataRaw());
  const std::vector<int64_t>& input_shape = input_tensor->Shape().GetDims();
  const int64_t input_rank = static_cast<int64_t>(input_tensor->Shape().NumDimensions());

  const std::vector<int64_t>& indices_data = parse_and_validate_indices_tensor(indices_tensor, axis, input_shape);
  const std::vector<int64_t>& indices_shape = indices_tensor->Shape().GetDims();

  char* output_data = reinterpret_cast<char*>(output_tensor->MutableDataRaw());

  const TensorPitches input_shape_pitches(*input_tensor);

  int64_t num_inner_dim = calculate_num_inner_dim(indices_shape);
  int64_t inner_dim_size = indices_shape[input_rank - 1];
  bool processing_inner_dim = (axis == input_rank - 1) ? true : false;

  int64_t base_offset = 0;
  int64_t indices_counter = -1;
  std::vector<int64_t> process_dims(input_rank, 0);

  size_t element_size = input_tensor->DataType()->Size();

  if (!processing_inner_dim) {
    while (num_inner_dim-- != 0) {
      base_offset = compute_base_offset(process_dims, input_shape_pitches, axis);

      // process 1 chunk of 'inner dimension' length
      for (int64_t i = 0; i < inner_dim_size; ++i) {
        memcpy(output_data, input_data + (base_offset + (indices_data[++indices_counter] * input_shape_pitches[axis]) + i) * element_size, element_size);
        output_data += element_size;
      }

      increment_over_inner_dim(process_dims, indices_shape);
    }
  } else {
    while (num_inner_dim-- != 0) {
      base_offset = compute_base_offset(process_dims, input_shape_pitches, axis);

      // process 1 chunk of 'inner dimension' length
      for (int64_t i = 0; i < inner_dim_size; ++i) {
        // for innermost axis, input_shape_pitches[axis] = 1 (so no need to multiply)
        memcpy(output_data, input_data + (base_offset + indices_data[++indices_counter]) * element_size, element_size);
        output_data += element_size;
      }

      increment_over_inner_dim(process_dims, indices_shape);
    }
  }

  return Status::OK();
}

Status GatherElements::Compute(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const TensorShape& input_data_shape = input_tensor->Shape();
  int64_t input_data_rank = static_cast<int64_t>(input_data_shape.NumDimensions());

  const Tensor* indices_tensor = context->Input<Tensor>(1);
  const TensorShape& indices_shape = indices_tensor->Shape();
  int64_t indices_rank = static_cast<int64_t>(indices_shape.NumDimensions());

  if (input_data_rank < 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Rank of input 'data' needs to be >= 1");

  if (input_data_rank != indices_rank)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'");

  Tensor* output_tensor = context->Output(0, TensorShape(indices_shape));

  const auto& input_data_type = input_tensor->DataType();
  if (input_data_type != output_tensor->DataType())
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Data type of input 'data' should match the data type of the output");

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  int64_t axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  if (input_data_type == DataTypeImpl::GetType<std::string>())
    return CoreImplString(input_tensor, indices_tensor, output_tensor, axis);

  else
    return CoreImpl(input_tensor, indices_tensor, output_tensor, axis);
}

}  // namespace onnxruntime
