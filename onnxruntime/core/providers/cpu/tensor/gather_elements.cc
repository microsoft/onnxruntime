// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_elements.h"
#include "onnxruntime_config.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    GatherElements,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElements);

ONNX_CPU_OPERATOR_KERNEL(
    GatherElements,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElements);

// Some helpers needed for GatherElements op -

// The following method computes the offset in the flattened array
// using every axis except the inner dimension (as the offset is just 1)
// and the axis that 'GatherElements' is processing for as that requires the corresponding
// 'indices' value
// This prevents the need to compute this offset for every element within the same 'inner_dimension' chunk
// as this value just differs by 1 for the chunk elements and we can have this cached and re-use as needed
static inline size_t compute_base_offset(const TensorShapeVector& shape, const TensorPitches& pitches, int64_t skip_axis) {
  // in this context, rank can never be < 1, so saving checking overhead
  auto loop_size = static_cast<int64_t>(shape.size()) - 1;

  size_t base_offset = 0;

  for (int64_t i = 0; i < loop_size; ++i) {
    if (i != skip_axis)
      base_offset += shape[i] * pitches[i];
  }

  return base_offset;
}

// This method computes the number of 'inner_dimension' chunks
// Example: input = [2, 3]     output = 2
//          input = [3, 2, 4]  output = 3 * 2 = 6
//          input  = [2]       output = 1
static int64_t calculate_num_inner_dim(const TensorShape& dims) {
  // in this context, rank can never be < 1, so saving checking overhead
  return dims.SizeToDimension(dims.NumDimensions() - 1);
}

// This method computes increments over an 'inner_dimension'
// Example 1: current_dims = [0, x] tensor_dims = [3, 1], then current_dims = [1, x]
//            current_dims = [1, x] tensor_dims = [3, 1], then current_dims = [2, x]
//            current_dims = [2, x] tensor_dims = [3, 1], then current_dims = [0, x]

// Example 2: current_dims = [0, 0, x] tensor_dims = [1, 2, 2], then current_dims = [0, 1, x]
//            current_dims = [0, 1, x] tensor_dims = [1, 2, 2], then current_dims = [0, 0, x]
static inline void increment_over_inner_dim(TensorShapeVector& current_dims, const TensorShape& tensor_dims) {
  // in this context, rank can never be < 1, so saving checking overhead
  int64_t rank = static_cast<int64_t>(current_dims.size());

  // 'reset' innermost dimension value
  current_dims[rank - 1] = 0;

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

#if defined(_MSC_VER)
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline)) inline
#endif

template <typename T>
FORCEINLINE int64_t GetIndex(size_t i, const T* indices, int64_t axis_size) {
  int64_t index = indices[i];
  if (index < 0)  // Handle negative indices
    index += axis_size;
  if (std::make_unsigned_t<T>(index) >= std::make_unsigned_t<T>(axis_size))
    ORT_THROW("GatherElements op: Value in indices must be within bounds [-", axis_size, " , ", axis_size - 1, "]. Actual value is ", indices[i]);
  return index;
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#ifdef HAS_CLASS_MEMACCESS
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#endif

template <typename Tin>
static void core_impl(const Tensor* input_tensor, const Tensor* indices_tensor, Tensor* output_tensor, int64_t axis) {
  // Get input & output pointers
  const int8_t* input_data = reinterpret_cast<const int8_t*>(input_tensor->DataRaw());
  int8_t* output_data = reinterpret_cast<int8_t*>(output_tensor->MutableDataRaw());
  size_t element_size = input_tensor->DataType()->Size();
  bool is_string = input_tensor->IsDataTypeString();

  const int64_t input_rank = static_cast<int64_t>(input_tensor->Shape().NumDimensions());

  const TensorShape& indices_shape = indices_tensor->Shape();
  size_t num_inner_dim = calculate_num_inner_dim(indices_shape);
  size_t inner_dim_size = indices_shape[input_rank - 1];
  const Tin* indices = indices_tensor->Data<Tin>();

  TensorShapeVector process_dims(input_rank, 0);
  const TensorPitches input_shape_pitches(*input_tensor);
  int64_t axis_pitch = input_shape_pitches[axis];
  int64_t axis_size = input_tensor->Shape()[axis];

  auto DoAxis = [](auto* output, auto* input, auto* indices, size_t inner_dim_size, int64_t axis_size, int64_t axis_pitch) {
    for (size_t i = 0; i < inner_dim_size; i++)
      output[i] = input[GetIndex(i, indices, axis_size) * axis_pitch + i];
  };

  // Special case required for innermost axis, no axis_pitch multiply needed or adding i
  auto DoInnermostAxis = [](auto* output, auto* input, auto* indices, size_t inner_dim_size, int64_t axis_size, int64_t /*axis_pitch*/) {
    for (size_t i = 0; i < inner_dim_size; i++)
      output[i] = input[GetIndex(i, indices, axis_size)];
  };

  auto MainLoop = [&](auto* output, auto* input, auto LoopFn) {
    while (num_inner_dim-- != 0) {
      LoopFn(output, input + compute_base_offset(process_dims, input_shape_pitches, axis), indices, inner_dim_size, axis_size, axis_pitch);
      output += inner_dim_size;
      indices += static_cast<Tin>(inner_dim_size);
      increment_over_inner_dim(process_dims, indices_shape);
    }
  };

  // Iterate over the elements based on the element size (or if it's a string). For everything but strings
  // we do a binary copy, so handling the 1, 2, 4, and 8 byte sizes covers all cases
  auto SwitchOnSizes = [&](auto LoopFn) {
    if (is_string)
      MainLoop(reinterpret_cast<std::string*>(output_data), reinterpret_cast<const std::string*>(input_data), LoopFn);
    else if (element_size == sizeof(uint32_t))
      MainLoop(reinterpret_cast<uint32_t*>(output_data), reinterpret_cast<const uint32_t*>(input_data), LoopFn);
    else if (element_size == sizeof(uint16_t))
      MainLoop(reinterpret_cast<uint16_t*>(output_data), reinterpret_cast<const uint16_t*>(input_data), LoopFn);
    else if (element_size == sizeof(uint8_t))
      MainLoop(reinterpret_cast<uint8_t*>(output_data), reinterpret_cast<const uint8_t*>(input_data), LoopFn);
    else if (element_size == sizeof(uint64_t))
      MainLoop(reinterpret_cast<uint64_t*>(output_data), reinterpret_cast<const uint64_t*>(input_data), LoopFn);
    else
      ORT_THROW("GatherElements op: Unsupported tensor type, size:", element_size);
  };

  if (axis == input_rank - 1)
    SwitchOnSizes(DoInnermostAxis);
  else
    SwitchOnSizes(DoAxis);
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

Status GatherElements::ValidateInputShapes(const TensorShape& input_data_shape,
                                           const TensorShape& indices_shape,
                                           int64_t axis) {
  int64_t input_data_rank = static_cast<int64_t>(input_data_shape.NumDimensions());
  int64_t indices_rank = static_cast<int64_t>(indices_shape.NumDimensions());

  // GatherElements cannot operate on scalars
  if (input_data_rank < 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Cannot operate on scalar input");

  // The ranks of the inputs must be the same
  if (input_data_rank != indices_rank)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'");

  // Except for the axis of interest all other dim values of the 'indices' input must be within bounds
  // of the corresponding 'data' input dim value
  for (int64_t i = 0; i < indices_rank; ++i) {
    // for all axes except the axis of interest,
    // make sure that the corresponding 'indices' shape
    // value is within bounds of the corresponding 'data' shape
    if (i != axis) {
      if (indices_shape[i] < 0 || indices_shape[i] > input_data_shape[i])
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "GatherElements op: 'indices' shape should have values within bounds of 'data' shape. "
                               "Invalid value in indices shape is: ",
                               indices_shape[i]);
    }
  }

  return Status::OK();
}

Status GatherElements::Compute(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const TensorShape& input_data_shape = input_tensor->Shape();

  const Tensor* indices_tensor = context->Input<Tensor>(1);
  const TensorShape& indices_shape = indices_tensor->Shape();

  int64_t axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  auto status = ValidateInputShapes(input_data_shape, indices_shape, axis);
  if (!status.IsOK())
    return status;

  Tensor* output_tensor = context->Output(0, indices_shape);

  const auto& input_data_type = input_tensor->DataType();
  if (input_data_type != output_tensor->DataType())
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GatherElements op: Data type of input 'data' should match the data type of the output");

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  if (indices_tensor->IsDataType<int32_t>())
    core_impl<int32_t>(input_tensor, indices_tensor, output_tensor, axis);
  else
    core_impl<int64_t>(input_tensor, indices_tensor, output_tensor, axis);

  return Status::OK();
}

}  // namespace onnxruntime
