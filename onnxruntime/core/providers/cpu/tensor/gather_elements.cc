// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
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

// Compute the number of 'inner_dimension' elements
// Example: input = [2, 3]     output = 2
//          input = [3, 2, 4]  output = 3 * 2 = 6
//          input  = [2]       output = 1
static int64_t CalculateInnerDimCount(const TensorShape& dims) {
  // Rank can never be < 1, no need to check
  return dims.SizeToDimension(dims.NumDimensions() - 1);
}

// Computes the offset into the input array given the inner_dim count
//
// If the input indices tensor matched the input tensor this would simply just be inner_dim_size * inner_dim
// But since the indices tensor can be smaller we need to do the math based on the smaller size, as the
// output tensor size matches the input indices tensor size.
//
// The calculation is fairly straightforward, starting with the second to innermost axis we muldiv the inner_dim
// by the indices shape size. We also skip this calculation on the skip_axis as that's handled elsewhere.
//
static inline size_t CalculateOffset(size_t inner_dim, const TensorPitches& input_shape_pitches, size_t skip_axis,
                                     const TensorShape& indices_shape) {
  // in this context, rank can never be < 1, so saving checking overhead
  size_t rank = input_shape_pitches.size();

  // nothing to increment over
  if (rank == 1) {
    return 0;
  }

  size_t base_offset = 0;

  for (size_t axis = rank - 1; axis-- > 0;) {
    auto dim = indices_shape[axis];
    if (axis != skip_axis)
      base_offset += SafeInt<size_t>(inner_dim % dim) * input_shape_pitches[axis];
    inner_dim /= SafeInt<size_t>(dim);
  }

  return base_offset;
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
    ORT_THROW("Index out of range");
  return index;
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#ifdef HAS_CLASS_MEMACCESS
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#endif

template <typename Tin>
static void core_impl(const Tensor* input_tensor, const Tensor* indices_tensor, Tensor* output_tensor, int64_t axis,
                      concurrency::ThreadPool* ttp) {
  // Get input & output pointers
  const int8_t* input_data = reinterpret_cast<const int8_t*>(input_tensor->DataRaw());
  int8_t* output_data = reinterpret_cast<int8_t*>(output_tensor->MutableDataRaw());
  size_t element_size = input_tensor->DataType()->Size();
  bool is_string = input_tensor->IsDataTypeString();

  const int64_t input_rank = static_cast<int64_t>(input_tensor->Shape().NumDimensions());

  const TensorShape& indices_shape = indices_tensor->Shape();
  size_t num_inner_dim = onnxruntime::narrow<size_t>(CalculateInnerDimCount(indices_shape));
  size_t inner_dim_size = onnxruntime::narrow<size_t>(indices_shape[SafeInt<size_t>(input_rank) - 1]);
  const Tin* indices_data = indices_tensor->Data<Tin>();

  const TensorPitches input_shape_pitches(*input_tensor);
  int64_t axis_pitch = input_shape_pitches[onnxruntime::narrow<size_t>(axis)];
  int64_t axis_size = input_tensor->Shape()[onnxruntime::narrow<size_t>(axis)];

  bool innermost_axis = axis == input_rank - 1;
  bool index_error = false;

  auto MainLoop = [&](auto* output_data, auto* input_data) {
    auto BatchWork = [&](size_t inner_dim) {
      ORT_TRY {
        auto output = output_data + inner_dim_size * inner_dim;
        auto input = input_data + CalculateOffset(inner_dim, input_shape_pitches, onnxruntime::narrow<size_t>(axis), indices_shape);
        auto indices = indices_data + inner_dim_size * inner_dim;

        if (innermost_axis) {
          for (size_t i = 0; i < inner_dim_size; i++)
            output[i] = input[GetIndex(i, indices, axis_size)];
        } else {
          for (size_t i = 0; i < inner_dim_size; i++)
            output[i] = input[GetIndex(i, indices, axis_size) * axis_pitch + i];
        }
      }
      ORT_CATCH(const std::exception&) {
        index_error = true;
      }
    };

    concurrency::ThreadPool::TryBatchParallelFor(ttp, num_inner_dim, BatchWork, 0);
  };

  // Iterate over the elements based on the element size (or if it's a string). For everything but strings
  // we do a binary copy, so handling the 1, 2, 4, and 8 byte sizes covers all cases
  if (is_string)
    MainLoop(reinterpret_cast<std::string*>(output_data), reinterpret_cast<const std::string*>(input_data));
  else if (element_size == sizeof(uint32_t))
    MainLoop(reinterpret_cast<uint32_t*>(output_data), reinterpret_cast<const uint32_t*>(input_data));
  else if (element_size == sizeof(uint16_t))
    MainLoop(reinterpret_cast<uint16_t*>(output_data), reinterpret_cast<const uint16_t*>(input_data));
  else if (element_size == sizeof(uint8_t))
    MainLoop(reinterpret_cast<uint8_t*>(output_data), reinterpret_cast<const uint8_t*>(input_data));
  else if (element_size == sizeof(uint64_t))
    MainLoop(reinterpret_cast<uint64_t*>(output_data), reinterpret_cast<const uint64_t*>(input_data));
  else
    ORT_THROW("GatherElements op: Unsupported tensor type, size:", element_size);

  if (index_error)
    ORT_THROW("GatherElements op: Out of range value in index tensor");
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
      if (indices_shape[onnxruntime::narrow<size_t>(i)] < 0 || indices_shape[onnxruntime::narrow<size_t>(i)] > input_data_shape[onnxruntime::narrow<size_t>(i)])
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "GatherElements op: 'indices' shape should have values within bounds of 'data' shape. "
                               "Invalid value in indices shape is: ",
                               indices_shape[onnxruntime::narrow<size_t>(i)]);
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

  auto* ttp = context->GetOperatorThreadPool();
  if (indices_tensor->IsDataType<int32_t>())
    core_impl<int32_t>(input_tensor, indices_tensor, output_tensor, axis, ttp);
  else
    core_impl<int64_t>(input_tensor, indices_tensor, output_tensor, axis, ttp);

  return Status::OK();
}

}  // namespace onnxruntime
