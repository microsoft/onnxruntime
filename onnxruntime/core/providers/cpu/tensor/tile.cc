// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "gsl/gsl"
#include "core/providers/cpu/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ::onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Tile,
    6,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<int8_t>(),
                                            DataTypeImpl::GetTensorType<int16_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>(),
                                            DataTypeImpl::GetTensorType<int64_t>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<uint16_t>(),
                                            DataTypeImpl::GetTensorType<uint32_t>(),
                                            DataTypeImpl::GetTensorType<uint64_t>(),
                                            DataTypeImpl::GetTensorType<bool>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Tile);

Status TileCoreForFixedSizeTypes(const Tensor& input_tensor, Tensor& output_tensor, const int64_t* repeats, TensorAxisCounters& input_counters, const TensorPitches& output_pitches, size_t element_size) {
  const auto& input_shape = input_tensor.Shape().GetDims();
  const size_t dimension_count = input_shape.size();

  const auto* input = reinterpret_cast<const uint8_t*>(input_tensor.DataRaw());
  auto* output = reinterpret_cast<uint8_t*>(output_tensor.MutableDataRaw());

  // some helper variables that will be used along the way
  size_t block_size = 0;
  int64_t num_repeats = 0;
  const uint8_t* copy = nullptr;
  const int64_t innermost_dim = input_shape[dimension_count - 1];

  while (input_counters) {
    // Copy the input data over
    block_size = innermost_dim * element_size;
    memcpy(output, input, block_size);
    output += block_size;
    input += block_size;

    // Tile data for the innermost axis
    copy = output - block_size;
    num_repeats = repeats[dimension_count - 1] - 1;
    for (int64_t repeat = 0; repeat < num_repeats; ++repeat) {
      memcpy(output, copy, block_size);
      output += block_size;
    }

    // Tile data for other axes
    while (input_counters.Increment()) {
      ptrdiff_t pitch = output_pitches[input_counters.Axis()] * input_shape[input_counters.Axis()];
      block_size = pitch * element_size;
      copy = output - block_size;
      num_repeats = repeats[input_counters.Axis()] - 1;
      for (int64_t repeat = 0; repeat < num_repeats; ++repeat) {
        memcpy(output, copy, block_size);
        output += block_size;
      }
    }
  }
  return Status::OK();
}

Status Tile::Compute(OpKernelContext* ctx) const {
  const auto* tensor_pointer = ctx->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "Input count of Tile OP mismatch, the first one is empty");
  const Tensor& input_tensor = *tensor_pointer;
  const auto& input_shape = input_tensor.Shape();
  const size_t input_rank = input_shape.NumDimensions();
  tensor_pointer = ctx->Input<Tensor>(1);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "Input count of Tile OP mismatch, the second one is empty");
  const Tensor& repeats_tensor = *tensor_pointer;
  if (input_rank < 1)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "the tensor to be tiled using Tile OP must be atleast 1 dimensional");
  if (repeats_tensor.Shape().NumDimensions() != 1)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must be 1 dimensional");
  if (size_t(repeats_tensor.Shape().Size()) != input_rank)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must have the same length as the 'input' tensor");

  // Calculate the shape of the output tensor
  auto* repeats = repeats_tensor.template Data<int64_t>();
  std::vector<int64_t> output_dims = input_shape.GetDims();
  for (size_t axis = 0; axis < input_rank; axis++) {
    output_dims[axis] *= repeats[axis];
  }

  TensorShape outputShape(output_dims);
  auto& output_tensor = *ctx->Output(0, outputShape);

  // Repeat tensor input can have 0 as a valid value
  // check if the computed outputshape size is 0 and
  // return an empty tensor if so.
  if (outputShape.Size() == 0) {
    return Status::OK();
  }

  TensorAxisCounters input_counters(input_tensor);
  TensorPitches output_pitches(output_tensor);

  static_assert(sizeof(float) == sizeof(int32_t), "Float and Int32 are of different sizes");
  static_assert(sizeof(double) == sizeof(int64_t), "Double and Int64 are of different sizes");

  if (input_tensor.IsDataType<float>() ||
      input_tensor.IsDataType<int32_t>() ||
      input_tensor.IsDataType<uint32_t>())
    return TileCoreForFixedSizeTypes(input_tensor, output_tensor, repeats, input_counters, output_pitches, sizeof(float));

  if (input_tensor.IsDataType<double>() || input_tensor.IsDataType<int64_t>() ||
      input_tensor.IsDataType<uint64_t>())
    return TileCoreForFixedSizeTypes(input_tensor, output_tensor, repeats, input_counters, output_pitches, sizeof(double));

  else if (input_tensor.IsDataType<int8_t>() ||
           input_tensor.IsDataType<uint8_t>())
    return TileCoreForFixedSizeTypes(input_tensor, output_tensor, repeats, input_counters, output_pitches, sizeof(int8_t));

  if (input_tensor.IsDataType<int16_t>() || input_tensor.IsDataType<uint16_t>())
    return TileCoreForFixedSizeTypes(input_tensor, output_tensor, repeats, input_counters, output_pitches, sizeof(int16_t));

  else if (input_tensor.IsDataType<bool>())
    return TileCoreForFixedSizeTypes(input_tensor, output_tensor, repeats, input_counters, output_pitches, sizeof(bool));

  // TODO: Support 'string' and 'float16' types for completeness
  else
    ORT_THROW("Tile doesn't have an implementation yet for the type: ", input_tensor.DataType());
}
}  // namespace onnxruntime
