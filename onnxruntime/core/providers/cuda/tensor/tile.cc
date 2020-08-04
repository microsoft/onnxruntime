// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"
#include "tile_impl.h"
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Tile,
    kOnnxDomain,
    6,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(1)
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Tile);

Status Tile::ComputeInternal(OpKernelContext* ctx) const {
  auto& input_tensor = *ctx->Input<Tensor>(0);
  auto& repeats_tensor = *ctx->Input<Tensor>(1);
  int32_t rank = static_cast<int32_t>(input_tensor.Shape().NumDimensions());

  if (repeats_tensor.Shape().NumDimensions() != 1)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must be 1 dimensional");
  if (repeats_tensor.Shape().Size() != rank)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must have the same length as the 'input' tensor");

  // Calculate the shape of the output tensor
  auto* repeats = repeats_tensor.template Data<int64_t>();
  const auto& input_shape = input_tensor.Shape().GetDims();
  std::vector<int64_t> output_dims(input_shape);
  for (auto axis = 0; axis < rank; axis++)
    output_dims[axis] *= repeats[axis];
  TensorShape outputShape(output_dims);
  auto& output_tensor = *ctx->Output(0, outputShape);

  void* output_data = output_tensor.MutableDataRaw();
  const void* input_data = input_tensor.DataRaw();

  TensorPitches input_pitches(input_shape);
  TArray<int64_t> input_strides(input_pitches);

  TArray<fast_divmod> fdm_input_shape(rank);
  for (int32_t i = 0; i < input_shape.size(); ++i) {
    fdm_input_shape[i] = fast_divmod(gsl::narrow_cast<int>(input_shape[i]));
  }

  TArray<fast_divmod> fdm_output_strides(rank);
  TensorPitches output_pitches(output_dims);
  for (auto i = 0; i < rank; i++) {
    fdm_output_strides[i] = fast_divmod(static_cast<int>(output_pitches[i]));
  }

  static_assert(sizeof(float) == sizeof(int32_t), "Float and Int32 are of different sizes");
  static_assert(sizeof(double) == sizeof(int64_t), "Double and Int64 are of different sizes");

  if (output_tensor.Shape().Size() > 0) {
    if (input_tensor.IsDataType<float>() ||
        input_tensor.IsDataType<int32_t>()) {
      TileImpl(
          rank,
          fdm_input_shape,
          input_strides,
          reinterpret_cast<const typename ToCudaType<float>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<typename ToCudaType<float>::MappedType*>(output_data),
          output_tensor.Shape().Size());
    } else if (input_tensor.IsDataType<double>() ||
               input_tensor.IsDataType<int64_t>()) {
      TileImpl(
          rank,
          fdm_input_shape,
          input_strides,
          reinterpret_cast<const typename ToCudaType<double>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<typename ToCudaType<double>::MappedType*>(output_data),
          output_tensor.Shape().Size());
    } else if (input_tensor.IsDataType<MLFloat16>()) {
      TileImpl(
          rank,
          fdm_input_shape,
          input_strides,
          reinterpret_cast<const typename ToCudaType<MLFloat16>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<typename ToCudaType<MLFloat16>::MappedType*>(output_data),
          output_tensor.Shape().Size());
    } else {
      // Won't hit this as the kernel doesn't claim support for any type that will trigger this
      ORT_THROW("Tile doesn't have an implementation yet for the type: ", input_tensor.DataType());
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
