// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/tile.h"
#include "core/providers/cpu/tensor/utils.h"
#include "tile_impl.h"
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Tile,                                                       \
      kOnnxDomain,                                                \
      6,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()), \
      Tile<T>);

template <typename T>
Status Tile<T>::ComputeInternal(OpKernelContext* ctx) const {
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

  T* output_data = output_tensor.template MutableData<T>();
  const T* input_data = input_tensor.template Data<T>();

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

  if (output_tensor.Shape().Size() > 0) {
    TileImpl(
        rank,
        fdm_input_shape,
        input_strides,
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input_data),
        fdm_output_strides,
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_data),
        output_tensor.Shape().Size());
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Tile<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
