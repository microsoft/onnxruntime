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
  size_t rank = input_tensor.Shape().NumDimensions();

  if (repeats_tensor.Shape().NumDimensions() != 1)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must be 1 dimensional");
  if (size_t(repeats_tensor.Shape().Size()) != rank)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "'repeat' input tensor must have the same length as the 'input' tensor");

  // Calculate the shape of the output tensor
  auto* repeats = repeats_tensor.template Data<int64_t>();
  const auto& input_shape = input_tensor.Shape().GetDims();
  std::vector<int64_t> output_dims(input_shape);
  for (size_t axis = 0; axis < rank; axis++)
    output_dims[axis] *= repeats[axis];
  TensorShape outputShape(output_dims);
  auto& output_tensor = *ctx->Output(0, outputShape);

  T* output_data = output_tensor.template MutableData<T>();
  const T* input_data = input_tensor.template Data<T>();
  CudaAsyncBuffer<int64_t> input_strides(this, rank);
  CudaAsyncBuffer<fast_divmod> fdm_input_shape(this, rank);
  CudaAsyncBuffer<fast_divmod> fdm_output_strides(this, rank);

  ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_shape));
  ORT_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));

  auto fdm_input_shape_span = fdm_input_shape.CpuSpan();
  for (size_t i = 0; i < input_shape.size(); ++i)
    fdm_input_shape_span[i] = fast_divmod(gsl::narrow_cast<int>(input_shape[i]));

  ORT_RETURN_IF_ERROR(fdm_input_shape.CopyToGpu());
  ORT_RETURN_IF_ERROR(input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());

  if (output_tensor.Shape().Size() > 0) {
    TileImpl(
        rank,
        fdm_input_shape.GpuPtr(),
        input_strides.GpuPtr(),
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input_data),
        fdm_output_strides.GpuPtr(),
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
