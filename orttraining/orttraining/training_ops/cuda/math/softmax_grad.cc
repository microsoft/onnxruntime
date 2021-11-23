// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/softmax_grad.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/providers/cuda/tensor/transpose.h"

namespace onnxruntime {
namespace cuda {

template <typename T, bool is_log_softmax>
Status SoftMaxGradComputeHelper(
    cudaStream_t stream,
    const T* dY,
    const TensorShape& input_shape,
    const T* Y,
    T* dX,
    cudnnHandle_t handle,
    int64_t axis) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const int64_t normalized_axis = HandleNegativeAxis(axis, input_shape.NumDimensions());

  int64_t N = input_shape.SizeToDimension(normalized_axis);
  int64_t D = input_shape.SizeFromDimension(normalized_axis);
  std::vector<int64_t> dims({N, 1, 1, D});  // cudnn expects 4D shape in NCHW format

  auto dY_data = reinterpret_cast<const CudaT*>(dY);
  auto Y_data = reinterpret_cast<const CudaT*>(Y);
  auto dX_data = reinterpret_cast<CudaT*>(dX);

  if (D <= 1024 && D * sizeof(T) <= 4096) {
    dispatch_softmax_backward<CudaT, CudaT, AccumulationType_t<CudaT>, is_log_softmax>(
        stream, dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
    return Status::OK();
  }

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>()));
  CUDNN_RETURN_IF_ERROR(
      cudnnSoftmaxBackward(
          handle,
          is_log_softmax ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_INSTANCE,
          &alpha,
          input_tensor,
          Y_data,
          input_tensor,
          dY_data,
          &beta,
          output_tensor,
          dX_data));

  return Status::OK();
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
// cudnnSoftmaxForward/Backward doesn't support BFloat16.
#define SPECIALIZED_SOFTMAXGRAD_HELPER_IMPL_BFloat16(is_log_softmax)                                                     \
  template <>                                                                                                            \
  Status SoftMaxGradComputeHelper<BFloat16, is_log_softmax>(                                                             \
      cudaStream_t stream,                                                                                               \
      const BFloat16* dY,                                                                                                \
      const TensorShape& input_shape,                                                                                    \
      const BFloat16* Y,                                                                                                 \
      BFloat16* dX,                                                                                                      \
      cudnnHandle_t,                                                                                                     \
      int64_t axis) {                                                                                                    \
    typedef typename ToCudaType<BFloat16>::MappedType CudaT;                                                             \
    const int64_t normalized_axis = HandleNegativeAxis(axis, input_shape.NumDimensions());                               \
    int64_t N = input_shape.SizeToDimension(normalized_axis);                                                            \
    int64_t D = input_shape.SizeFromDimension(normalized_axis);                                                          \
    auto dY_data = reinterpret_cast<const CudaT*>(dY);                                                                   \
    auto Y_data = reinterpret_cast<const CudaT*>(Y);                                                                     \
    auto dX_data = reinterpret_cast<CudaT*>(dX);                                                                         \
    dispatch_softmax_backward<CudaT, CudaT, AccumulationType_t<CudaT>, is_log_softmax>(                                  \
        stream, dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N)); \
    return Status::OK();                                                                                                 \
  }

SPECIALIZED_SOFTMAXGRAD_HELPER_IMPL_BFloat16(true)
    SPECIALIZED_SOFTMAXGRAD_HELPER_IMPL_BFloat16(false)
#endif

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      SoftmaxGrad,                                                                         \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                                     \
                                                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      SoftmaxGrad_13,                                                                      \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                                     \
                                                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LogSoftmaxGrad,                                                                      \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                                     \
                                                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LogSoftmaxGrad_13,                                                                   \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);

        template <typename T>
        Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* dY = ctx->Input<Tensor>(0);
  const TensorShape& input_shape{dY->Shape()};
  const Tensor* Y = ctx->Input<Tensor>(1);
  Tensor* dX = ctx->Output(0, input_shape);
  size_t rank = input_shape.NumDimensions();
  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));
  bool is_transpose_required = opset_ >= 13 && axis != (rank - 1);

  std::unique_ptr<Tensor> transposed_dY;
  std::unique_ptr<Tensor> transposed_Y;
  std::vector<int64_t> transposed_input_dims;
  std::unique_ptr<Tensor> intermediate_output;  // output that the softmax implementation will write into while using transposed input
  std::vector<size_t> permutation(rank);

  if (is_transpose_required) {
    AllocatorPtr alloc;
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK())
      return status;

    std::iota(std::begin(permutation), std::end(permutation), 0);

    // swap the innermost dim with the dim corresponding to axis
    permutation[axis] = rank - 1;
    permutation[rank - 1] = axis;

    transposed_input_dims.reserve(rank);
    for (auto e : permutation) {
      transposed_input_dims.push_back(input_shape[e]);
    }

    // Allocate a temporary tensor to hold transposed input
    auto temp_input0 = Tensor::Create(Y->DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(prop_,
                                               Stream(),
                                               CublasHandle(),
                                               permutation, *Y, *temp_input0));
    transposed_Y = std::move(temp_input0);
    auto temp_input1 = Tensor::Create(Y->DataType(), TensorShape(transposed_input_dims), alloc);
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(prop_,
                                               Stream(),
                                               CublasHandle(),
                                               permutation, *dY, *temp_input1));
    transposed_dY = std::move(temp_input1);

    // Allocate memory for the intermediate output
    intermediate_output = Tensor::Create(dX->DataType(), TensorShape(transposed_input_dims), alloc);
  }
  const T* dY_data = is_transpose_required ? transposed_dY->template Data<T>() : dY->template Data<T>();
  const T* Y_data = is_transpose_required ? transposed_Y->template Data<T>() : Y->template Data<T>();
  T* dX_data = is_transpose_required ? intermediate_output->template MutableData<T>() : dX->template MutableData<T>();
  const TensorShape* compute_input_shape = is_transpose_required ? &transposed_Y->Shape() : &input_shape;
  Status status;
  if (log_softmax_) {
    status = SoftMaxGradComputeHelper<T, true>(Stream(), dY_data, *compute_input_shape, Y_data, dX_data, CudnnHandle(), is_transpose_required ? static_cast<int64_t>(rank) - 1 : axis);
  } else {
    status = SoftMaxGradComputeHelper<T, false>(Stream(), dY_data, *compute_input_shape, Y_data, dX_data, CudnnHandle(), is_transpose_required ? static_cast<int64_t>(rank) - 1 : axis);
  }

  if (!status.IsOK()) {
    return status;
  }

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(prop_,
                                               Stream(),
                                               CublasHandle(),
                                               permutation, *intermediate_output, *dX));
  }
  return Status::OK();
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
SPECIALIZED_GRADIENT(double)
SPECIALIZED_GRADIENT(MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
SPECIALIZED_GRADIENT(BFloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime
