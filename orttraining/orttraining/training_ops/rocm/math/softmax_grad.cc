// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/math/softmax_grad.h"

#include "core/providers/common.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/math/softmax.h"
#include "core/providers/rocm/shared_inc/accumulation_type.h"
#include "core/providers/rocm/tensor/transpose.h"

namespace onnxruntime {
namespace rocm {

template <typename T, bool is_log_softmax>
Status SoftMaxGradComputeHelper(
    hipStream_t stream,
    const T* dY,
    const TensorShape& input_shape,
    const T* Y,
    T* dX,
    miopenHandle_t handle,
    int64_t axis) {
  typedef typename ToHipType<T>::MappedType HipT;

  const int64_t normalized_axis = HandleNegativeAxis(axis, input_shape.NumDimensions());

  int64_t N = input_shape.SizeToDimension(normalized_axis);
  int64_t D = input_shape.SizeFromDimension(normalized_axis);
  std::vector<int64_t> dims({N, 1, 1, D});  // miopen expects 4D shape in NCHW format

  auto dY_data = reinterpret_cast<const HipT*>(dY);
  auto Y_data = reinterpret_cast<const HipT*>(Y);
  auto dX_data = reinterpret_cast<HipT*>(dX);

  if (D <= 1024 && D * sizeof(T) <= 4096) {
    dispatch_softmax_backward<HipT, HipT, AccumulationType_t<HipT>, is_log_softmax>(
        stream, dX_data, dY_data, Y_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
    return Status::OK();
  }

  const auto alpha = Consts<HipT>::One;
  const auto beta = Consts<HipT>::Zero;
  MiopenTensor input_tensor;
  MiopenTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(dims, MiopenTensor::GetDataType<HipT>()));
  ORT_RETURN_IF_ERROR(output_tensor.Set(dims, MiopenTensor::GetDataType<HipT>()));
  MIOPEN_RETURN_IF_ERROR(
      miopenSoftmaxBackward_V2(
          handle,
          &alpha,
          input_tensor,
          Y_data,
          input_tensor,
          dY_data,
          &beta,
          output_tensor,
          dX_data,
          is_log_softmax? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE,
          MIOPEN_SOFTMAX_MODE_INSTANCE));

  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      SoftmaxGrad,                                                                         \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      SoftmaxGrad_13,                                                                      \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LogSoftmaxGrad,                                                                      \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SoftmaxGrad<T>);                                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LogSoftmaxGrad_13,                                                                   \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
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
                                               RocblasHandle(),
                                               permutation, *Y, *temp_input0));
    transposed_Y = std::move(temp_input0);
    auto temp_input1 = Tensor::Create(Y->DataType(), TensorShape(transposed_input_dims), alloc);
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(prop_,
                                               Stream(),
                                               RocblasHandle(),
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
    status = SoftMaxGradComputeHelper<T, true>(Stream(), dY_data, *compute_input_shape, Y_data, dX_data, MiopenHandle(), is_transpose_required ? static_cast<int64_t>(rank) - 1 : axis);
  } else {
    status = SoftMaxGradComputeHelper<T, false>(Stream(), dY_data, *compute_input_shape, Y_data, dX_data, MiopenHandle(), is_transpose_required ? static_cast<int64_t>(rank) - 1 : axis);
  }

  if (!status.IsOK()) {
    return status;
  }

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(prop_,
                                               Stream(),
                                               RocblasHandle(),
                                               permutation, *intermediate_output, *dX));
  }
  return Status::OK();
}

#define SPECIALIZED_GRADIENT(T)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T) \
  template Status SoftmaxGrad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float)
// SPECIALIZED_GRADIENT(double)
SPECIALIZED_GRADIENT(MLFloat16)

}  // namespace rocm
}  // namespace onnxruntime
