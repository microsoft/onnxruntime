// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/math/softmax.h"

#include "core/providers/common.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/shared_inc/accumulation_type.h"
#include "core/providers/rocm/tensor/transpose.h"

namespace onnxruntime {
namespace rocm {

template <typename T, typename TOut, bool IsLogSoftmax>
Status SoftMaxComputeHelper(
    hipStream_t stream,
    const T* X,
    const TensorShape& input_shape,
    TOut* Y,
    int64_t axis,
    RocmTuningContext* tuning_ctx) {
  typedef typename ToHipType<T>::MappedType HipT_IN;
  typedef typename ToHipType<TOut>::MappedType HipT_OUT;
  typedef typename ToHipType<T>::MappedType HipT_ACCUM;

  int64_t N = input_shape.SizeToDimension(axis);
  int64_t D = input_shape.SizeFromDimension(axis);
  auto Y_data = reinterpret_cast<HipT_OUT*>(Y);
  auto X_data = reinterpret_cast<const HipT_IN*>(X);

  if (D <= 1024 && D * sizeof(T) <= 4096) {
    return dispatch_warpwise_softmax_forward<HipT_IN, HipT_OUT, AccumulationType_t<HipT_ACCUM>, IsLogSoftmax>(
        stream, Y_data, X_data, gsl::narrow_cast<int>(D),
        gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N), tuning_ctx);
  }
  return dispatch_blockwise_softmax_forward<HipT_IN, HipT_OUT, AccumulationType_t<HipT_ACCUM>, IsLogSoftmax>(
      stream, Y_data, X_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D),
      gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N), tuning_ctx);
}

#define SPECIALIZED_SOFTMAX_HELPER_IMPL(T, TOut)                                                                              \
  template Status SoftMaxComputeHelper<T, TOut, false>(hipStream_t stream, const T* input, const TensorShape& shape, TOut* Y, \
                                                       int64_t axis, RocmTuningContext* tuning_ctx);                          \
  template Status SoftMaxComputeHelper<T, TOut, true>(hipStream_t stream, const T* input, const TensorShape& shape, TOut* Y,  \
                                                      int64_t axis, RocmTuningContext* tuning_ctx);

SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16, float)
SPECIALIZED_SOFTMAX_HELPER_IMPL(float, float)
// MIOpen double data type not supported
// SPECIALIZED_SOFTMAX_HELPER_IMPL(double, double)
SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16, MLFloat16)
SPECIALIZED_SOFTMAX_HELPER_IMPL(BFloat16, BFloat16)

#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Softmax,                                                                             \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Softmax,                                                                             \
      kOnnxDomain,                                                                         \
      11, 12,                                                                              \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Softmax,                                                                             \
      kOnnxDomain,                                                                         \
      13,                                                                                  \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      LogSoftmax,                                                                          \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      LogSoftmax,                                                                          \
      kOnnxDomain,                                                                         \
      11, 12,                                                                              \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LogSoftmax,                                                                          \
      kOnnxDomain,                                                                         \
      13,                                                                                  \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

template <typename T>
Status Softmax<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& input_shape{X->Shape()};
  size_t rank = input_shape.NumDimensions();
  Tensor* Y = ctx->Output(0, input_shape);

  // special case when there is a dim value of 0 in the shape.
  if (input_shape.Size() == 0)
    return Status::OK();

  // handle negative and enforce axis is valid
  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));

  bool is_transpose_required = false;
  std::unique_ptr<Tensor> transposed_input;
  std::vector<int64_t> transposed_input_dims;
  std::unique_ptr<Tensor> intermediate_output;  // output that the softmax implementation will write into while using transposed input
  std::vector<size_t> permutation(rank);

  // The "semantic" meaning of axis has changed in opset-13.
  // Please compare: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
  // with https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softmax-11 for detailed explanations
  // To account for the opset-13 behavior, our plan will be to transpose the "axis" dim to the innermost dim
  // and perform softmax and then reverse the transpose. We can skip the transposing aspect if the axis is already
  // the innermost dim
  if (opset_ >= 13 && axis != (rank - 1)) {
    is_transpose_required = true;
  }

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
    auto temp_input = Tensor::Create(X->DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(rocm_ep_->GetDeviceProp(),
                                               Stream(ctx),
                                               GetRocblasHandle(ctx),
                                               permutation, *X, *temp_input));
    transposed_input = std::move(temp_input);

    // Allocate memory for the intermediate output
    intermediate_output = Tensor::Create(Y->DataType(), TensorShape(transposed_input_dims), alloc);
  }

  const T* X_data = nullptr;
  T* Y_data = nullptr;
  const TensorShape* compute_input_shape = nullptr;

  if (is_transpose_required) {  // use intermediate buffers to compute the softmax values
    X_data = transposed_input->Data<T>();
    Y_data = intermediate_output->MutableData<T>();
    compute_input_shape = &transposed_input->Shape();
  } else {  // use the node input/output directly
    X_data = X->Data<T>();
    Y_data = Y->MutableData<T>();
    compute_input_shape = &input_shape;
  }

  Status status;
  if (log_softmax_) {
    status = SoftMaxComputeHelper<T, T, true>(Stream(ctx), X_data, *compute_input_shape, Y_data,
                                              is_transpose_required ? static_cast<int64_t>(rank) - 1
                                                                    : static_cast<int64_t>(axis),
                                              GetTuningContext());
  } else {
    status = SoftMaxComputeHelper<T, T, false>(Stream(ctx), X_data, *compute_input_shape, Y_data,
                                               is_transpose_required ? static_cast<int64_t>(rank) - 1
                                                                     : static_cast<int64_t>(axis),
                                               GetTuningContext());
  }

  if (!status.IsOK())
    return status;

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(rocm_ep_->GetDeviceProp(),
                                               Stream(ctx),
                                               GetRocblasHandle(ctx),
                                               permutation, *intermediate_output, *Y));
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Softmax<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
// MIOpen double data type not supported
// SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)
SPECIALIZED_COMPUTE(BFloat16)

}  // namespace rocm
}  // namespace onnxruntime
