// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/softmax_grad.h"
#include "orttraining/training_ops/cuda/math/softmax_grad_impl.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/providers/cuda/tensor/transpose.h"

namespace onnxruntime {
namespace cuda {

namespace {

template <typename T>
struct DispatchSoftmaxGradImpl {
  Status operator()(cudaStream_t stream, cudnnHandle_t cudnn_handle, Tensor* dX, const Tensor* dY, const Tensor* Y,
                    int element_count, int batch_count, bool is_log_softmax) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* input_grad_data = reinterpret_cast<CudaT*>(dX->MutableData<T>());
    const CudaT* output_grad_data = reinterpret_cast<const CudaT*>(dY->Data<T>());
    const CudaT* softmax_output_data = reinterpret_cast<const CudaT*>(Y->Data<T>());
    return SoftmaxGradImpl<CudaT>(stream, cudnn_handle, input_grad_data, output_grad_data, softmax_output_data,
                                  element_count, batch_count, is_log_softmax);
  }
};

}  // namespace

// MIOpen doesn't support double so ROCm kernel doesn't have double support for now.
#ifdef USE_ROCM
#define SOFTMAX_GRAD_TYPES float, MLFloat16, BFloat16
#else
#define SOFTMAX_GRAD_TYPES float, double, MLFloat16, BFloat16
#endif

#define REGISTER_SOFTMAX_GRAD_KERNEL(name)                                                                \
  ONNX_OPERATOR_KERNEL_EX(                                                                                \
      name, kMSDomain, 1, kCudaExecutionProvider,                                                         \
      (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<SOFTMAX_GRAD_TYPES>()), \
      SoftmaxGrad);

REGISTER_SOFTMAX_GRAD_KERNEL(SoftmaxGrad)
REGISTER_SOFTMAX_GRAD_KERNEL(SoftmaxGrad_13)
REGISTER_SOFTMAX_GRAD_KERNEL(LogSoftmaxGrad)
REGISTER_SOFTMAX_GRAD_KERNEL(LogSoftmaxGrad_13)

#undef REGISTER_SOFTMAX_GRAD_KERNEL

Status SoftmaxGrad::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* dY = ctx->Input<Tensor>(0);
  const TensorShape& input_shape{dY->Shape()};
  const Tensor* Y = ctx->Input<Tensor>(1);
  Tensor* dX = ctx->Output(0, input_shape);
  size_t rank = input_shape.NumDimensions();
  size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));
  bool is_transpose_required = is_since_opset_13_ && axis != (rank - 1);

  std::unique_ptr<Tensor> transposed_dY;
  std::unique_ptr<Tensor> transposed_Y;
  std::unique_ptr<Tensor> transposed_dX;
  InlinedVector<size_t> permutation(rank);

  int batch_count = static_cast<int>(input_shape.SizeToDimension(axis));
  int element_count = static_cast<int>(input_shape.SizeFromDimension(axis));

  if (is_transpose_required) {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

    // swap the innermost dim with the dim corresponding to axis
    std::iota(std::begin(permutation), std::end(permutation), 0);
    permutation[axis] = rank - 1;
    permutation[rank - 1] = axis;

    TensorShapeVector transposed_input_dims;
    for (auto e : permutation) {
      transposed_input_dims.emplace_back(input_shape[e]);
    }
    TensorShape transposed_input_shape(transposed_input_dims);

    // Allocate a temporary tensor to hold transposed input
    auto temp_input0 = Tensor::Create(Y->DataType(), transposed_input_shape, alloc);
    ORT_RETURN_IF_ERROR(
        Transpose::DoTranspose(GetDeviceProp(), Stream(ctx), GetCublasHandle(ctx), permutation, *Y, *temp_input0));
    transposed_Y = std::move(temp_input0);

    auto temp_input1 = Tensor::Create(dY->DataType(), transposed_input_shape, alloc);
    ORT_RETURN_IF_ERROR(
        Transpose::DoTranspose(GetDeviceProp(), Stream(ctx), GetCublasHandle(ctx), permutation, *dY, *temp_input1));
    transposed_dY = std::move(temp_input1);

    // Allocate memory for the intermediate output
    transposed_dX = Tensor::Create(dX->DataType(), transposed_input_shape, alloc);

    axis = rank - 1;
    element_count = static_cast<int>(transposed_input_dims[rank - 1]);
    batch_count = static_cast<int>(input_shape.Size()) / element_count;
  }

  utils::MLTypeCallDispatcher<SOFTMAX_GRAD_TYPES> t_disp(dY->GetElementType());
  Status status = t_disp.InvokeRet<Status, DispatchSoftmaxGradImpl>(
      Stream(ctx), GetCudnnHandle(ctx), is_transpose_required ? transposed_dX.get() : dX,
      is_transpose_required ? transposed_dY.get() : dY, is_transpose_required ? transposed_Y.get() : Y, element_count,
      batch_count, is_log_softmax_);
  ORT_RETURN_IF_ERROR(status);

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(
        Transpose::DoTranspose(GetDeviceProp(), Stream(ctx), GetCublasHandle(ctx), permutation, *transposed_dX, *dX));
  }

  return Status::OK();
}

#undef SOFTMAX_GRAD_TYPES

}  // namespace cuda
}  // namespace onnxruntime
