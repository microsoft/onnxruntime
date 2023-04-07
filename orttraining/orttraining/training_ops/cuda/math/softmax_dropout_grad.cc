// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/softmax_dropout_grad.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "orttraining/training_ops/cuda/math/softmax_dropout_grad_impl.h"

namespace onnxruntime {
namespace cuda {

namespace {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct DispatchSoftmaxDropoutGradImpl {
  Status operator()(cudaStream_t stream, cudnnHandle_t cudnn_handle, Tensor* dX, const Tensor* dY, const Tensor* mask,
                    const Tensor* softmax_Y, int element_count, int batch_count, const float ratio) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* input_grad_data = reinterpret_cast<CudaT*>(dX->MutableData<T>());
    const CudaT* output_grad_data = reinterpret_cast<const CudaT*>(dY->Data<T>());
    const bool* mask_data = reinterpret_cast<const bool*>(mask->Data<bool>());
    const CudaT* softmax_output_data = reinterpret_cast<const CudaT*>(softmax_Y->Data<T>());
    return SoftmaxDropoutGradImpl(stream, cudnn_handle, input_grad_data, output_grad_data, mask_data,
                                  softmax_output_data, element_count, batch_count, ratio);
  }
};

}  // namespace

#ifdef USE_ROCM
#define SOFTMAX_DROPOUT_GRAD_TYPES float, MLFloat16
#else
#define SOFTMAX_DROPOUT_GRAD_TYPES float, MLFloat16, double
#endif

ONNX_OPERATOR_KERNEL_EX(SoftmaxDropoutGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<SOFTMAX_DROPOUT_GRAD_TYPES>())
                            .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
                            .InputMemoryType(OrtMemTypeCPUInput, 3),
                        SoftmaxDropoutGrad);

Status SoftmaxDropoutGrad::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* dY = ctx->Input<Tensor>(0);
  const TensorShape& input_shape = dY->Shape();
  const Tensor* mask = ctx->Input<Tensor>(1);
  const Tensor* softmax_Y = ctx->Input<Tensor>(2);

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, input_shape.NumDimensions()));
  const int batch_count = static_cast<int>(input_shape.SizeToDimension(axis));
  const int element_count = static_cast<int>(input_shape.SizeFromDimension(axis));

  // Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = ctx->Input<Tensor>(3);
  if (ratio) {
    utils::MLTypeCallDispatcher<MLFloat16, float, double> ratio_t_disp(ratio->GetElementType());
    ratio_t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  Tensor* dX = ctx->Output(0, input_shape);
  utils::MLTypeCallDispatcher<SOFTMAX_DROPOUT_GRAD_TYPES> t_disp(dY->GetElementType());
  return t_disp.InvokeRet<Status, DispatchSoftmaxDropoutGradImpl>(Stream(ctx), GetCudnnHandle(ctx), dX, dY, mask, softmax_Y,
                                                                  element_count, batch_count, ratio_data);
}

#undef SOFTMAX_DROPOUT_GRAD_TYPES

}  // namespace cuda
}  // namespace onnxruntime
