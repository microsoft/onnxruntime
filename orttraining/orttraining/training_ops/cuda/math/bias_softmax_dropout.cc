// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/bias_softmax_dropout.h"

#include "core/providers/cuda/cuda_common.h"
#include "orttraining/training_ops/cuda/math/bias_softmax_dropout_impl.h"

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
struct DispatchBiasSoftmaxDropoutImpl {
  Status operator()(cudaStream_t stream, const cudaDeviceProp& prop, cudnnHandle_t cudnn_handle, Tensor* dropout_Y,
                    Tensor* mask, Tensor* softmax_Y, const Tensor* X, const Tensor* B, int element_count,
                    int batch_count, bool is_inner_broadcast, int bias_broadcast_size, const float ratio,
                    PhiloxGenerator& generator) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* dropout_output_data = reinterpret_cast<CudaT*>(dropout_Y->MutableData<T>());
    bool* mask_data = reinterpret_cast<bool*>(mask->MutableData<bool>());
    CudaT* softmax_output_data = reinterpret_cast<CudaT*>(softmax_Y->MutableData<T>());
    const CudaT* input_data = reinterpret_cast<const CudaT*>(X->Data<T>());
    const CudaT* bias_data = reinterpret_cast<const CudaT*>(B->Data<T>());
    return BiasSoftmaxDropoutImpl<CudaT>(stream, prop, cudnn_handle, dropout_output_data, mask_data,
                                         softmax_output_data, input_data, bias_data, element_count, batch_count,
                                         is_inner_broadcast, bias_broadcast_size, ratio, generator);
  }
};

}  // namespace

#ifdef USE_ROCM
#define BIAS_SOFTMAX_DROPOUT_TYPES float, MLFloat16
#else
#define BIAS_SOFTMAX_DROPOUT_TYPES float, MLFloat16, double
#endif

ONNX_OPERATOR_KERNEL_EX(BiasSoftmaxDropout, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<BIAS_SOFTMAX_DROPOUT_TYPES>())
                            .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
                            .InputMemoryType(OrtMemTypeCPUInput, 2),
                        BiasSoftmaxDropout);

Status BiasSoftmaxDropout::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const TensorShape& X_shape = X->Shape();
  const TensorShape& B_shape = B->Shape();

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, X_shape.NumDimensions()));
  const int batch_count = static_cast<int>(X_shape.SizeToDimension(axis));
  const int element_count = static_cast<int>(X_shape.SizeFromDimension(axis));
  int bias_broadcast_size = static_cast<int>(B_shape.Size() / element_count);
  if (is_inner_broadcast_) bias_broadcast_size = batch_count / bias_broadcast_size;

  // Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = ctx->Input<Tensor>(2);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double> ratio_t_disp(ratio->GetElementType());
    ratio_t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  Tensor* dropout_Y = ctx->Output(0, X_shape);
  Tensor* mask = ctx->Output(1, X_shape);
  Tensor* softmax_Y = ctx->Output(2, X_shape);

  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();
  utils::MLTypeCallDispatcher<BIAS_SOFTMAX_DROPOUT_TYPES> t_disp(X->GetElementType());
  return t_disp.InvokeRet<Status, DispatchBiasSoftmaxDropoutImpl>(
      Stream(), GetDeviceProp(), CudnnHandle(), dropout_Y, mask, softmax_Y, X, B, element_count, batch_count,
      is_inner_broadcast_, bias_broadcast_size, ratio_data, generator);
}

#undef BIAS_SOFTMAX_DROPOUT_TYPES

}  // namespace cuda
}  // namespace onnxruntime
