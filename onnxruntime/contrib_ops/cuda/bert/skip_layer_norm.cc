// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"
#include "skip_layer_norm.h"
#include "skip_layer_norm_impl.h"
#include "contrib_ops/cpu/skip_layer_norm_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SkipLayerNormalization,                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SkipLayerNorm<T, false>);                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SkipSimplifiedLayerNormalization,                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SkipLayerNorm<T, true>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T, bool Simplified>
SkipLayerNorm<T, Simplified>::SkipLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);

  const CUDAExecutionProvider* cuda_ep = static_cast<const CUDAExecutionProvider*>(op_kernel_info.GetExecutionProvider());

  strict_ = cuda_ep->IsSkipLayerNormInStrictMode();
}

template <typename T, bool Simplified>
Status SkipLayerNorm<T, Simplified>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* skip = ctx->Input<Tensor>(1);
  const Tensor* gamma = ctx->Input<Tensor>(2);

  const Tensor* beta = Simplified ? nullptr : ctx->Input<Tensor>(3);
  const Tensor* bias = Simplified ? ctx->Input<Tensor>(3) : ctx->Input<Tensor>(4);

  Tensor* output = ctx->Output(0, input->Shape());

  // For inferencing, we support one more optional output which is the sum
  // of the input and skip tensors
  Tensor* skip_input_bias_add_output = ctx->Output(3, input->Shape());

  const auto& input_dims = input->Shape().GetDims();
  size_t input_dims_size = input_dims.size();
  const auto& skip_dims = skip->Shape().GetDims();
  size_t skip_dims_size = skip_dims.size();

  int hidden_size = static_cast<int>(input_dims[input_dims_size - 1]);

  ORT_RETURN_IF_ERROR(onnxruntime::contrib::skip_layer_norm_helper::CheckInputs<Tensor>(input,
                                                                                        skip,
                                                                                        gamma,
                                                                                        beta,
                                                                                        bias,
                                                                                        hidden_size,
                                                                                        input_dims_size));

  const bool skip_broadcasted = (skip_dims[0] == 1 || skip_dims_size == 2) ? true : false;
  const int skip_size = static_cast<int>(skip_dims[skip_dims_size - 1] * skip_dims[skip_dims_size - 2]);

  int row_count = gsl::narrow<int>(input->Shape().SizeToDimension(input_dims_size - 1));
  typedef typename ToCudaType<T>::MappedType CudaT;

  if (strict_) {
    HostApplyLayerNorm<CudaT, float, CudaT, Simplified>(
        GetDeviceProp(),
        Stream(ctx),
        reinterpret_cast<CudaT*>(output->MutableData<T>()),                             // Y_data
        nullptr,                                                                        // mean_data
        nullptr,                                                                        // inv_var_data
        reinterpret_cast<const CudaT*>(input->Data<T>()),                               // X_data
        row_count,                                                                      // n1
        hidden_size,                                                                    // n2
        (double)epsilon_,                                                               // epsilon
        reinterpret_cast<const CudaT*>(gamma->Data<T>()),                               // gamma
        (beta != nullptr) ? reinterpret_cast<const CudaT*>(beta->Data<T>()) : nullptr,  // beta
        reinterpret_cast<const CudaT*>(skip->Data<T>()),                                // skip or residual to add
        (bias != nullptr) ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,  // bias to add
        skip_input_bias_add_output != nullptr ? reinterpret_cast<CudaT*>(skip_input_bias_add_output->MutableData<T>()) : nullptr);
  } else {
    LaunchSkipLayerNormKernel<CudaT, Simplified>(
        Stream(ctx),
        reinterpret_cast<CudaT*>(output->MutableData<T>()),
        skip_input_bias_add_output != nullptr ? reinterpret_cast<CudaT*>(skip_input_bias_add_output->MutableData<T>()) : nullptr,
        reinterpret_cast<const CudaT*>(input->Data<T>()),
        reinterpret_cast<const CudaT*>(skip->Data<T>()),
        reinterpret_cast<const CudaT*>(gamma->Data<T>()),
        (beta != nullptr) ? reinterpret_cast<const CudaT*>(beta->Data<T>()) : nullptr,
        (bias != nullptr) ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,
        epsilon_,
        hidden_size,
        row_count,
        skip_broadcasted,
        skip_size);
  }

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
