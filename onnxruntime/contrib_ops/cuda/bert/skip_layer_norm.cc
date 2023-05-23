// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"
#include "skip_layer_norm.h"
#include "skip_layer_norm_impl.h"

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

  if (input->Shape() != skip->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "skip is expected to have same shape as input");
  }

  if (input->Shape().Size() == 0) {
    return Status::OK();
  }

  const auto& input_dims = input->Shape().GetDims();
  size_t input_dims_size = input_dims.size();
  if (input_dims_size != 3 && input_dims_size != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 or 2 dimensions, got ", input_dims_size);
  }

  int hidden_size = static_cast<int>(input_dims[input_dims_size - 1]);

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Last dimension of gamma and input does not match");
  }

  if (!Simplified) {
    if (nullptr != beta) {
      const auto& beta_dims = beta->Shape().GetDims();
      if (beta_dims.size() != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "beta is expected to have 1 dimension, got ", beta_dims.size());
      }
      if (beta_dims[0] != hidden_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Last dimension of beta and input does not match");
      }
    }
  }

  if (nullptr != bias) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "bias is expected to have 1 dimension, got ", bias_dims.size());
    }
    if (bias_dims[0] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Last dimension of bias and input does not match");
    }
  }

  if (strict_) {
    int row_count = gsl::narrow<int>(input->Shape().SizeToDimension(input_dims_size - 1));
    typedef typename ToCudaType<T>::MappedType CudaT;
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

    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return Status::OK();
  }

  int sequence_length = static_cast<int>(input_dims[1]);
  int64_t element_count = input_dims[0] * sequence_length * hidden_size;
  size_t element_size = sizeof(T);
  typedef typename ToCudaType<T>::MappedType CudaT;
  return LaunchSkipLayerNormKernel<CudaT, Simplified>(
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
      static_cast<int>(element_count),
      element_size);

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
