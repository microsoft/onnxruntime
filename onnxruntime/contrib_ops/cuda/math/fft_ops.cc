// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "fft_ops.h"
#include "fft_ops_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Rfft,                                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Rfft<T>);                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Irfft,                                                      \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Irfft<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status FFTBase<T>::DoFFT(OpKernelContext* context, const Tensor* X, bool complex_input, bool complex_output, bool inverse) const {
  typedef typename ::onnxruntime::cuda::ToCudaType<T>::MappedType CudaT;

  ORT_ENFORCE((complex_input || complex_output) && (complex_input != complex_output),
              "Only support RFFT and IRFFT, so either input or output has to be complex type and the other is real type. Got complex input:",
              complex_input, " complex output: ", complex_output);

  TensorShape input_shape = X->Shape();
  int64_t input_ndim = input_shape.NumDimensions();
  ORT_ENFORCE(input_ndim >= signal_ndim_, "signal_ndim cannot be greater than the dimension of Input: ", signal_ndim_, " > ", input_ndim);
  auto signal_tensor_ndim = signal_ndim_ + static_cast<int64_t>(complex_input);  // add complex dim

  //calculate batch size
  int64_t batch_ndim = input_ndim - signal_tensor_ndim;
  int64_t batch_size = (batch_ndim == 0 ? 1 : input_shape.SizeToDimension(batch_ndim));

  //infer output shape
  //copy the input shape up to the second last dimention
  std::vector<int64_t> output_dims, signal_dims;
  int i = 0;
  for (; i < batch_ndim + signal_ndim_ - 1; ++i) {
    output_dims.push_back(input_shape[i]);
    if (i >= batch_ndim) {
      signal_dims.push_back(input_shape[i]);
    }
  }

  //process the last dim(s)
  if (onesided_) {
    if (complex_input && !complex_output) {  //IRFFT
      int64_t inferred_size = input_shape[i] * 2 - 1;
      output_dims.push_back(inferred_size);
      signal_dims.push_back(inferred_size);
    } else if (!complex_input && complex_output) {  // RFFT
      output_dims.push_back(input_shape[i] / 2 + 1);
      signal_dims.push_back(input_shape[i]);
    }
  } else {  // not onesided
    output_dims.push_back(input_shape[i]);
    signal_dims.push_back(input_shape[i]);
  }

  if (complex_output) {
    output_dims.push_back(2);
  }

  //Making plan
  //TODO: add plan cache
  cufftHandle plan;
  cufftResult result;
  result = cufftCreate(&plan);
  //TODO: replace it with a util func
  ORT_ENFORCE(result == CUFFT_SUCCESS, "Failed to create a cuFFT plan: ", result);

  size_t ws_size_t;

  cudaDataType itype, otype, exec_type;
  if (X->IsDataType<float>()) {
    itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
    otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
    exec_type = CUDA_C_32F;
  } else if (X->IsDataType<double>()) {
    itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
    otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
    exec_type = CUDA_C_64F;
  } else if (X->IsDataType<MLFloat16>()) {
    itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
    otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
    exec_type = CUDA_C_16F;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuFFT does not support tensor type: ", X->DataType());
  }

  result = cufftXtMakePlanMany(plan, static_cast<int>(signal_ndim_), const_cast<int64_t*>(signal_dims.data()),
                               /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
                               /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
                               batch_size, &ws_size_t, exec_type);

  //TODO: replace it with a util func
  ORT_ENFORCE(result == CUFFT_SUCCESS, "Failed to create a cuFFT plan: ", result);

  Tensor* Y = const_cast<OpKernelContext*>(context)->Output(0, TensorShape(output_dims));
  auto* x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto* y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  result = cufftXtExec(plan, const_cast<CudaT*>(x_data), y_data, inverse ? CUFFT_INVERSE : CUFFT_FORWARD);

  ORT_ENFORCE(result == CUFFT_SUCCESS, "Failed to exec the cuFFT plan: ", result);

  cufftDestroy(plan);

  if (inverse) {
    PostProcess(signal_dims, Y, y_data);
  }

  return Status::OK();
}

template <typename T>
Status Rfft<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);

  return FFTBase::DoFFT(context, X, false, true, false);
}

template <typename T>
Status Irfft<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);

  return FFTBase::DoFFT(context, X, true, false, true);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
