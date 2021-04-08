// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_grad.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
// #include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/torch_wrapper/torch_wrapper.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace cuda {

#ifdef USE_TORCH
#define HANDLE_EXTERNAL_OUTPUTS_ATTR .ExternalOutputs()
#else
#define HANDLE_EXTERNAL_OUTPUTS_ATTR
#endif

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      ConvGrad,                                                                 \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
      HANDLE_EXTERNAL_OUTPUTS_ATTR,                                             \
      ConvGrad<T>);

REGISTER_GRADIENT_KERNEL_TYPED(float)
REGISTER_GRADIENT_KERNEL_TYPED(double)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionBackwardDataWorkspaceSize(
      args.handle,
      args.w_desc,
      args.o_desc,
      args.c_desc,
      args.i_desc,
      algo,
      sz);
}

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(
      args.handle,
      args.i_desc,
      args.o_desc,
      args.c_desc,
      args.w_desc,
      algo,
      sz);
}

template <typename T>
Status ConvGrad<T>::PrepareArgs(const Tensor& input, const Tensor& output, const Tensor& weight, const Tensor* bias) const {
  const TensorShape& i_shape = input.Shape();
  std::vector<int64_t> i_dims = i_shape.GetDims();

  const TensorShape& o_shape = output.Shape();
  std::vector<int64_t> o_dims = o_shape.GetDims();

  const TensorShape& w_shape = weight.Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();

  // Update Attributes
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&input, &weight));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(w_shape, kernel_shape));
  auto rank = kernel_shape.size();

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(rank * 2, 0);
  }

  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(rank, 1);
  }

  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(rank, 1);
  }

  // cudnn only takes 4D or 5D input, so pad dimensions if needed
  if (rank < 2) {
    i_dims.push_back(1);
    o_dims.push_back(1);
    w_dims.push_back(1);

    pads.insert(pads.begin() + rank, 0);
    pads.insert(pads.end(), 0);
    kernel_shape.push_back(1);
    strides.push_back(1);
    dilations.push_back(1);
  }

  args_.handle = CudnnHandle();
  ORT_RETURN_IF_ERROR(args_.i_desc.Set(i_dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(args_.o_desc.Set(o_dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(args_.w_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(args_.c_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                       gsl::narrow_cast<int>(conv_attrs_.group),
                                       CUDNN_CROSS_CORRELATION, CudnnTensor::GetDataType<CudaT>()));

  if (bias) {
    const TensorShape& b_shape = bias->Shape();
    ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
    std::vector<int64_t> b_dims(2 + kernel_shape.size(), 1);
    b_dims[1] = b_shape[0];
    ORT_RETURN_IF_ERROR(args_.b_desc.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
  }

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInternal(OpKernelContext* context) const {
#ifdef USE_TORCH
  auto* ctx_internal = static_cast<OpKernelContextInternal*>(context);
  OrtValue grad_output = *ctx_internal->GetInputMLValue(0);
  OrtValue input = *ctx_internal->GetInputMLValue(1);
  OrtValue weight = *ctx_internal->GetInputMLValue(2);
  bool should_compute_input_grad = (ctx_internal->GetOutputMLValue(0) != nullptr);
  bool should_compute_weight_grad = (ctx_internal->GetOutputMLValue(1) != nullptr);
  bool should_compute_bias_grad = (ctx_internal->GetOutputMLValue(2) != nullptr);

  // ONNX supports asymmetric padding, whereas PyTorch supports only symmetric padding.
  // i.e., onnx_pads = torch_padding + torch_padding
  std::vector<int64_t> torch_padding(conv_attrs_.pads.begin(), conv_attrs_.pads.begin() + conv_attrs_.pads.size() / 2);
  std::vector<OrtValue> result = torch_wrapper::convolution_backward(
      grad_output, input, weight, conv_attrs_.strides, torch_padding, conv_attrs_.dilations, conv_attrs_.group,
      should_compute_input_grad, should_compute_weight_grad, should_compute_bias_grad);
  if (should_compute_input_grad) {
    ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(0, result[0]));
  }

  if (should_compute_weight_grad) {
    ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(1, result[1]));
  }

  if (should_compute_bias_grad) {
    ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(2, result[2]));
  }

  return Status::OK();
#endif

  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);
  const int64_t M = W->Shape()[0];

  Tensor* dX = context->Output(0, X->Shape());
  Tensor* dW = context->Output(1, W->Shape());
  Tensor* dB = context->Output(2, {M});

  ORT_RETURN_IF_ERROR(PrepareArgs(*dX, *dY, *dW, dB));

  ORT_RETURN_IF_ERROR(ComputeWeightGradient(dW, dY, X));
  ORT_RETURN_IF_ERROR(ComputeInputGradient(dX, dY, W));
  ORT_RETURN_IF_ERROR(ComputeBiasGradient(dB, dY));

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeWeightGradient(Tensor* dW, const Tensor* dY, const Tensor* X) const {
  if (dW == nullptr) return Status::OK();

  cudnnConvolutionBwdFilterAlgoPerf_t perf;
  perf.algo = kDefaultConvBwdFilterAlgo;
  CUDNN_RETURN_IF_ERROR(getWorkspaceSize(args_, perf.algo, &perf.memory));
  // CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.c_desc, perf.mathType));

  void* dw_data = dW->template MutableData<T>();
  const void* dy_data = dY->template Data<T>();
  const void* x_data = X->template Data<T>();
  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(perf.memory);

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardFilter(
          args_.handle,
          &one, args_.i_desc, x_data,
          args_.o_desc, dy_data,
          args_.c_desc, perf.algo, workspace.get(), perf.memory,
          &zero, args_.w_desc, dw_data));

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInputGradient(Tensor* dX, const Tensor* dY, const Tensor* W) const {
  if (dX == nullptr) return Status::OK();

  cudnnConvolutionBwdDataAlgoPerf_t perf;
  perf.algo = kDefaultConvBwdDataAlgo;
  CUDNN_RETURN_IF_ERROR(getWorkspaceSize(args_, perf.algo, &perf.memory));
  // CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.c_desc, perf.mathType));

  void* dx_data = dX->template MutableData<T>();
  const void* dy_data = dY->template Data<T>();
  const void* w_data = W->template Data<T>();
  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(perf.memory);

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardData(
          args_.handle,
          &one, args_.w_desc, w_data,
          args_.o_desc, dy_data,
          args_.c_desc, perf.algo, workspace.get(), perf.memory,
          &zero, args_.i_desc, dx_data));

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeBiasGradient(Tensor* dB, const Tensor* dY) const {
  if (dB == nullptr) return Status::OK();

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  void* db_data = dB->template MutableData<T>();
  const void* dy_data = dY->template Data<T>();

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardBias(
          args_.handle,
          &one, args_.o_desc, dy_data,
          &zero, args_.b_desc, db_data));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
