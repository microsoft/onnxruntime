// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_transpose_grad.h"

namespace onnxruntime::cuda {

#define REGISTER_CONVTRANSPOSE_GRADIENT_KERNEL_TYPED(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(ConvTransposeGrad, kMSDomain, 1, T, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                               \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                ConvTransposeGrad<T>);

REGISTER_CONVTRANSPOSE_GRADIENT_KERNEL_TYPED(float)
REGISTER_CONVTRANSPOSE_GRADIENT_KERNEL_TYPED(double)
REGISTER_CONVTRANSPOSE_GRADIENT_KERNEL_TYPED(MLFloat16)

template <typename T>
Status ConvTransposeGrad<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);
  Tensor* dX = context->Output(0, X->Shape());
  Tensor* dW = context->Output(1, W->Shape());
  Tensor* dB = context->Output(2, {W->Shape()[1] * conv_attrs_.group});

  if (dX) {
    ORT_RETURN_IF_ERROR(PrepareConvForwardArgs(*dY, *W, *dX, GetCudnnHandle(context), args_dx_));
    ORT_RETURN_IF_ERROR(ComputeInputGradient(context->GetComputeStream(), args_dx_));
  }

  if (dW || dB) {
    ORT_RETURN_IF_ERROR(PrepareConvBackwardFilterArgs(*dY, *W, *X, dW, dB, GetCudnnHandle(context), args_dw_));
    if (dW) ORT_RETURN_IF_ERROR(ComputeWeightGradient(context->GetComputeStream(), args_dw_));
    if (dB) ORT_RETURN_IF_ERROR(ComputeBiasGradient(args_dw_));
  }

  return Status::OK();
}

template <typename T>
Status ConvTransposeGrad<T>::ComputeInputGradient(onnxruntime::Stream* stream, const ConvArgs& args) const {
  return AlgoIterator<T_FwdPerf>(args).TryAll(
      static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider()),
      Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
      [&](const T_FwdPerf& algo_perf) -> Status {
        const auto one = Consts<CudaT>::One;
        const auto zero = Consts<CudaT>::Zero;
        IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(algo_perf.memory, stream);
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args.conv_desc, algo_perf.mathType));
        CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(
            args.handle, &one, args.x_tensor, args.x_data, args.w_desc, args.w_data, args.conv_desc,
            algo_perf.algo, workspace.get(), algo_perf.memory, &zero, args.y_tensor, args.y_data));
        return Status::OK();
      });
}

template <typename T>
Status ConvTransposeGrad<T>::ComputeWeightGradient(onnxruntime::Stream* stream, const ConvArgs& args) const {
  return AlgoIterator<T_BwdFilterPerf>(args).TryAll(
      static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider()),
      Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
      [&](const T_BwdFilterPerf& algo_perf) -> Status {
        const auto one = Consts<CudaT>::One;
        const auto zero = Consts<CudaT>::Zero;
        IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(algo_perf.memory, stream);
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args.conv_desc, algo_perf.mathType));
        CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardFilter(
            args.handle, &one, args.x_tensor, args.x_data, args.y_tensor, args.dy_data, args.conv_desc,
            algo_perf.algo, workspace.get(), algo_perf.memory, &zero, args.w_desc, args.dw_data));
        return Status::OK();
      });
}

template <typename T>
Status ConvTransposeGrad<T>::ComputeBiasGradient(const ConvArgs& args) const {
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardBias(args.handle, &one, args.x_tensor, args.x_data, &zero,
                                                     args.b_tensor, args.db_data));
  return Status::OK();
}

template <typename T>
Status ConvTransposeGrad<T>::PrepareConvForwardArgs(const Tensor& X, const Tensor& W,
                                                    Tensor& Y, cudnnHandle_t cudnn_handle,
                                                    ConvArgs& args) const {
  const TensorShape& x_shape = X.Shape();
  auto x_dims = x_shape.AsShapeVector();
  args.x_data = reinterpret_cast<const CudaT*>(X.template Data<T>());

  const TensorShape& w_shape = W.Shape();
  auto w_dims = w_shape.AsShapeVector();
  args.w_data = reinterpret_cast<const CudaT*>(W.template Data<T>());

  const TensorShape& y_shape = Y.Shape();
  auto y_dims = y_shape.AsShapeVector();
  args.y_data = reinterpret_cast<CudaT*>(Y.template MutableData<T>());

  args.dy_data = nullptr;
  args.db_data = nullptr;
  args.dx_data = nullptr;
  args.dw_data = nullptr;

  bool x_dims_changed = (args.last_x_dims != x_dims);
  bool w_dims_changed = (args.last_w_dims != w_dims);
  if (x_dims_changed || w_dims_changed) {
    if (x_dims_changed) args.last_x_dims = x_dims;
    if (w_dims_changed) args.last_w_dims = w_dims;

    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&X, &W));

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(w_shape, kernel_shape));
    auto rank = kernel_shape.size();

    ConvPadVector pads(conv_attrs_.pads);
    if (pads.empty()) {
      pads.resize(rank * 2, 0);
    }

    TensorShapeVector dilations(conv_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(rank, 1);
    }

    TensorShapeVector strides(conv_attrs_.strides);
    if (strides.empty()) {
      strides.resize(rank, 1);
    }

    const CUDAExecutionProvider* cuda_ep =
        static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());

    if (rank < 2) {
      if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
        x_dims.insert(x_dims.begin() + 2, 1);
        y_dims.insert(y_dims.begin() + 2, 1);
        w_dims.insert(w_dims.begin() + 2, 1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.begin(), 0);
        kernel_shape.insert(kernel_shape.begin(), 1);
        strides.insert(strides.begin(), 1);
        dilations.insert(dilations.begin(), 1);
      } else {
        x_dims.push_back(1);
        y_dims.push_back(1);
        w_dims.push_back(1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }
    }

    memset(&args.params, 0, sizeof(ConvParams));
    args.params.device_id = static_cast<int8_t>(cuda_ep->GetDeviceId());
    args.params.data_type = CudnnTensor::GetDataType<CudaT>();
    args.params.input_dim = static_cast<uint8_t>(x_dims.size());
    for (size_t i = 0; i < x_dims.size(); i++) {
      args.params.input_size[i] = static_cast<int>(x_dims[i]);
      args.params.weight_size[i] = static_cast<int>(w_dims[i]);
    }
    for (size_t i = 0; i < rank; i++) {
      args.params.padding[i] = static_cast<int>(pads[i]);
      args.params.padding[i + rank] = static_cast<int>(pads[i + rank]);
      args.params.stride[i] = static_cast<int>(strides[i]);
      args.params.dilation[i] = static_cast<int>(dilations[i]);
    }
    args.params.groups = conv_attrs_.group;
    int algo_mode = cuda_ep->GetCudnnConvAlgo();
    ORT_ENFORCE(algo_mode > -1 && algo_mode < 3,
                "Algo mode should be EXHAUSTIVE (0), HEURISTIC (1) or DEFAULT (2), but got ", algo_mode);
    args.params.algo_mode = algo_mode;

    args.handle = cudnn_handle;
    ORT_RETURN_IF_ERROR(args.w_desc.Set(w_dims, args.params.data_type));
    ORT_RETURN_IF_ERROR(args.x_tensor.Set(x_dims, args.params.data_type));
    ORT_RETURN_IF_ERROR(args.y_tensor.Set(y_dims, args.params.data_type));
    ORT_RETURN_IF_ERROR(args.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                           gsl::narrow_cast<int>(conv_attrs_.group), CUDNN_CROSS_CORRELATION,
                                           args.params.data_type,
                                           UseTF32()));
  }

  return Status::OK();
}

template <typename T>
Status ConvTransposeGrad<T>::PrepareConvBackwardFilterArgs(const Tensor& X, const Tensor& W, const Tensor& dY,
                                                           Tensor* dW, Tensor* dB, cudnnHandle_t cudnn_handle,
                                                           ConvArgs& args) const {
  const TensorShape& x_shape = X.Shape();
  auto x_dims = x_shape.AsShapeVector();
  args.x_data = reinterpret_cast<const CudaT*>(X.template Data<T>());

  const TensorShape& y_shape = dY.Shape();
  auto y_dims = y_shape.AsShapeVector();
  args.dy_data = reinterpret_cast<const CudaT*>(dY.template Data<T>());

  const TensorShape& w_shape = W.Shape();
  auto w_dims = w_shape.AsShapeVector();

  args.y_data = nullptr;
  args.dw_data = dW ? reinterpret_cast<CudaT*>(dW->template MutableData<T>()) : nullptr;
  args.db_data = dB ? reinterpret_cast<CudaT*>(dB->template MutableData<T>()) : nullptr;
  args.dx_data = nullptr;
  args.w_data = nullptr;

  bool x_dims_changed = (args.last_x_dims != x_dims);
  bool w_dims_changed = (args.last_w_dims != w_dims);
  if (x_dims_changed || w_dims_changed) {
    if (x_dims_changed) args.last_x_dims = x_dims;
    if (w_dims_changed) args.last_w_dims = w_dims;

    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&X, &W));

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(w_shape, kernel_shape));
    auto rank = kernel_shape.size();

    ConvPadVector pads(conv_attrs_.pads);
    if (pads.empty()) {
      pads.resize(rank * 2, 0);
    }

    TensorShapeVector dilations(conv_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(rank, 1);
    }

    TensorShapeVector strides(conv_attrs_.strides);
    if (strides.empty()) {
      strides.resize(rank, 1);
    }

    const CUDAExecutionProvider* cuda_ep =
        static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());

    if (rank < 2) {
      if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
        x_dims.insert(x_dims.begin() + 2, 1);
        y_dims.insert(y_dims.begin() + 2, 1);
        w_dims.insert(w_dims.begin() + 2, 1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.begin(), 0);
        kernel_shape.insert(kernel_shape.begin(), 1);
        strides.insert(strides.begin(), 1);
        dilations.insert(dilations.begin(), 1);
      } else {
        x_dims.push_back(1);
        y_dims.push_back(1);
        w_dims.push_back(1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }
    }

    memset(&args.params, 0, sizeof(ConvParams));
    args.params.device_id = static_cast<int8_t>(cuda_ep->GetDeviceId());
    args.params.data_type = CudnnTensor::GetDataType<CudaT>();
    args.params.input_dim = static_cast<uint8_t>(x_dims.size());
    for (size_t i = 0; i < x_dims.size(); i++) {
      args.params.input_size[i] = static_cast<int>(x_dims[i]);
      args.params.weight_size[i] = static_cast<int>(w_dims[i]);
    }
    for (size_t i = 0; i < rank; i++) {
      args.params.padding[i] = static_cast<int>(pads[i]);
      args.params.padding[i + rank] = static_cast<int>(pads[i + rank]);
      args.params.stride[i] = static_cast<int>(strides[i]);
      args.params.dilation[i] = static_cast<int>(dilations[i]);
    }
    args.params.groups = conv_attrs_.group;
    int algo_mode = cuda_ep->GetCudnnConvAlgo();
    ORT_ENFORCE(algo_mode > -1 && algo_mode < 3,
                "Algo mode should be EXHAUSTIVE (0), HEURISTIC (1) or DEFAULT (2), but got ", algo_mode);
    args.params.algo_mode = algo_mode;

    args.handle = cudnn_handle;
    ORT_RETURN_IF_ERROR(args.w_desc.Set(w_dims, args.params.data_type));
    ORT_RETURN_IF_ERROR(args.x_tensor.Set(x_dims, args.params.data_type));
    ORT_RETURN_IF_ERROR(args.y_tensor.Set(y_dims, args.params.data_type));
    ORT_RETURN_IF_ERROR(args.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                           gsl::narrow_cast<int>(conv_attrs_.group), CUDNN_CROSS_CORRELATION,
                                           args.params.data_type,
                                           UseTF32()));

    if (dB) {
      const auto& b_shape = dB->Shape();
      ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
      TensorShapeVector b_dims(2 + kernel_shape.size());
      b_dims[0] = 1;           // N
      b_dims[1] = b_shape[0];  // C
      for (size_t i = 0; i < kernel_shape.size(); i++)
        b_dims[2 + i] = 1;

      ORT_RETURN_IF_ERROR(args.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime::cuda
