// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_grad.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(ConvGrad, kMSDomain, 1, T, kCudaExecutionProvider,                                   \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                ConvGrad<T>);

REGISTER_GRADIENT_KERNEL_TYPED(float)
REGISTER_GRADIENT_KERNEL_TYPED(double)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)

template <typename T>
Status ConvGrad<T>::PrepareArgs(const Tensor& x, const Tensor& dY, const Tensor& w, Tensor* dB, Tensor* dX,
                                Tensor* dW, cudnnHandle_t cudnn_handle) const {
  const TensorShape& x_shape = x.Shape();
  auto x_dims = x_shape.AsShapeVector();
  args_.x_data = reinterpret_cast<const CudaT*>(x.template Data<T>());

  const TensorShape& dy_shape = dY.Shape();
  auto dy_dims = dy_shape.AsShapeVector();
  args_.dy_data = reinterpret_cast<const CudaT*>(dY.template Data<T>());

  const TensorShape& w_shape = w.Shape();
  auto w_dims = w_shape.AsShapeVector();
  args_.w_data = reinterpret_cast<const CudaT*>(w.template Data<T>());

  args_.db_data = dB ? reinterpret_cast<CudaT*>(dB->template MutableData<T>()) : nullptr;
  args_.dx_data = dX ? reinterpret_cast<CudaT*>(dX->template MutableData<T>()) : nullptr;
  args_.dw_data = dW ? reinterpret_cast<CudaT*>(dW->template MutableData<T>()) : nullptr;

  bool x_dims_changed = (args_.last_x_dims != x_dims);
  bool w_dims_changed = (args_.last_w_dims != w_dims);
  if (x_dims_changed || w_dims_changed) {
    if (x_dims_changed) args_.last_x_dims = x_dims;
    if (w_dims_changed) args_.last_w_dims = w_dims;

    // Update Attributes
    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&x, &w));

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

    // cuDNN only takes 4D or 5D x tensor, so pad dimensions if needed.
    if (rank < 2) {
      if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
        x_dims.insert(x_dims.begin() + 2, 1);
        dy_dims.insert(dy_dims.begin() + 2, 1);
        w_dims.insert(w_dims.begin() + 2, 1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.begin(), 0);
        kernel_shape.insert(kernel_shape.begin(), 1);
        strides.insert(strides.begin(), 1);
        dilations.insert(dilations.begin(), 1);
      } else {
        x_dims.push_back(1);
        dy_dims.push_back(1);
        w_dims.push_back(1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }
    }

    memset(&args_.params, 0, sizeof(ConvParams));
    args_.params.device_id = static_cast<int8_t>(cuda_ep->GetDeviceId());
    args_.params.data_type = CudnnTensor::GetDataType<CudaT>();
    args_.params.input_dim = static_cast<uint8_t>(x_dims.size());
    for (size_t i = 0; i < x_dims.size(); i++) {
      args_.params.input_size[i] = static_cast<int>(x_dims[i]);
      args_.params.weight_size[i] = static_cast<int>(w_dims[i]);
    }
    for (size_t i = 0; i < rank; i++) {
      args_.params.padding[i] = static_cast<int>(pads[i]);
      args_.params.padding[i + rank] = static_cast<int>(pads[i + rank]);
      args_.params.stride[i] = static_cast<int>(strides[i]);
      args_.params.dilation[i] = static_cast<int>(dilations[i]);
    }
    args_.params.groups = conv_attrs_.group;
    int algo_mode = cuda_ep->GetCudnnConvAlgo();
    ORT_ENFORCE(algo_mode > -1 && algo_mode < 3,
                "Algo mode should be EXHAUSTIVE (0), HEURISTIC (1) or DEFAULT (2), but got ", algo_mode);
    args_.params.algo_mode = algo_mode;

    args_.handle = cudnn_handle;
    ORT_RETURN_IF_ERROR(args_.w_desc.Set(w_dims, args_.params.data_type));
    ORT_RETURN_IF_ERROR(args_.x_tensor.Set(x_dims, args_.params.data_type));
    ORT_RETURN_IF_ERROR(args_.y_tensor.Set(dy_dims, args_.params.data_type));
    ORT_RETURN_IF_ERROR(args_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                            gsl::narrow_cast<int>(conv_attrs_.group), CUDNN_CROSS_CORRELATION,
                                            args_.params.data_type));

    if (dB) {
      const TensorShape& db_shape = dB->Shape();
      ORT_RETURN_IF_NOT(db_shape.NumDimensions() == 1, "bias should be 1D");
      std::vector<int64_t> db_dims(2 + kernel_shape.size(), 1);
      db_dims[1] = db_shape[0];
      ORT_RETURN_IF_ERROR(args_.b_tensor.Set(db_dims, CudnnTensor::GetDataType<CudaT>()));
    }
  }

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);
  Tensor* dX = context->Output(0, X->Shape());
  Tensor* dW = context->Output(1, W->Shape());
  Tensor* dB = context->Output(2, {W->Shape()[0]});
  ORT_RETURN_IF_ERROR(PrepareArgs(*X, *dY, *W, dB, dX, dW, GetCudnnHandle(context)));
  if (dX) ORT_RETURN_IF_ERROR(ComputeInputGradient(context->GetComputeStream()));
  if (dW) ORT_RETURN_IF_ERROR(ComputeWeightGradient(context->GetComputeStream()));
  if (dB) ORT_RETURN_IF_ERROR(ComputeBiasGradient());
  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInputGradient(onnxruntime::Stream* stream) const {
  return AlgoIterator<T_BwdDataPerf>(args_).TryAll(
      static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider()),
      Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
      [&](const T_BwdDataPerf& algo_perf) -> Status {
        const auto one = Consts<CudaT>::One;
        const auto zero = Consts<CudaT>::Zero;
        IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(algo_perf.memory, stream);
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.conv_desc, algo_perf.mathType));
        CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardData(
            args_.handle, &one, args_.w_desc, args_.w_data, args_.y_tensor, args_.dy_data, args_.conv_desc,
            algo_perf.algo, workspace.get(), algo_perf.memory, &zero, args_.x_tensor, args_.dx_data));
        return Status::OK();
      });
}

template <typename T>
Status ConvGrad<T>::ComputeWeightGradient(onnxruntime::Stream* stream) const {
  return AlgoIterator<T_BwdFilterPerf>(args_).TryAll(
      static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider()),
      Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
      [&](const T_BwdFilterPerf& algo_perf) -> Status {
        const auto one = Consts<CudaT>::One;
        const auto zero = Consts<CudaT>::Zero;
        IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(algo_perf.memory, stream);
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(args_.conv_desc, algo_perf.mathType));
        CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardFilter(
            args_.handle, &one, args_.x_tensor, args_.x_data, args_.y_tensor, args_.dy_data, args_.conv_desc,
            algo_perf.algo, workspace.get(), algo_perf.memory, &zero, args_.w_desc, args_.dw_data));
        return Status::OK();
      });
}

template <typename T>
Status ConvGrad<T>::ComputeBiasGradient() const {
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardBias(args_.handle, &one, args_.y_tensor, args_.dy_data, &zero,
                                                     args_.b_tensor, args_.db_data));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
