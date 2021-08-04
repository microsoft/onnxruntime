// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/conv_grad.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL_TYPED(T)                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(ConvGrad, kMSDomain, 1, T, kCudaExecutionProvider,                                   \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                ConvGrad<T>);

REGISTER_GRADIENT_KERNEL_TYPED(float)
REGISTER_GRADIENT_KERNEL_TYPED(double)
REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16)

// TODO: we can cache the descriptors, and only update if the x shape changes
template <typename T>
Status ConvGrad<T>::PrepareArgs(const Tensor& x, const Tensor& dY, const Tensor& w, Tensor* dB, Tensor* dX,
                                Tensor* dW) const {
  const TensorShape& x_shape = x.Shape();
  std::vector<int64_t> x_dims = x_shape.GetDims();
  s_.x_data = reinterpret_cast<const CudaT*>(x.template Data<T>());

  const TensorShape& dy_shape = dY.Shape();
  std::vector<int64_t> dy_dims = dy_shape.GetDims();
  s_.dy_data = reinterpret_cast<const CudaT*>(dY.template Data<T>());

  const TensorShape& w_shape = w.Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();
  s_.w_data = reinterpret_cast<const CudaT*>(w.template Data<T>());

  s_.db_data = dB ? reinterpret_cast<CudaT*>(dB->template MutableData<T>()) : nullptr;
  s_.dx_data = dX ? reinterpret_cast<CudaT*>(dX->template MutableData<T>()) : nullptr;
  s_.dw_data = dW ? reinterpret_cast<CudaT*>(dW->template MutableData<T>()) : nullptr;

  bool x_dims_changed = s_.last_x_dims != x_dims;
  bool w_dims_changed = s_.last_w_dims != w_dims;
  if (x_dims_changed || w_dims_changed) {
    if (x_dims_changed) {
      s_.last_x_dims = x_dims;
    }

    if (w_dims_changed) {
      s_.last_w_dims = w_dims;
      s_.cached_benchmark_results.clear();
      s_.filter_cached_benchmark_results.clear();
    }

    // Update Attributes
    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&x, &w));

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

    // cudnn only takes 4D or 5D x tensor, so pad dimensions if needed
    if (rank < 2) {
      x_dims.push_back(1);
      dy_dims.push_back(1);
      w_dims.push_back(1);
      pads.insert(pads.begin() + rank, 0);
      pads.insert(pads.end(), 0);
      kernel_shape.push_back(1);
      strides.push_back(1);
      dilations.push_back(1);
    }

    if (w_dims_changed) {
      ORT_RETURN_IF_ERROR(s_.w_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));
    }
    ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims, CudnnTensor::GetDataType<CudaT>()));
    ORT_RETURN_IF_ERROR(s_.y_tensor.Set(dy_dims, CudnnTensor::GetDataType<CudaT>()));
    ORT_RETURN_IF_ERROR(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                         gsl::narrow_cast<int>(conv_attrs_.group), CUDNN_CROSS_CORRELATION,
                                         CudnnTensor::GetDataType<CudaT>()));

    if (dB) {
      const TensorShape& db_shape = dB->Shape();
      ORT_RETURN_IF_NOT(db_shape.NumDimensions() == 1, "bias should be 1D");
      std::vector<int64_t> db_dims(2 + kernel_shape.size(), 1);
      db_dims[1] = db_shape[0];
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(db_dims, CudnnTensor::GetDataType<CudaT>()));
    }

    const CUDAExecutionProvider* cuda_ep =
        static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
    int cudnn_conv_algo = cuda_ep->GetCudnnConvAlgo();
    ORT_ENFORCE(cudnn_conv_algo > -1 && cudnn_conv_algo < 3, "cudnn_conv_algo should be 0, 1 or 2, but got ",
                cudnn_conv_algo);

    if (dX) {
      if (!s_.cached_benchmark_results.contains(x_dims)) {
        cudnnConvolutionBwdDataAlgoPerf_t data_perf;
        int data_algo_count = 1;
        switch (cudnn_conv_algo) {
          case 0: {
            IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);
            CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
                CudnnHandle(), s_.w_desc, s_.w_data, s_.y_tensor, s_.dy_data, s_.conv_desc, s_.x_tensor, s_.dx_data, 1,
                &data_algo_count, &data_perf, algo_search_workspace.get(), AlgoSearchWorkspaceSize));
          } break;
          case 1: {
            CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                CudnnHandle(), s_.w_desc, s_.y_tensor, s_.conv_desc, s_.x_tensor, 1, &data_algo_count, &data_perf));
          } break;
          default: {
            data_perf.algo = kDefaultConvBwdDataAlgo;
            data_perf.mathType = std::is_same<T, MLFloat16>::value ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
            CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
                CudnnHandle(), s_.w_desc, s_.y_tensor, s_.conv_desc, s_.x_tensor, data_perf.algo, &data_perf.memory));
          }
        }

        s_.cached_benchmark_results.insert(x_dims, {data_perf.algo, data_perf.memory, data_perf.mathType});
      }

      const auto& data_perf_params = s_.cached_benchmark_results.at(x_dims);
      s_.algo = data_perf_params.algo;
      s_.workspace_bytes = data_perf_params.memory;
      CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, data_perf_params.mathType));
    }

    if (dW) {
      if (!s_.filter_cached_benchmark_results.contains(x_dims)) {
        cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;
        int filter_algo_count = 1;
        switch (cudnn_conv_algo) {
          case 0: {
            IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);
            CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                CudnnHandle(), s_.x_tensor, s_.x_data, s_.y_tensor, s_.dy_data, s_.conv_desc, s_.w_desc, s_.dw_data, 1,
                &filter_algo_count, &filter_perf, algo_search_workspace.get(), AlgoSearchWorkspaceSize));
          } break;
          case 1: {
            CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                CudnnHandle(), s_.x_tensor, s_.y_tensor, s_.conv_desc, s_.w_desc, 1, &filter_algo_count, &filter_perf));
          } break;
          default: {
            filter_perf.algo = kDefaultConvBwdFilterAlgo;
            filter_perf.mathType = std::is_same<T, MLFloat16>::value ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
            CUDNN_RETURN_IF_ERROR(
                cudnnGetConvolutionBackwardFilterWorkspaceSize(CudnnHandle(), s_.x_tensor, s_.y_tensor, s_.conv_desc,
                                                               s_.w_desc, filter_perf.algo, &filter_perf.memory));
          }
        }

        s_.filter_cached_benchmark_results.insert(x_dims, {filter_perf.algo, filter_perf.memory, filter_perf.mathType});
      }

      const auto& filter_perf_params = s_.filter_cached_benchmark_results.at(x_dims);
      s_.filter_algo = filter_perf_params.algo;
      s_.filter_workspace_bytes = filter_perf_params.memory;
      if (!dX) {
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, filter_perf_params.mathType));
      }
    }
  }

  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);

  const int64_t M = W->Shape()[0];

  Tensor* dX = context->Output(0, X->Shape());
  Tensor* dW = context->Output(1, W->Shape());
  Tensor* dB = context->Output(2, {M});

  ORT_RETURN_IF_ERROR(PrepareArgs(*X, *dY, *W, dB, dX, dW));

  if (dX) ORT_RETURN_IF_ERROR(ComputeInputGradient());
  if (dW) ORT_RETURN_IF_ERROR(ComputeWeightGradient());
  if (dB) ORT_RETURN_IF_ERROR(ComputeBiasGradient());
  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeInputGradient() const {
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes);
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardData(CudnnHandle(), &one, s_.w_desc, s_.w_data, s_.y_tensor, s_.dy_data,
                                                     s_.conv_desc, s_.algo, workspace.get(), s_.workspace_bytes, &zero,
                                                     s_.x_tensor, s_.dx_data));
  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeWeightGradient() const {
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.filter_workspace_bytes);
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionBackwardFilter(CudnnHandle(), &one, s_.x_tensor, s_.x_data, s_.y_tensor,
                                                       s_.dy_data, s_.conv_desc, s_.filter_algo, workspace.get(),
                                                       s_.filter_workspace_bytes, &zero, s_.w_desc, s_.dw_data));
  return Status::OK();
}

template <typename T>
Status ConvGrad<T>::ComputeBiasGradient() const {
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;
  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardBias(CudnnHandle(), &one, s_.y_tensor, s_.dy_data, &zero, s_.b_tensor, s_.db_data));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
