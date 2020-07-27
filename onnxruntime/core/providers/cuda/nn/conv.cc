// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

// Op Set 11 for Conv only update document to clearify default dilations and strides value.
// which are already convered by op set 11 cpu versoin, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      Conv,                                                                     \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Conv,                                                                     \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status Conv<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();
  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());

  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();
  auto w_data = reinterpret_cast<const CudaT*>(W->template Data<T>());

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  bool has_bias = (num_inputs == 3);

  CudaT* y_data = nullptr;

  {
    std::lock_guard<OrtMutex> lock(s_.mutex);
    // TODO: add a global cache if need to handle cases for multiple frames running simultaneuously with different batch_size
    bool input_dims_changed = (s_.last_x_dims != x_dims);
    bool w_dims_changed = (s_.last_w_dims != w_dims);
    if (input_dims_changed || w_dims_changed) {
      if (input_dims_changed)
        s_.last_x_dims = x_dims;

      if (w_dims_changed) {
        s_.last_w_dims = w_dims;
        s_.cached_benchmark_results.clear();
      }

      const int64_t N = X->Shape()[0];
      const int64_t M = W->Shape()[0];

      ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

      std::vector<int64_t> kernel_shape;
      ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));
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

      std::vector<int64_t> y_dims;
      y_dims.insert(y_dims.begin(), {N, M});
      ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(x_shape.Slice(2), kernel_shape,
                                                       strides, dilations, pads, y_dims, true));
      s_.y_dims = y_dims;
      Tensor* Y = context->Output(0, TensorShape(s_.y_dims));
      y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

      std::vector<int64_t> x_dims_cudnn = x_dims;
      std::vector<int64_t> y_dims_cudnn = y_dims;
      if (rank < 2) {
        // cudnn only takes 4D or 5D input, so pad dimensions if needed
        x_dims_cudnn.push_back(1);
        y_dims_cudnn.push_back(1);
        w_dims.push_back(1);
        pads.insert(pads.begin() + rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }

      if (w_dims_changed)
        ORT_RETURN_IF_ERROR(s_.filter_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

      // Special case when there is a dim value of 0 in the shape.
      // Return only after we have cached the following for subsequent runs :
      // 1) `w_dims` in the `filter_desc`
      // 2) `y_dims` in s_.y_dims
      if (Y->Shape().Size() == 0) {
        return Status::OK();
      }

      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));

      cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
      ORT_RETURN_IF_ERROR(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                           mode, CudnnTensor::GetDataType<CudaT>()));
      CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(s_.conv_desc, gsl::narrow_cast<int>(conv_attrs_.group)));

      if (has_bias) {
        const Tensor* B = context->Input<Tensor>(2);
        const auto& b_shape = B->Shape();
        ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
        std::vector<int64_t> b_dims(2 + kernel_shape.size());
        b_dims[0] = 1;           // N
        b_dims[1] = b_shape[0];  // C
        for (size_t i = 0; i < kernel_shape.size(); i++) b_dims[2 + i] = 1;

        ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      }

      if (!s_.cached_benchmark_results.contains(x_dims_cudnn)) {
        IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);

        // set math type to tensor core before algorithm search
        if (std::is_same<T, MLFloat16>::value)
          CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));

        cudnnConvolutionFwdAlgoPerf_t perf;
        int algo_count = 1;
        CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
            CudnnHandle(),
            s_.x_tensor,
            x_data,
            s_.filter_desc,
            w_data,
            s_.conv_desc,
            s_.y_tensor,
            y_data,
            1,
            &algo_count,
            &perf,
            algo_search_workspace.get(),
            AlgoSearchWorkspaceSize));
        s_.cached_benchmark_results.insert(x_dims_cudnn, {perf.algo, perf.memory, perf.mathType});
      }

      const auto& perf = s_.cached_benchmark_results.at(x_dims_cudnn);
      CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
      s_.algo = perf.algo;
      s_.workspace_bytes = perf.memory;
    }

    if (!y_data) {
      Tensor* Y = context->Output(0, TensorShape(s_.y_dims));
      // special case when there is a dim value of 0 in the shape.
      if (Y->Shape().Size() == 0)
        return Status::OK();

      y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
    }

    const auto alpha = Consts<CudaT>::One;
    const auto beta = Consts<CudaT>::Zero;

    IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes);

    CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(CudnnHandle(),
                                                  &alpha,
                                                  s_.x_tensor,
                                                  x_data,
                                                  s_.filter_desc,
                                                  w_data,
                                                  s_.conv_desc,
                                                  s_.algo,
                                                  workspace.get(),
                                                  s_.workspace_bytes,
                                                  &beta,
                                                  s_.y_tensor,
                                                  y_data));

    if (has_bias) {
      const Tensor* B = context->Input<Tensor>(2);
      auto b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
      CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, s_.b_tensor, b_data, &alpha, s_.y_tensor, y_data));
    }
  }

  return Status::OK();
}

CudnnConvolutionDescriptor::CudnnConvolutionDescriptor() : desc_(nullptr) {
}

CudnnConvolutionDescriptor::~CudnnConvolutionDescriptor() {
  if (desc_ != nullptr) {
    cudnnDestroyConvolutionDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status CudnnConvolutionDescriptor::Set(
    size_t rank,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& dilations,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t data_type) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateConvolutionDescriptor(&desc_));

  std::vector<int> pad_dims(rank);
  std::vector<int> stride_dims(rank);
  std::vector<int> dilation_dims(rank);
  for (size_t i = 0; i < rank; i++) {
    pad_dims[i] = gsl::narrow_cast<int>(pads[i]);
    stride_dims[i] = gsl::narrow_cast<int>(strides[i]);
    dilation_dims[i] = gsl::narrow_cast<int>(dilations[i]);
  }

  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionNdDescriptor(
      desc_,
      gsl::narrow_cast<int>(rank),
      pad_dims.data(),
      stride_dims.data(),
      dilation_dims.data(),
      mode,
      data_type));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
