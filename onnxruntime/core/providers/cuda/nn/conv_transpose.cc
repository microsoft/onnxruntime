// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_transpose.h"

namespace onnxruntime {
namespace cuda {

// Op Set 11 for ConvTranspose only update document to clearify default dilations and strides value.
// which are already covered by op set 11 cpu version, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      ConvTranspose,                                                            \
      kOnnxDomain,                                                              \
      1, 10,                                                                    \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T>);                                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      ConvTranspose,                                                            \
      kOnnxDomain,                                                              \
      11,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTranspose<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status ConvTranspose<T>::ComputeInternal(OpKernelContext* context) const {
  return DoConvTranspose(context, false);
}

template <typename T>
Status ConvTranspose<T>::DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();
  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());

  if (X->Shape().NumDimensions() != 4) {
    // This condition is not true for two tests in ONNX tests series:
    // test_convtranspose_1d_cpu, test_convtranspose_3d_cpu.
    // TODO: the error message should tell which operator raises it.
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input X must be 4-dimensional.",
                           " X: ", X->Shape().ToString().c_str());
  }
  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();
  auto w_data = reinterpret_cast<const CudaT*>(W->template Data<T>());

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  bool has_bias = dynamic_padding ? num_inputs == 4 : num_inputs == 3;

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

      ConvTransposeAttributes::Prepare p;
      ORT_RETURN_IF_ERROR(conv_transpose_attrs_.PrepareForCompute(context, has_bias, p, dynamic_padding));

      const auto& y_dims = p.Y->Shape().GetDims();
      s_.y_dims = y_dims;

      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims, CudnnTensor::GetDataType<CudaT>()));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims, CudnnTensor::GetDataType<CudaT>()));

      if (w_dims_changed)
        ORT_RETURN_IF_ERROR(s_.filter_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));

      cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
      ORT_RETURN_IF_ERROR(s_.conv_desc.Set(p.kernel_shape.size(), p.pads, p.strides,
                                           p.dilations, mode, CudnnTensor::GetDataType<CudaT>()));
      CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(s_.conv_desc,
                                                          gsl::narrow_cast<int>(conv_transpose_attrs_.group)));

      if (has_bias) {
        const auto& b_shape = p.B->Shape();
        ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
        std::vector<int64_t> b_dims(2 + p.kernel_shape.size());
        b_dims[0] = 1;           // N
        b_dims[1] = b_shape[0];  // C
        for (size_t i = 0; i < p.kernel_shape.size(); i++)
          b_dims[2 + i] = 1;

        ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      }

      y_data = reinterpret_cast<CudaT*>(p.Y->template MutableData<T>());

      if (!s_.cached_benchmark_results.contains(x_dims)) {
        IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);

        // set math type to tensor core before algorithm search
        if (std::is_same<T, MLFloat16>::value)
          CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));

        cudnnConvolutionBwdDataAlgoPerf_t perf;
        int algo_count = 1;
        CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionBackwardDataAlgorithmEx(
            CudnnHandle(),
            s_.filter_desc,
            w_data,
            s_.x_tensor,
            x_data,
            s_.conv_desc,
            s_.y_tensor,
            y_data,
            1,
            &algo_count,
            &perf,
            algo_search_workspace.get(),
            AlgoSearchWorkspaceSize));
        s_.cached_benchmark_results.insert(x_dims, {perf.algo, perf.memory, perf.mathType});
      }

      const auto& perf = s_.cached_benchmark_results.at(x_dims);
      CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
      s_.algo = perf.algo;
      s_.workspace_bytes = perf.memory;
    }
  }

  if (!y_data) {
    Tensor* Y = context->Output(0, TensorShape(s_.y_dims));
    y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
  }

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;

  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes);

  CUDNN_RETURN_IF_ERROR(
      cudnnConvolutionBackwardData(
          CudnnHandle(),
          &alpha,
          s_.filter_desc,
          w_data,
          s_.x_tensor,
          x_data,
          s_.conv_desc,
          s_.algo,
          workspace.get(),
          s_.workspace_bytes,
          &beta,
          s_.y_tensor,
          y_data));

  if (has_bias) {
    const Tensor* B = dynamic_padding ? context->Input<Tensor>(3) : context->Input<Tensor>(2);
    auto b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, s_.b_tensor, b_data, &alpha, s_.y_tensor, y_data));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
