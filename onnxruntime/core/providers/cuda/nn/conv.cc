// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/tensor/slice.h"

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

static Status SliceOutUnwantedOutputSection(const void* input_data,
                                            const std::vector<int64_t>& input_dims,
                                            void* output_data,
                                            const std::vector<int64_t>& output_dims,
                                            std::vector<int64_t> starts,
                                            const std::vector<int64_t>& ends,
                                            const std::vector<int64_t>& axes,
                                            size_t element_size) {
  SliceOp::PrepareForComputeMetadata compute_metadata(input_dims);

  SliceBase::PrepareForCompute(starts, ends, axes, compute_metadata);

  // As a sanity check, ensure that the slice operator's output shape matches with the expected output shape
  ORT_ENFORCE(compute_metadata.output_dims_ == output_dims);

  return SliceCuda::Impl(input_data, input_dims, output_data, compute_metadata, element_size);
}

template <typename T>
Status Conv<T>::UpdateConvState(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();
  s_.x_data = reinterpret_cast<const void*>(X->template Data<T>());
  s_.element_size = X->DataType()->Size();

  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();
  s_.w_data = reinterpret_cast<const void*>(W->template Data<T>());

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // We may have to write the CuDNN Conv results to a temporary buffer when we deal with
  // asymmetric padding as we have to take the results written to this temporary buffer and slice out
  // extraneous portions of the result
  //IAllocatorUniquePtr<void> memory_for_cudnn_conv_results;

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
    y_dims.reserve(2 + rank);  // rank indicates number of feature dimensions - so add 2 to account for 'N' and 'C'
    y_dims.insert(y_dims.begin(), {N, M});

    std::vector<int64_t> y_dims_with_adjusted_pads;
    y_dims_with_adjusted_pads.reserve(2 + rank);  // rank indicates number of feature dimensions - so add 2 to account for 'N' and 'C'
    y_dims_with_adjusted_pads.insert(y_dims_with_adjusted_pads.begin(), {N, M});

    bool post_slicing_required = false;
    std::vector<int64_t> slice_starts;
    slice_starts.reserve(rank);

    std::vector<int64_t> slice_ends;
    slice_ends.reserve(rank);

    std::vector<int64_t> slice_axes;
    slice_axes.reserve(rank);

    ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShapeWithAdjustedPads(x_shape.Slice(2), kernel_shape,
                                                                     strides, dilations, pads, y_dims, y_dims_with_adjusted_pads,
                                                                     post_slicing_required, slice_starts, slice_ends, slice_axes));
    ORT_ENFORCE(y_dims.size() == y_dims_with_adjusted_pads.size());
    s_.y_dims = y_dims;
    s_.y_dims_with_adjusted_pads = y_dims_with_adjusted_pads;
    s_.post_slicing_required = post_slicing_required;
    s_.slice_starts = slice_starts;
    s_.slice_ends = slice_ends;
    s_.slice_axes = slice_axes;

    TensorShape y_shape(s_.y_dims);
    auto Y = context->Output(0, y_shape);
    if (Y->Shape().Size() == 0) {
      return Status::OK();
    }
    //IAllocatorUniquePtr<void> memory_for_cudnn_conv_results;
    if (post_slicing_required) {
      // Post slicing needed. Create and fill in the Conv results in an intermediate buffer.
      memory_for_cudnn_conv_results_ = GetScratchBuffer<void>(TensorShape(y_dims_with_adjusted_pads).Size() * s_.element_size);
      s_.y_data = reinterpret_cast<CudaT*>(memory_for_cudnn_conv_results_.get());
    } else {
      s_.y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
    }

    std::vector<int64_t> x_dims_cudnn = x_dims;
    std::vector<int64_t> y_dims_cudnn = !post_slicing_required ? y_dims : y_dims_with_adjusted_pads;
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

    if (w_dims_changed) {
      ORT_RETURN_IF_ERROR(s_.filter_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));
    }
    ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
    ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
    //cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    ORT_RETURN_IF_ERROR(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                         CUDNN_CROSS_CORRELATION, CudnnTensor::GetDataType<CudaT>()));
    CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(s_.conv_desc, gsl::narrow_cast<int>(conv_attrs_.group)));

    if (context->InputCount() >= 3) {
      const Tensor* B = context->Input<Tensor>(2);
      const auto& b_shape = B->Shape();
      ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
      std::vector<int64_t> b_dims(2 + kernel_shape.size());
      b_dims[0] = 1;           // N
      b_dims[1] = b_shape[0];  // C
      for (size_t i = 0; i < kernel_shape.size(); i++) b_dims[2 + i] = 1;
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      s_.b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
    }

    if (!s_.cached_benchmark_results.contains(x_dims_cudnn)) {
      IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);

      // set math type to tensor core before algorithm search
      if (std::is_same<T, MLFloat16>::value)
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));

      cudnnConvolutionFwdAlgoPerf_t perf;
      int algo_count = 1;
      const CUDAExecutionProvider* cuda_ep = static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
      int cudnn_conv_algo = cuda_ep->GetCudnnConvAlgo();
      ORT_ENFORCE(cudnn_conv_algo > -1 && cudnn_conv_algo < 3, "cudnn_conv_algo should be 0, 1 or 2, but got ", cudnn_conv_algo);
      switch (cudnn_conv_algo) {
        case 0:
          CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
              CudnnHandle(),
              s_.x_tensor,
              s_.x_data,
              s_.filter_desc,
              s_.w_data,
              s_.conv_desc,
              s_.y_tensor,
              s_.y_data,
              1,
              &algo_count,
              &perf,
              algo_search_workspace.get(),
              AlgoSearchWorkspaceSize));
          break;

        case 1:
          CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
              CudnnHandle(),
              s_.x_tensor,
              s_.filter_desc,
              s_.conv_desc,
              s_.y_tensor,
              1,
              &algo_count,
              &perf));
          break;

        default:
          perf.algo = kDefaultConvAlgo;
          CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
              CudnnHandle(),
              s_.x_tensor,
              s_.filter_desc,
              s_.conv_desc,
              s_.y_tensor,
              perf.algo,
              &perf.memory));
          if (std::is_same<T, MLFloat16>::value) {
            perf.mathType = CUDNN_TENSOR_OP_MATH;
          } else {
            perf.mathType = CUDNN_DEFAULT_MATH;
          }
      }

      s_.cached_benchmark_results.insert(x_dims_cudnn, {perf.algo, perf.memory, perf.mathType});
    }

    const auto& perf = s_.cached_benchmark_results.at(x_dims_cudnn);
    CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
    s_.algo = perf.algo;
    s_.workspace_bytes = perf.memory;
  }
  /*
  if (!s_.y_data) {
    Y = context->Output(0, TensorShape(s_.y_dims));
    // special case when there is a dim value of 0 in the shape.
    if (Y->Shape().Size() == 0)
      return Status::OK();

    if (!s_.post_slicing_required) {
      s_.y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
    } else {
      // Post slicing needed. Create and fill in the Conv results in an intermediate buffer.
      memory_for_cudnn_conv_results = GetScratchBuffer<void>(TensorShape(s_.y_dims_with_adjusted_pads).Size() * element_size);
      s_.y_data = reinterpret_cast<CudaT*>(memory_for_cudnn_conv_results.get());
    }
  }*/
  return Status::OK();
}  // UpdateState

template <typename T>
Status Conv<T>::ComputeInternal(OpKernelContext* context) const {
  std::lock_guard<OrtMutex> lock(s_.mutex);
  ORT_RETURN_IF_ERROR(UpdateConvState(context));
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(CudnnHandle(),
                                                &alpha_,
                                                s_.x_tensor,
                                                s_.x_data,
                                                s_.filter_desc,
                                                s_.w_data,
                                                s_.conv_desc,
                                                s_.algo,
                                                GetScratchBuffer<void>(s_.workspace_bytes).get(),
                                                s_.workspace_bytes,
                                                &beta_,
                                                s_.y_tensor,
                                                s_.y_data));

  if (nullptr != s_.b_data) {
    //const Tensor* B = context->Input<Tensor>(2);
    //auto b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha_, s_.b_tensor, s_.b_data, &alpha_, s_.y_tensor,
                                         s_.y_data));
  }

  // To deal with asymmetric padding, we may have over-padded on one or both sides of the spatial dimensions
  // This may have lead to extra results that are unnecessary and hence we slice that off here
  if (s_.post_slicing_required) {
    SliceOutUnwantedOutputSection(s_.y_data, s_.y_dims_with_adjusted_pads, context->Output<T>(0),
                                  s_.y_dims, s_.slice_starts, s_.slice_ends, s_.slice_axes, s_.element_size);
  }

  return Status::OK();
}
/*
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

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  size_t element_size = X->DataType()->Size();

  Tensor* Y = nullptr;

  // We may have to write the CuDNN Conv results to a temporary buffer when we deal with
  // asymmetric padding as we have to take the results written to this temporary buffer and slice out
  // extraneous portions of the result
  IAllocatorUniquePtr<void> memory_for_cudnn_conv_results;

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
      y_dims.reserve(2 + rank);  // rank indicates number of feature dimensions - so add 2 to account for 'N' and 'C'
      y_dims.insert(y_dims.begin(), {N, M});

      std::vector<int64_t> y_dims_with_adjusted_pads;
      y_dims_with_adjusted_pads.reserve(2 + rank);  // rank indicates number of feature dimensions - so add 2 to account for 'N' and 'C'
      y_dims_with_adjusted_pads.insert(y_dims_with_adjusted_pads.begin(), {N, M});

      bool post_slicing_required = false;
      std::vector<int64_t> slice_starts;
      slice_starts.reserve(rank);

      std::vector<int64_t> slice_ends;
      slice_ends.reserve(rank);

      std::vector<int64_t> slice_axes;
      slice_axes.reserve(rank);

      ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShapeWithAdjustedPads(x_shape.Slice(2), kernel_shape,
                                                                       strides, dilations, pads, y_dims, y_dims_with_adjusted_pads,
                                                                       post_slicing_required, slice_starts, slice_ends, slice_axes));
      ORT_ENFORCE(y_dims.size() == y_dims_with_adjusted_pads.size());
      s_.y_dims = y_dims;
      s_.y_dims_with_adjusted_pads = y_dims_with_adjusted_pads;
      s_.post_slicing_required = post_slicing_required;
      s_.slice_starts = slice_starts;
      s_.slice_ends = slice_ends;
      s_.slice_axes = slice_axes;

      TensorShape y_shape(s_.y_dims);
      Y = context->Output(0, y_shape);
      if (!post_slicing_required) {
        // No post slicing needed. Fill the output tensor's buffer directly.
        y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
      } else {
        // Post slicing needed. Create and fill in the Conv results in an intermediate buffer.
        memory_for_cudnn_conv_results = GetScratchBuffer<void>(TensorShape(y_dims_with_adjusted_pads).Size() * element_size);
        y_data = reinterpret_cast<CudaT*>(memory_for_cudnn_conv_results.get());
      }

      std::vector<int64_t> x_dims_cudnn = x_dims;
      std::vector<int64_t> y_dims_cudnn = !post_slicing_required ? y_dims : y_dims_with_adjusted_pads;
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

      char activation_mode_[10] = {'\0'};
      if (GetEnvironmentVariableA(context->GetNodeName().c_str(), activation_mode_, 10) > 0) {
        s_.activation_descriptor.Set(activation_mode_);
      }

      if (!s_.cached_benchmark_results.contains(x_dims_cudnn)) {
        IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);

        // set math type to tensor core before algorithm search
        if (std::is_same<T, MLFloat16>::value)
          CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));

        cudnnConvolutionFwdAlgoPerf_t perf;
        int algo_count = 1;
        const CUDAExecutionProvider* cuda_ep = static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
        int cudnn_conv_algo = cuda_ep->GetCudnnConvAlgo();
        ORT_ENFORCE(cudnn_conv_algo > -1 && cudnn_conv_algo < 3, "cudnn_conv_algo should be 0, 1 or 2, but got ", cudnn_conv_algo);
        switch (cudnn_conv_algo) {
          case 0:
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
            break;

          case 1:
            CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
                CudnnHandle(),
                s_.x_tensor,
                s_.filter_desc,
                s_.conv_desc,
                s_.y_tensor,
                1,
                &algo_count,
                &perf));
            break;

          default:
            perf.algo = kDefaultConvAlgo;
            CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
                CudnnHandle(),
                s_.x_tensor,
                s_.filter_desc,
                s_.conv_desc,
                s_.y_tensor,
                perf.algo,
                &perf.memory));
            if (std::is_same<T, MLFloat16>::value) {
              perf.mathType = CUDNN_TENSOR_OP_MATH;
            } else {
              perf.mathType = CUDNN_DEFAULT_MATH;
            }
        }

        s_.cached_benchmark_results.insert(x_dims_cudnn, {perf.algo, perf.memory, perf.mathType});
      }

      const auto& perf = s_.cached_benchmark_results.at(x_dims_cudnn);
      CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, perf.mathType));
      s_.algo = perf.algo;
      s_.workspace_bytes = perf.memory;
    }

    if (!y_data) {
      Y = context->Output(0, TensorShape(s_.y_dims));
      // special case when there is a dim value of 0 in the shape.
      if (Y->Shape().Size() == 0)
        return Status::OK();

      if (!s_.post_slicing_required) {
        y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
      } else {
        // Post slicing needed. Create and fill in the Conv results in an intermediate buffer.
        memory_for_cudnn_conv_results = GetScratchBuffer<void>(TensorShape(s_.y_dims_with_adjusted_pads).Size() * element_size);
        y_data = reinterpret_cast<CudaT*>(memory_for_cudnn_conv_results.get());
      }
    }

    const auto alpha = Consts<CudaT>::One;
    const auto beta = Consts<CudaT>::Zero;

    IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes);

    if (s_.activation_descriptor.HasActivation()) {
      //std::cout << "conv activation on: " << context->GetNodeName() << std::endl;
      CUDNN_RETURN_IF_ERROR(cudnnConvolutionBiasActivationForward(CudnnHandle(),
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
                                                                  y_data,
                                                                  has_bias ? s_.b_tensor : s_.y_tensor,
                                                                  has_bias ? reinterpret_cast<const CudaT*>(context->Input<Tensor>(2)->template Data<T>()) : y_data,
                                                                  s_.activation_descriptor,
                                                                  s_.y_tensor,
                                                                  y_data));
    } else {
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
                                                    &beta,  //has_bias ? &alpha : &beta,
                                                    s_.y_tensor,
                                                    y_data));

      if (has_bias) {
        const Tensor* B = context->Input<Tensor>(2);
        auto b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
        CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, s_.b_tensor, b_data, &alpha, s_.y_tensor,
                                             y_data));
      }
    }
    // To deal with asymmetric padding, we may have over-padded on one or both sides of the spatial dimensions
    // This may have lead to extra results that are unnecessary and hence we slice that off here
    if (s_.post_slicing_required) {
      SliceOutUnwantedOutputSection(y_data, s_.y_dims_with_adjusted_pads, Y->MutableDataRaw(),
                                    s_.y_dims, s_.slice_starts, s_.slice_ends, s_.slice_axes, element_size);
    }
  }

  return Status::OK();
}
*/

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

CudnnActivationDescriptor::CudnnActivationDescriptor() : desc_(nullptr) {
}

CudnnActivationDescriptor::~CudnnActivationDescriptor() {
  Reset();
}

void CudnnActivationDescriptor::Reset() {
  if (desc_ != nullptr) {
    cudnnDestroyActivationDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status CudnnActivationDescriptor::Set(const char* activaton_mode) {
  Reset();
  cudnnActivationMode_t mode;
  if (strcmp(activaton_mode, "Sigmoid") == 0) {
    mode = cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID;
  } else if (strcmp(activaton_mode, "Relu") == 0) {
    mode = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU;
  } else if (strcmp(activaton_mode, "Tanh") == 0) {
    mode = cudnnActivationMode_t::CUDNN_ACTIVATION_TANH;
  } else {
    return Status(common::StatusCategory::ONNXRUNTIME,
                  common::StatusCode::INVALID_ARGUMENT,
                  "unsupported conv activation mode");
  }
  CUDNN_RETURN_IF_ERROR(cudnnCreateActivationDescriptor(&desc_));
  CUDNN_RETURN_IF_ERROR(cudnnSetActivationDescriptor(desc_, mode,
                                                     cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                                                     std::numeric_limits<double>::max()));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
