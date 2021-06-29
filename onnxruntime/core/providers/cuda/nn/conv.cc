// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/tensor/slice.h"

namespace onnxruntime {
namespace cuda {

// Op Set 11 for Conv only update document to clearify default dilations and strides value.
// which are already convered by op set 11 cpu versoin, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      11,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

Status SliceOutUnwantedOutputSection(cudaStream_t stream,
                                     const void* input_data, const std::vector<int64_t>& input_dims,
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

  return SliceCuda::Impl(stream, input_data, input_dims, output_data, compute_metadata, element_size);
}

template <typename T>
Status Conv<T>::UpdateState(OpKernelContext* context, bool bias_expected) const {
  //set X
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();
  s_.x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  s_.element_size = X->DataType()->Size();
  //set W
  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  std::vector<int64_t> w_dims = w_shape.GetDims();
  s_.w_data = reinterpret_cast<const CudaT*>(W->template Data<T>());
  //set B
  if (context->InputCount() >= 3) {
    const Tensor* B = context->Input<Tensor>(2);
    s_.b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
  } else {
    s_.b_data = nullptr;
  }
  //set Z
  if (context->InputCount() >= 4) {
    const Tensor* Z = context->Input<Tensor>(3);
    ORT_RETURN_IF_ERROR(s_.z_tensor.Set(Z->Shape().GetDims(), CudnnTensor::GetDataType<CudaT>()));
    s_.z_data = reinterpret_cast<const CudaT*>(Z->template Data<T>());
  } else {
    s_.z_data = nullptr;
  }
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

    s_.Y = context->Output(0, TensorShape(s_.y_dims));
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    if (post_slicing_required) {
      // Post slicing needed. Create and fill in the Conv results in an intermediate buffer.
      s_.memory_for_cudnn_conv_results = GetScratchBuffer<void>(TensorShape(y_dims_with_adjusted_pads).Size() * s_.element_size);
      s_.y_data = reinterpret_cast<CudaT*>(s_.memory_for_cudnn_conv_results.get());
    } else {
      // No post slicing needed. Fill the output tensor's buffer directly.
      s_.y_data = reinterpret_cast<CudaT*>(s_.Y->template MutableData<T>());
    }

    std::vector<int64_t> x_dims_cudnn = x_dims;
    std::vector<int64_t> y_dims_cudnn = !post_slicing_required ? y_dims : y_dims_with_adjusted_pads;
    if (rank < 2) {
      // TODO: Explore padding the provided input shape [N, C, D] to [N, C, 1, D]
      // especially for EXHAUSTIVE algo search which may result in a better algo selection.
      // Currently, we are padding it to [N, C, D, 1] as this seems to be the sweet spot
      // for all algo search options: EXHAUSTIVE, HEURISTIC, and DEFAULT.
      // See PR #7348 for more context.
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
      ORT_RETURN_IF_ERROR(s_.w_desc.Set(w_dims, CudnnTensor::GetDataType<CudaT>()));
    }
    ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
    ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
    ORT_RETURN_IF_ERROR(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                         gsl::narrow_cast<int>(conv_attrs_.group),
                                         CUDNN_CROSS_CORRELATION, CudnnTensor::GetDataType<CudaT>()));

    if (context->InputCount() >= 3) {
      const Tensor* B = context->Input<Tensor>(2);
      const auto& b_shape = B->Shape();
      ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
      std::vector<int64_t> b_dims(2 + kernel_shape.size(), 1);
      b_dims[1] = b_shape[0];
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      //s_.b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
    } else if (bias_expected) {
      std::vector<int64_t> b_dims(2 + kernel_shape.size(), 1);
      b_dims[1] = w_dims[0];
      auto malloc_size = b_dims[1] * sizeof(CudaT);
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>()));
      if (s_.b_zero) {
        CUDA_CALL_THROW(cudaFree(s_.b_zero));
        s_.b_zero = nullptr;
      }
      CUDA_CALL_THROW(cudaMalloc(&s_.b_zero, malloc_size));
      CUDA_CALL_THROW(cudaMemsetAsync(s_.b_zero, 0, malloc_size, Stream()));
    }

    if (!s_.cached_benchmark_results.contains(x_dims_cudnn)) {
      // set math type to tensor core before algorithm search
      if (std::is_same<T, MLFloat16>::value)
        CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));

      cudnnConvolutionFwdAlgoPerf_t perf;
      int algo_count = 1;
      const CUDAExecutionProvider* cuda_ep = static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());
      int cudnn_conv_algo = cuda_ep->GetCudnnConvAlgo();
      ORT_ENFORCE(cudnn_conv_algo > -1 && cudnn_conv_algo < 3, "cudnn_conv_algo should be 0, 1 or 2, but got ", cudnn_conv_algo);
      switch (cudnn_conv_algo) {
        case 0: {
          IAllocatorUniquePtr<void> algo_search_workspace = GetScratchBuffer<void>(AlgoSearchWorkspaceSize);
          CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
              CudnnHandle(),
              s_.x_tensor,
              s_.x_data,
              s_.w_desc,
              s_.w_data,
              s_.conv_desc,
              s_.y_tensor,
              s_.y_data,
              1,            // requestedAlgoCount
              &algo_count,  // returnedAlgoCount
              &perf,
              algo_search_workspace.get(),
              AlgoSearchWorkspaceSize));
          break;
        }
        case 1:
          CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
              CudnnHandle(),
              s_.x_tensor,
              s_.w_desc,
              s_.conv_desc,
              s_.y_tensor,
              1,            // requestedAlgoCount
              &algo_count,  // returnedAlgoCount
              &perf));
          break;

        default:
          perf.algo = kDefaultConvAlgo;
          CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
              CudnnHandle(),
              s_.x_tensor,
              s_.w_desc,
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
  } else {
    //set Y
    s_.Y = context->Output(0, TensorShape(s_.y_dims));
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    if (s_.post_slicing_required) {
      s_.memory_for_cudnn_conv_results = GetScratchBuffer<void>(TensorShape(s_.y_dims_with_adjusted_pads).Size() * s_.element_size);
      s_.y_data = reinterpret_cast<CudaT*>(s_.memory_for_cudnn_conv_results.get());
    } else {
      s_.y_data = reinterpret_cast<CudaT*>(s_.Y->template MutableData<T>());
    }
  }
  return Status::OK();
}

template <typename T>
Status Conv<T>::ComputeInternal(OpKernelContext* context) const {
  std::lock_guard<OrtMutex> lock(s_.mutex);
  ORT_RETURN_IF_ERROR(UpdateState(context));
  if (s_.Y->Shape().Size() == 0) {
    return Status::OK();
  }
  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  IAllocatorUniquePtr<void> workspace = GetWorkSpace();
  CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(CudnnHandle(),
                                                &alpha,
                                                s_.x_tensor,
                                                s_.x_data,
                                                s_.w_desc,
                                                s_.w_data,
                                                s_.conv_desc,
                                                s_.algo,
                                                workspace.get(),
                                                s_.workspace_bytes,
                                                &beta,
                                                s_.y_tensor,
                                                s_.y_data));
  if (nullptr != s_.b_data) {
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(CudnnHandle(), &alpha, s_.b_tensor, s_.b_data,
                                         &alpha, s_.y_tensor, s_.y_data));
  }
  // To deal with asymmetric padding, we may have over-padded on one or both sides of the spatial dimensions
  // This may have lead to extra results that are unnecessary and hence we slice that off here
  if (s_.post_slicing_required) {
    SliceOutUnwantedOutputSection(Stream(), s_.y_data, s_.y_dims_with_adjusted_pads, s_.Y->MutableDataRaw(),
                                  s_.y_dims, s_.slice_starts, s_.slice_ends, s_.slice_axes, s_.element_size);
  }
  return Status::OK();
}  // namespace cuda

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
    int groups,
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

  // This piece of code is copied from /pytorch/aten/src/ATen/cudnn/Descriptors.h
  // Setting math_type to CUDNN_DATA_FLOAT for half input
  cudnnDataType_t math_type = data_type;
  if (data_type == CUDNN_DATA_HALF) math_type = CUDNN_DATA_FLOAT;
  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionNdDescriptor(
      desc_,
      gsl::narrow_cast<int>(rank),
      pad_dims.data(),
      stride_dims.data(),
      dilation_dims.data(),
      mode,
      math_type));

  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionGroupCount(desc_, groups));

  // Copied from /pytorch/aten/src/ATen/cudnn/Descriptors.h
  // See Note [behavior of cudnnFind and cudnnGet] at /pytorch/aten/src/ATen/native/cudnn/Conv_v7.cpp
  CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(desc_, CUDNN_DEFAULT_MATH));
  if (data_type == CUDNN_DATA_HALF) {
    CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(desc_, CUDNN_TENSOR_OP_MATH));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
