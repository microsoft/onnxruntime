// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

Status SliceOutUnwantedOutputSection(cudaStream_t stream,
                                     const void* input_data, gsl::span<const int64_t> input_dims,
                                     void* output_data,
                                     const gsl::span<const int64_t>& output_dims,
                                     const gsl::span<const int64_t>& starts,
                                     const gsl::span<const int64_t>& ends,
                                     const gsl::span<const int64_t>& axes,
                                     size_t element_size) {
  SliceOp::PrepareForComputeMetadata compute_metadata(input_dims);

  ORT_THROW_IF_ERROR(SliceBase::PrepareForCompute(starts, ends, axes, compute_metadata));

  // As a sanity check, ensure that the slice operator's output shape matches with the expected output shape
  ORT_ENFORCE(SpanEq(gsl::make_span(compute_metadata.output_dims_), output_dims));

  return ::onnxruntime::cuda::SliceCuda::Impl(stream, input_data, input_dims, output_data,
                                              compute_metadata, element_size);
}

static cudnnStatus_t GetWorkspaceSize(cudnnHandle_t handle,
                                      const ::onnxruntime::cuda::CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s,
                                      cudnnConvolutionFwdAlgo_t algo,
                                      size_t* sz) {
  return cudnnGetConvolutionForwardWorkspaceSize(handle, s.x_tensor, s.w_desc, s.conv_desc, s.y_tensor, algo, sz);
}

static size_t GetMaxWorkspaceSize(cudnnHandle_t handle,
                                  const ::onnxruntime::cuda::CudnnConvState<cudnnConvolutionFwdAlgoPerf_t>& s,
                                  const cudnnConvolutionFwdAlgo_t* algo, int n_algo) {
  // TODO: get maximum available size from memory arena
  size_t free, total;
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));
  // Assuming 10% of fragmentation
  free = static_cast<size_t>(static_cast<double>(free) * 0.9);
  size_t max_ws_size = 0;
  for (int i = 0; i < n_algo; i++) {
    cudnnStatus_t err;
    size_t sz;
    err = GetWorkspaceSize(handle, s, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > free) continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

template <typename T>
class FusedConv : public onnxruntime::cuda::CudaKernel {
  using CudaT = typename ::onnxruntime::cuda::ToCudaType<T>::MappedType;

 public:
  FusedConv(const OpKernelInfo& info) : onnxruntime::cuda::CudaKernel(info), conv_attrs_(info) {
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
    std::string activation;
    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("activation", &activation));
    ORT_THROW_IF_ERROR(MapMode(activation));
    CUDNN_CALL_THROW(cudnnCreateActivationDescriptor(&activation_desc_));
    CUDNN_CALL_THROW(cudnnSetActivationDescriptor(
        activation_desc_, activation_mode_, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        std::numeric_limits<double>::max()));
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FusedConv);

  ~FusedConv() {
    if (activation_desc_) {
      CUDNN_CALL_THROW(cudnnDestroyActivationDescriptor(activation_desc_));
      activation_desc_ = nullptr;
    }
  }

  Status UpdateState(OpKernelContext* context, bool bias_expected) const {
    // set X
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& x_shape = X->Shape();
    const auto x_dims = x_shape.AsShapeVector();
    s_.x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
    s_.element_size = X->DataType()->Size();
    // set W
    const Tensor* W = context->Input<Tensor>(1);
    const TensorShape& w_shape = W->Shape();
    auto w_dims = w_shape.AsShapeVector();
    s_.w_data = reinterpret_cast<const CudaT*>(W->Data<T>());

    // set B
    if (context->InputCount() >= 3) {
      const Tensor* B = context->Input<Tensor>(2);
      s_.b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
    } else {
      s_.b_data = nullptr;
    }
    // set Z
    if (context->InputCount() >= 4) {
      const Tensor* Z = context->Input<Tensor>(3);
      ORT_RETURN_IF_ERROR(s_.z_tensor.Set(Z->Shape().GetDims(),
                                          ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>()));
      s_.z_data = reinterpret_cast<const CudaT*>(Z->Data<T>());
    } else {
      s_.z_data = nullptr;
    }
    bool input_dims_changed = (s_.last_x_dims != x_dims);
    bool w_dims_changed = (s_.last_w_dims != w_dims);
    if (input_dims_changed || w_dims_changed) {
      if (input_dims_changed)
        s_.last_x_dims = gsl::make_span(x_dims);

      if (w_dims_changed) {
        s_.last_w_dims = gsl::make_span(w_dims);
        s_.cached_benchmark_results.clear();
      }

      ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W->Shape()));

      TensorShapeVector kernel_shape;
      ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

      const size_t kernel_rank = kernel_shape.size();

      ConvPadVector pads(conv_attrs_.pads);
      if (pads.empty()) {
        pads.resize(kernel_rank * 2, 0);
      }
      TensorShapeVector dilations(conv_attrs_.dilations);
      if (dilations.empty()) {
        dilations.resize(kernel_rank, 1);
      }
      TensorShapeVector strides(conv_attrs_.strides);
      if (strides.empty()) {
        strides.resize(kernel_rank, 1);
      }

      TensorShapeVector y_dims;
      y_dims.reserve(2 + kernel_rank);  // add 2 to account for 'N' and 'C'

      const int64_t N = X->Shape()[0];
      const int64_t M = W->Shape()[0];
      y_dims.insert(y_dims.begin(), {N, M});

      bool post_slicing_required = false;
      TensorShapeVector slice_starts;
      slice_starts.reserve(kernel_rank);

      TensorShapeVector slice_ends;
      slice_ends.reserve(kernel_rank);

      TensorShapeVector slice_axes;
      slice_axes.reserve(kernel_rank);

      constexpr size_t spatial_dim_start = 2;
      const size_t spatial_dim_end = spatial_dim_start + kernel_rank;
      TensorShape spatial_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);

      TensorShapeVector y_dims_with_adjusted_pads(y_dims);
      ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShapeWithAdjustedPads(spatial_shape, kernel_shape,
                                                                       strides, dilations, pads, y_dims,
                                                                       y_dims_with_adjusted_pads, post_slicing_required,
                                                                       slice_starts, slice_ends, slice_axes));

      ORT_ENFORCE(y_dims.size() == y_dims_with_adjusted_pads.size());
      s_.y_dims = gsl::make_span(y_dims);
      s_.y_dims_with_adjusted_pads = y_dims_with_adjusted_pads;
      s_.post_slicing_required = post_slicing_required;
      s_.slice_starts = slice_starts;
      s_.slice_ends = slice_ends;
      s_.slice_axes = slice_axes;

      s_.Y = context->Output(0, TensorShape(s_.y_dims));
      if (post_slicing_required) {
        // Post slicing needed. Create and fill in the Conv results in an intermediate buffer.
        s_.memory_for_cudnn_conv_results =
            GetScratchBuffer<void>(TensorShape(y_dims_with_adjusted_pads).Size() * s_.element_size,
                                   context->GetComputeStream());
        s_.y_data = reinterpret_cast<CudaT*>(s_.memory_for_cudnn_conv_results.get());
      } else {
        // No post slicing needed. Fill the output tensor's buffer directly.
        s_.y_data = reinterpret_cast<CudaT*>(s_.Y->MutableData<T>());
      }

      const CUDAExecutionProvider* cuda_ep =
          static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());

      TensorShapeVector x_dims_cudnn{x_dims.begin(), x_dims.end()};
      TensorShapeVector y_dims_cudnn = !post_slicing_required ? y_dims : y_dims_with_adjusted_pads;
      if (kernel_rank < 2) {
        // TODO: Explore padding the provided input shape [N, C, D] to [N, C, 1, D]
        // especially for EXHAUSTIVE algo search which may result in a better algo selection.
        // ORTModule uses different algo search options (HEURISTIC, and use max workspace size) compared to
        // inference build (EXHAUSTIVE, 32M workspace size). We observed better perf when we pad input shape
        // [N,C,D] to [N,C,1,D], expecially on A100, and especially for ConvGrad.
        // PyTorch also pads to [N,C,1,D]. For inference build, we still pad it to [N, C, D, 1] as this seems
        // to be the sweet spot for all algo search options: EXHAUSTIVE, HEURISTIC, and DEFAULT.
        // See PR #7348 and #7702 for more context.
        if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
          x_dims_cudnn.insert(x_dims_cudnn.begin() + 2, 1);
          y_dims_cudnn.insert(y_dims_cudnn.begin() + 2, 1);
          w_dims.insert(w_dims.begin() + 2, 1);
          pads.insert(pads.begin() + kernel_rank, 0);
          pads.insert(pads.begin(), 0);
          kernel_shape.insert(kernel_shape.begin(), 1);
          strides.insert(strides.begin(), 1);
          dilations.insert(dilations.begin(), 1);
        } else {
          x_dims_cudnn.push_back(1);
          y_dims_cudnn.push_back(1);
          w_dims.push_back(1);
          pads.insert(pads.begin() + kernel_rank, 0);
          pads.insert(pads.end(), 0);
          kernel_shape.push_back(1);
          strides.push_back(1);
          dilations.push_back(1);
        }
      }

      if (w_dims_changed) {
        ORT_RETURN_IF_ERROR(s_.w_desc.Set(w_dims, ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>()));
      }

      // We must delay returning early until here so that the weight dims have been cached properly
      if (s_.Y->Shape().Size() == 0) {
        return Status::OK();
      }

      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims_cudnn, ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>()));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims_cudnn, ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>()));

      ORT_RETURN_IF_ERROR(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                           gsl::narrow_cast<int>(conv_attrs_.group), CUDNN_CROSS_CORRELATION,
                                           ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>(), UseTF32()));

      if (context->InputCount() >= 3) {
        const Tensor* B = context->Input<Tensor>(2);
        const auto& b_shape = B->Shape();
        ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
        TensorShapeVector b_dims(2 + kernel_shape.size(), 1);
        b_dims[1] = b_shape[0];
        ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>()));
        // s_.b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
      } else if (bias_expected) {
        TensorShapeVector b_dims(2 + kernel_shape.size(), 1);
        b_dims[1] = w_dims[0];
        auto malloc_size = b_dims[1] * sizeof(CudaT);
        ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, ::onnxruntime::cuda::CudnnTensor::GetDataType<CudaT>()));
        if (s_.b_zero) {
          CUDA_CALL_THROW(cudaFree(s_.b_zero));
          s_.b_zero = nullptr;
        }
        CUDA_CALL_THROW(cudaMalloc(&s_.b_zero, malloc_size));
        CUDA_CALL_THROW(cudaMemsetAsync(s_.b_zero, 0, malloc_size, Stream(context)));
      }

      if (!s_.cached_benchmark_results.contains(x_dims_cudnn)) {
        // set math type to tensor core before algorithm search
        if constexpr (std::is_same<T, MLFloat16>::value) {
          CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_TENSOR_OP_MATH));
        } else if constexpr (std::is_same<T, float>::value) {
          if (!UseTF32()) {
            CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(s_.conv_desc, CUDNN_FMA_MATH));
          }
        }

        cudnnConvolutionFwdAlgoPerf_t perf;
        int algo_count = 1;
        int cudnn_conv_algo = cuda_ep->GetCudnnConvAlgo();
        ORT_ENFORCE(cudnn_conv_algo > -1 && cudnn_conv_algo < 3, "cudnn_conv_algo should be 0, 1 or 2, but got ",
                    cudnn_conv_algo);
        switch (cudnn_conv_algo) {
          case 0: {
            static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
            size_t max_ws_size = cuda_ep->GetCudnnConvUseMaxWorkspace() ? GetMaxWorkspaceSize(GetCudnnHandle(context),
                                                                                              s_, kAllAlgos, num_algos)
                                                                        : ::onnxruntime::cuda::AlgoSearchWorkspaceSize;
            // Use GetTransientScratchBuffer() so the workspace can be freed instead of cached.
            // Because the benchmarking uses a huge amount of memory, e.g. a few GBs.
            IAllocatorUniquePtr<void> algo_search_workspace = GetTransientScratchBuffer<void>(max_ws_size);
            CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithmEx(
                GetCudnnHandle(context),
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
                max_ws_size));
            break;
          }
          case 1:
            CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
                GetCudnnHandle(context),
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
            CUDNN_RETURN_IF_ERROR(GetWorkspaceSize(GetCudnnHandle(context), s_, perf.algo, &perf.memory));

            if constexpr (std::is_same<T, MLFloat16>::value) {
              perf.mathType = CUDNN_TENSOR_OP_MATH;
            } else if (std::is_same<T, float>::value && !UseTF32()) {
              perf.mathType = CUDNN_FMA_MATH;
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
      // set Y
      s_.Y = context->Output(0, s_.y_dims);
      if (s_.Y->Shape().Size() == 0) {
        return Status::OK();
      }
      if (s_.post_slicing_required) {
        s_.memory_for_cudnn_conv_results = GetScratchBuffer<void>(
            TensorShape(s_.y_dims_with_adjusted_pads).Size() * s_.element_size, context->GetComputeStream());
        s_.y_data = reinterpret_cast<CudaT*>(s_.memory_for_cudnn_conv_results.get());
      } else {
        s_.y_data = reinterpret_cast<CudaT*>(s_.Y->MutableData<T>());
      }
    }
    return Status::OK();
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    std::lock_guard<std::mutex> lock(s_.mutex);
    auto cudnnHandle = this->GetCudnnHandle(context);
    ORT_RETURN_IF_ERROR(UpdateState(context, true));
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    bool has_z = nullptr != s_.z_data;
    bool has_b = nullptr != s_.b_data;
    const auto alpha = onnxruntime::cuda::Consts<CudaT>::One;
    const auto beta = onnxruntime::cuda::Consts<CudaT>::Zero;
    IAllocatorUniquePtr<void> workspace = GetWorkSpace(context->GetComputeStream());
    auto cudnn_status = cudnnConvolutionBiasActivationForward(cudnnHandle,
                                                              &alpha,
                                                              s_.x_tensor,
                                                              s_.x_data,
                                                              s_.w_desc,
                                                              s_.w_data,
                                                              s_.conv_desc,
                                                              s_.algo,
                                                              workspace.get(),
                                                              s_.workspace_bytes,
                                                              has_z ? &alpha : &beta,
                                                              has_z ? s_.z_tensor : s_.y_tensor,
                                                              has_z ? s_.z_data : s_.y_data,
                                                              s_.b_tensor,
                                                              has_b ? s_.b_data : s_.b_zero,
                                                              activation_desc_,
                                                              s_.y_tensor,
                                                              s_.y_data);
    if (CUDNN_STATUS_SUCCESS != cudnn_status) {
      CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(cudnnHandle,
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
      if (has_b) {
        CUDNN_RETURN_IF_ERROR(cudnnAddTensor(cudnnHandle, &alpha, s_.b_tensor, s_.b_data,
                                             &alpha, s_.y_tensor, s_.y_data));
      }
      if (has_z) {
        CUDNN_RETURN_IF_ERROR(cudnnAddTensor(cudnnHandle, &alpha, s_.z_tensor, s_.z_data,
                                             &alpha, s_.y_tensor, s_.y_data));
      }
      CUDNN_RETURN_IF_ERROR(cudnnActivationForward(cudnnHandle, activation_desc_, &alpha, s_.y_tensor,
                                                   s_.y_data, &beta, s_.y_tensor, s_.y_data));
    }
    if (s_.post_slicing_required) {
      ORT_RETURN_IF_ERROR(SliceOutUnwantedOutputSection(
          this->Stream(context), s_.y_data, s_.y_dims_with_adjusted_pads, s_.Y->MutableDataRaw(),
          s_.y_dims.GetDims(), s_.slice_starts, s_.slice_ends, s_.slice_axes, s_.element_size));
    }
    return Status::OK();
  }

 private:
  Status MapMode(const std::string& activaton_mode) {
    if (activaton_mode == "Relu") {
      activation_mode_ = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU;
    } else {
      return ORT_MAKE_STATUS(
          StatusCategory::ONNXRUNTIME, StatusCode::INVALID_ARGUMENT,
          "unsupported conv activation mode \"", activaton_mode, "\"");
    }
    return Status::OK();
  }

  inline IAllocatorUniquePtr<void> GetWorkSpace(onnxruntime::Stream* stream) const {
    return GetScratchBuffer<void>(s_.workspace_bytes, stream);
  }

  ConvAttributes conv_attrs_;
  mutable ::onnxruntime::cuda::CudnnConvState<cudnnConvolutionFwdAlgoPerf_t> s_;
  constexpr static auto kDefaultConvAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  constexpr static cudnnConvolutionFwdAlgo_t kAllAlgos[] = {
      CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_FFT,
      CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
  };

  cudnnActivationMode_t activation_mode_;
  cudnnActivationDescriptor_t activation_desc_ = nullptr;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime