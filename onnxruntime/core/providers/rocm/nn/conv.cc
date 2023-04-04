// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/nn/conv.h"
#include "core/common/span_utils.h"
#include "core/providers/rocm/nn/conv_impl.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tensor/slice.h"

namespace onnxruntime {
namespace rocm {

// Op Set 11 for Conv only update document to clearify default dilations and strides value.
// which are already convered by op set 11 cpu versoin, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, false>);                                                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Conv,                                                                                \
      kOnnxDomain,                                                                         \
      11,                                                                                  \
      T,                                                                                   \
      kRocmExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, false>);

REGISTER_KERNEL_TYPED(float)
// not yet supported in MIOpen
// REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T, bool NHWC>
const miopenConvFwdAlgorithm_t Conv<T, NHWC>::kAllAlgos[] = {
    miopenConvolutionFwdAlgoGEMM,
    miopenConvolutionFwdAlgoDirect,
    miopenConvolutionFwdAlgoFFT,
    miopenConvolutionFwdAlgoWinograd,
    miopenConvolutionFwdAlgoImplicitGEMM};

miopenStatus_t GetWorkspaceSize(miopenHandle_t handle, const MiopenConvState<miopenConvAlgoPerf_t>& s,
                                miopenConvFwdAlgorithm_t algo, size_t* sz) {
  return miopenConvolutionForwardGetWorkSpaceSize(handle, s.w_desc, s.x_tensor, s.conv_desc, s.y_tensor, sz);
}

size_t GetMaxWorkspaceSize(miopenHandle_t handle, const MiopenConvState<miopenConvAlgoPerf_t>& s,
                           const miopenConvFwdAlgorithm_t* algo, int n_algo) {
  // TODO: get maximum available size from memory arean
  size_t free, total;
  HIP_CALL_THROW(hipMemGetInfo(&free, &total));
  // Assuming 10% of fragmentation
  free = static_cast<size_t>(static_cast<double>(free) * 0.9);
  size_t max_ws_size = 0;
  for (int i = 0; i < n_algo; i++) {
    miopenStatus_t err;
    size_t sz;
    err = GetWorkspaceSize(handle, s, algo[i], &sz);
    if (miopenStatusSuccess != err || sz == 0 || sz < max_ws_size || sz > free) continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

Status SliceOutUnwantedOutputSection(hipStream_t stream,
                                     const void* input_data,
                                     const gsl::span<const int64_t>& input_dims,
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

  return SliceRocm::Impl(stream, input_data, input_dims, output_data, compute_metadata, element_size);
}

template <typename T, bool NHWC>
Status Conv<T, NHWC>::UpdateState(OpKernelContext* context, bool bias_expected) const {
  // set X
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto x_dims = x_shape.AsShapeVector();
  s_.x_data = reinterpret_cast<const HipT*>(X->Data<T>());
  s_.element_size = X->DataType()->Size();
  // set W
  const Tensor* W = context->Input<Tensor>(1);
  const TensorShape& w_shape = W->Shape();
  auto w_dims = w_shape.AsShapeVector();
  s_.w_data = reinterpret_cast<const HipT*>(W->Data<T>());

  // Make sure input and weight are 4D for NHWC since we set 4D descriptor for NHWC.
  constexpr bool channels_last = NHWC;
  if (channels_last && (x_shape.NumDimensions() != 4 || w_shape.NumDimensions() != 4)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of dimensions of X and W should be 4 for channels_last format (NHWC)");
  }

  // set B
  if (context->InputCount() >= 3) {
    const Tensor* B = context->Input<Tensor>(2);
    s_.b_data = reinterpret_cast<const HipT*>(B->Data<T>());
  } else {
    s_.b_data = nullptr;
  }
  // set Z
  if (context->InputCount() >= 4) {
    const Tensor* Z = context->Input<Tensor>(3);
    ORT_RETURN_IF_ERROR(s_.z_tensor.Set(Z->Shape().GetDims(), MiopenTensor::GetDataType<HipT>()));
    s_.z_data = reinterpret_cast<const HipT*>(Z->Data<T>());
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
      s_.cached_benchmark_fwd_results.clear();
    }

    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W->Shape(), channels_last, channels_last));

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape, channels_last));

    const size_t kernel_rank = kernel_shape.size();

    ConvAttributes::ConvPadVector pads(conv_attrs_.pads);
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
    if (channels_last) {
      y_dims.push_back(N);
    } else {
      y_dims.insert(y_dims.begin(), {N, M});
    }

    bool post_slicing_required = false;
    TensorShapeVector slice_starts;
    slice_starts.reserve(kernel_rank);

    TensorShapeVector slice_ends;
    slice_ends.reserve(kernel_rank);

    TensorShapeVector slice_axes;
    slice_axes.reserve(kernel_rank);

    const size_t spatial_dim_start = channels_last ? 1 : 2;
    const size_t spatial_dim_end = spatial_dim_start + kernel_rank;
    TensorShape spatial_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);

    TensorShapeVector y_dims_with_adjusted_pads(y_dims);
    ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShapeWithAdjustedPads(spatial_shape, kernel_shape,
                                                                     strides, dilations, pads, y_dims, y_dims_with_adjusted_pads,
                                                                     post_slicing_required, slice_starts, slice_ends, slice_axes,
                                                                     channels_last));

    if (channels_last) {
      y_dims.push_back(M);
      y_dims_with_adjusted_pads.push_back(M);
    }

    ORT_ENFORCE(y_dims.size() == y_dims_with_adjusted_pads.size());
    s_.y_dims = gsl::make_span(y_dims);
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
      s_.memory_for_miopen_conv_results = GetScratchBuffer<void>(TensorShape(y_dims_with_adjusted_pads).Size() * s_.element_size, context->GetComputeStream());
      s_.y_data = reinterpret_cast<HipT*>(s_.memory_for_miopen_conv_results.get());
    } else {
      // No post slicing needed. Fill the output tensor's buffer directly.
      s_.y_data = reinterpret_cast<HipT*>(s_.Y->MutableData<T>());
    }

    TensorShapeVector x_dims_miopen{x_dims.begin(), x_dims.end()};
    TensorShapeVector y_dims_miopen = !post_slicing_required ? y_dims : y_dims_with_adjusted_pads;
    if (kernel_rank < 2) {
      // TODO: Remove asym padding correction.
      x_dims_miopen.push_back(1);
      y_dims_miopen.push_back(1);
      w_dims.push_back(1);
      pads.insert(pads.begin() + kernel_rank, 0);
      pads.insert(pads.end(), 0);
      kernel_shape.push_back(1);
      strides.push_back(1);
      dilations.push_back(1);
    }

    if (w_dims_changed) {
      if (channels_last) {
        ORT_RETURN_IF_ERROR(s_.w_desc.Set(MiopenTensor::GetDataType<HipT>(),
                                          miopenTensorNHWC,
                                          w_dims[0],
                                          w_dims[3],
                                          w_dims[1],
                                          w_dims[2]));
      } else {
        ORT_RETURN_IF_ERROR(s_.w_desc.Set(w_dims, MiopenTensor::GetDataType<HipT>()));
      }
    }

    if (channels_last) {
      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(MiopenTensor::GetDataType<HipT>(),
                                          miopenTensorNHWC,
                                          x_dims_miopen[0],
                                          x_dims_miopen[3],
                                          x_dims_miopen[1],
                                          x_dims_miopen[2]));

      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(MiopenTensor::GetDataType<HipT>(),
                                          miopenTensorNHWC,
                                          y_dims_miopen[0],
                                          y_dims_miopen[3],
                                          y_dims_miopen[1],
                                          y_dims_miopen[2]));
    } else {
      ORT_RETURN_IF_ERROR(s_.x_tensor.Set(x_dims_miopen, MiopenTensor::GetDataType<HipT>()));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims_miopen, MiopenTensor::GetDataType<HipT>()));
    }

    ORT_RETURN_IF_ERROR(s_.conv_desc.Set(kernel_shape.size(), pads, strides, dilations,
                                         gsl::narrow_cast<int>(conv_attrs_.group),
                                         miopenConvolution, MiopenTensor::GetDataType<HipT>()));

    if (context->InputCount() >= 3) {
      const Tensor* B = context->Input<Tensor>(2);
      const auto& b_shape = B->Shape();
      ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 1, "bias should be 1D");
      TensorShapeVector b_dims(2 + kernel_shape.size(), 1);
      b_dims[1] = b_shape[0];
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, MiopenTensor::GetDataType<HipT>()));
    } else if (bias_expected) {
      TensorShapeVector b_dims(2 + kernel_shape.size(), 1);
      b_dims[1] = w_dims[0];
      auto malloc_size = b_dims[1] * sizeof(HipT);
      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, MiopenTensor::GetDataType<HipT>()));
      if (s_.b_zero) {
        HIP_CALL_THROW(hipFree(s_.b_zero));
        s_.b_zero = nullptr;
      }
      HIP_CALL_THROW(hipMalloc(&s_.b_zero, malloc_size));
      HIP_CALL_THROW(hipMemsetAsync(s_.b_zero, 0, malloc_size, Stream(context)));
    }

    if (!s_.cached_benchmark_fwd_results.contains(x_dims_miopen)) {
      miopenConvAlgoPerf_t perf;
      int algo_count = 1;
      const ROCMExecutionProvider* rocm_ep = static_cast<const ROCMExecutionProvider*>(this->Info().GetExecutionProvider());
      static constexpr int num_algos = MIOPEN_CONVOLUTION_FWD_ALGO_COUNT;
      size_t max_ws_size = rocm_ep->GetMiopenConvUseMaxWorkspace() ? GetMaxWorkspaceSize(GetMiopenHandle(context), s_, kAllAlgos, num_algos)
                                                                   : AlgoSearchWorkspaceSize;
      IAllocatorUniquePtr<void> algo_search_workspace = GetTransientScratchBuffer<void>(max_ws_size);
      MIOPEN_RETURN_IF_ERROR(miopenFindConvolutionForwardAlgorithm(
          GetMiopenHandle(context),
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
          max_ws_size,
          false));  // Do not do exhaustive algo search.
      s_.cached_benchmark_fwd_results.insert(x_dims_miopen, {perf.fwd_algo, perf.memory});
    }
    const auto& perf = s_.cached_benchmark_fwd_results.at(x_dims_miopen);
    s_.fwd_algo = perf.fwd_algo;
    s_.workspace_bytes = perf.memory;
  } else {
    // set Y
    s_.Y = context->Output(0, TensorShape(s_.y_dims));
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    if (s_.post_slicing_required) {
      s_.memory_for_miopen_conv_results = GetScratchBuffer<void>(TensorShape(s_.y_dims_with_adjusted_pads).Size() * s_.element_size, context->GetComputeStream());
      s_.y_data = reinterpret_cast<HipT*>(s_.memory_for_miopen_conv_results.get());
    } else {
      s_.y_data = reinterpret_cast<HipT*>(s_.Y->MutableData<T>());
    }
  }
  return Status::OK();
}

template <typename T, bool NHWC>
Status Conv<T, NHWC>::ComputeInternal(OpKernelContext* context) const {
  std::lock_guard<OrtMutex> lock(s_.mutex);
  ORT_RETURN_IF_ERROR(UpdateState(context));
  if (s_.Y->Shape().Size() == 0) {
    return Status::OK();
  }
  const auto alpha = Consts<HipT>::One;
  const auto beta = Consts<HipT>::Zero;
  IAllocatorUniquePtr<void> workspace = GetWorkSpace(context->GetComputeStream());
  auto miopen_handle = GetMiopenHandle(context);
  MIOPEN_RETURN_IF_ERROR(miopenConvolutionForward(miopen_handle,
                                                  &alpha,
                                                  s_.x_tensor,
                                                  s_.x_data,
                                                  s_.w_desc,
                                                  s_.w_data,
                                                  s_.conv_desc,
                                                  s_.fwd_algo,
                                                  &beta,
                                                  s_.y_tensor,
                                                  s_.y_data,
                                                  workspace.get(),
                                                  s_.workspace_bytes));

  constexpr bool channels_last = NHWC;
  if (nullptr != s_.b_data && !channels_last) {
    MIOPEN_RETURN_IF_ERROR(miopenConvolutionForwardBias(miopen_handle, &alpha, s_.b_tensor, s_.b_data,
                                                        &beta, s_.y_tensor, s_.y_data));
  }
  // To deal with asymmetric padding, we may have over-padded on one or both sides of the spatial dimensions
  // This may have lead to extra results that are unnecessary and hence we slice that off here
  if (s_.post_slicing_required) {
    ORT_RETURN_IF_ERROR(SliceOutUnwantedOutputSection(Stream(context), s_.y_data, s_.y_dims_with_adjusted_pads,
                                                      s_.Y->MutableDataRaw(), s_.y_dims.GetDims(), s_.slice_starts,
                                                      s_.slice_ends, s_.slice_axes, s_.element_size));
  }
  if (nullptr != s_.b_data && channels_last) {
    const Tensor* B = context->Input<Tensor>(2);
    const auto& b_shape = B->Shape();

    ConvBiasImpl(Stream(context), reinterpret_cast<const HipT*>(s_.Y->MutableDataRaw()),
                 reinterpret_cast<const HipT*>(B->Data<T>()),
                 reinterpret_cast<HipT*>(s_.Y->MutableDataRaw()), b_shape[0], s_.Y->Shape().Size());
  }
  return Status::OK();
}

MiopenConvolutionDescriptor::MiopenConvolutionDescriptor() : desc_(nullptr) {
}

MiopenConvolutionDescriptor::~MiopenConvolutionDescriptor() {
  if (desc_ != nullptr) {
    miopenDestroyConvolutionDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status MiopenConvolutionDescriptor::Set(
    size_t rank,
    gsl::span<const int64_t> pads,
    gsl::span<const int64_t> strides,
    gsl::span<const int64_t> dilations,
    int groups,
    miopenConvolutionMode_t mode,
    miopenDataType_t data_type) {
  if (!desc_)
    MIOPEN_RETURN_IF_ERROR(miopenCreateConvolutionDescriptor(&desc_));

  InlinedVector<int> pad_dims(rank);
  InlinedVector<int> stride_dims(rank);
  InlinedVector<int> dilation_dims(rank);
  for (size_t i = 0; i < rank; i++) {
    pad_dims[i] = gsl::narrow_cast<int>(pads[i]);
    stride_dims[i] = gsl::narrow_cast<int>(strides[i]);
    dilation_dims[i] = gsl::narrow_cast<int>(dilations[i]);
  }

  MIOPEN_RETURN_IF_ERROR(miopenInitConvolutionNdDescriptor(
      desc_,
      gsl::narrow_cast<int>(rank),
      pad_dims.data(),
      stride_dims.data(),
      dilation_dims.data(),
      mode));

  MIOPEN_RETURN_IF_ERROR(miopenSetConvolutionGroupCount(desc_, groups));

  return Status::OK();
}

#ifndef DISABLE_CONTRIB_OPS
// template instantiation for NhwcConv
template class Conv<float, true>;
template class Conv<MLFloat16, true>;
#endif

}  // namespace rocm
}  // namespace onnxruntime
