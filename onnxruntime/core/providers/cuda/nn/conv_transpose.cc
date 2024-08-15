// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include <utility>

#include "conv_transpose.h"
#include "core/providers/cuda/tensor/transpose.h"

#if CUDNN_MAJOR < 9
// if compiled with cuDNN 8 we want to use the legacy cuDNN API
#include "conv_transpose_8.h"
#endif

// To suppress FP static analyzer warnings:
// https://msdata.visualstudio.com/Vienna/_workitems/edit/1944928 and
// https://msdata.visualstudio.com/Vienna/_workitems/edit/1944950
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 26110)
#pragma warning(disable : 26117)
#endif

namespace onnxruntime {
namespace cuda {

// Op Set 11 for ConvTranspose only update document to clarify default dilations and strides value.
// which are already covered by op set 11 cpu version, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T, DOMAIN, NHWC)                                                                       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                                           \
      ConvTranspose, DOMAIN, 1, 10, T, kCudaExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), ConvTranspose<T, NHWC>);  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(ConvTranspose, DOMAIN, 11, T, kCudaExecutionProvider,                                \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                ConvTranspose<T, NHWC>);

REGISTER_KERNEL_TYPED(float, kOnnxDomain, false)
REGISTER_KERNEL_TYPED(double, kOnnxDomain, false)
REGISTER_KERNEL_TYPED(MLFloat16, kOnnxDomain, false)

#ifdef ENABLE_CUDA_NHWC_OPS
REGISTER_KERNEL_TYPED(float, kMSInternalNHWCDomain, true)
REGISTER_KERNEL_TYPED(MLFloat16, kMSInternalNHWCDomain, true)
#endif

// First input (in this case X) is in case NHWC == true also in NHWC format, the other inputs in NCHW
template <typename T, bool NHWC>
Status ConvTranspose<T, NHWC>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed,
                                       [[maybe_unused]] PrePackedWeights* prepacked_weights) {
  is_packed = false;
  // only layout of weight input is adjusted via PrePack
  if constexpr (NHWC) {
    if (is_nhwc_domain_ && input_idx == 1) {  // InputTensors::IN_W
      // Transpose from {M, C/group, kH, kW} to {M, kH, kW, C/group}
      auto orig_shape = tensor.Shape();
      auto shape_size = orig_shape.GetDims().size();

      InlinedVector<size_t, 5> perm;
      perm.push_back(0);
      for (size_t i = 2; i < shape_size; i++) perm.push_back(i);
      perm.push_back(1);
      gsl::span<size_t> permutation(perm.data(), shape_size);

      TensorShapeVector nhwc_dims;
      for (size_t i = 0; i < shape_size; i++) {
        nhwc_dims.push_back(orig_shape[perm[i]]);
      }

      W_ = Tensor::Create(tensor.DataType(), TensorShape(nhwc_dims), std::move(alloc));

      auto status = cuda::Transpose::DoTranspose(GetDeviceProp(),
                                                 DefaultCudaStream(),
                                                 DefaultCublasHandle(),
                                                 permutation, tensor, *W_);
      if (!status.IsOK()) {
        return status;
      }
      CUDA_CALL_THROW(cudaStreamSynchronize(DefaultCudaStream()));
      is_packed = true;
    } else {
      W_already_nhwc = true;
    }
  } else {
    ORT_UNUSED_PARAMETER(tensor);
    ORT_UNUSED_PARAMETER(input_idx);
    ORT_UNUSED_PARAMETER(alloc);
    ORT_UNUSED_PARAMETER(prepacked_weights);
  }

  return Status::OK();
}

#if CUDNN_MAJOR >= 9
#if !defined(__CUDACC__)

template <typename T, bool Layout>
Status ConvTranspose<T, Layout>::CreateCudnnFeExecutionPlan(const onnxruntime::TensorShapeVector& x_dims,
                                                            const onnxruntime::TensorShapeVector& w_dims,
                                                            const Tensor* B,
                                                            const TensorShapeVector& y_dims,
                                                            cudnnContext* handle,
                                                            const cudnn_frontend::HeurMode_t heur_mode,
                                                            const std::vector<int64_t>& pads,
                                                            const std::vector<int64_t>& strides,
                                                            const std::vector<int64_t>& dilations,
                                                            const bool fuse_bias,
                                                            const bool fuse_act,
                                                            const bool w_in_nhwc,
                                                            const bool use_tf32) const {
  s_.bias_fused = fuse_bias;
  s_.act_fused = fuse_act;
  s_.variant_pack.clear();  // clear variant pack, as stored pointers to tensors change
  s_.cudnn_fe_graph = std::make_unique<cudnn_frontend::graph::Graph>();
  cudnn_frontend::DataType_t data_type = CudnnFeTensor::GetDataType<CudaT>();
  s_.cudnn_fe_graph->set_io_data_type(data_type).set_intermediate_data_type(data_type);
  if (data_type == cudnn_frontend::DataType_t::HALF) {
    s_.cudnn_fe_graph->set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  } else {
    s_.cudnn_fe_graph->set_compute_data_type(data_type);
  }

  s_.cudnn_fe_X = s_.cudnn_fe_graph->tensor(CudnnFeTensor(x_dims, "x", data_type, Layout == LAYOUT_NHWC).Get());
  s_.cudnn_fe_W = s_.cudnn_fe_graph->tensor(CudnnFeTensor(w_dims, "w", data_type, w_in_nhwc).Get());

  auto conv_options = cudnn_frontend::graph::Conv_dgrad_attributes()
                          .set_pre_padding(std::vector<int64_t>(pads.begin(),
                                                                pads.begin() + pads.size() / 2))
                          .set_post_padding(std::vector<int64_t>(pads.begin() + pads.size() / 2, pads.end()))
                          .set_stride(strides)
                          .set_dilation(dilations);
  s_.cudnn_fe_conv_Y = s_.cudnn_fe_graph->conv_dgrad(s_.cudnn_fe_X, s_.cudnn_fe_W, conv_options);
  auto cudnn_fe_y_tensor = CudnnFeTensor(y_dims, "y", data_type, Layout == LAYOUT_NHWC).Get();

  if (B == nullptr) {
    s_.cudnn_fe_Y = s_.cudnn_fe_conv_Y;
  } else {
    int64_t bias_size;
    if (B != nullptr) {
      bias_size = B->Shape()[0];
    } else {
      bias_size = w_dims[0];
    }

    if (fuse_bias) {
      onnxruntime::TensorShapeVector b_dims;
      for (size_t i = 0; i < x_dims.size(); i++) {
        b_dims.push_back(i == 1 ? bias_size : 1);
      }
      auto bias_tensor = CudnnFeTensor(b_dims, "b", data_type, Layout == LAYOUT_NHWC).Get();
      auto bias_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::ADD);
      s_.cudnn_fe_B = s_.cudnn_fe_graph->tensor(bias_tensor);
      s_.cudnn_fe_Y = s_.cudnn_fe_graph->pointwise(s_.cudnn_fe_conv_Y, s_.cudnn_fe_B, bias_options);
    } else {
      s_.cudnn_fe_Y = s_.cudnn_fe_conv_Y;

      TensorShapeVector b_dims(y_dims.size(), 1);
      TensorShapeVector b_strides(y_dims.size(), 1);
      b_dims[1] = bias_size;
      b_strides[0] = bias_size;

      ORT_RETURN_IF_ERROR(s_.b_tensor.Set(b_dims, CudnnTensor::GetDataType<CudaT>(), b_strides));
      ORT_RETURN_IF_ERROR(s_.y_tensor.Set(y_dims, CudnnTensor::GetDataType<CudaT>(), cudnn_fe_y_tensor.get_stride()));

      /* Creating an own CUDNN Frontend graph for the bias addition.
      s_.cudnn_fe_bias_graph = std::make_unique<cudnn_frontend::graph::Graph>();
      s_.cudnn_fe_bias_graph->set_io_data_type(data_type)
          .set_compute_data_type(data_type == cudnn_frontend::DataType_t::HALF ?
                                              cudnn_frontend::DataType_t::FLOAT : data_type)
          .set_intermediate_data_type(data_type);
      s_.cudnn_fe_bias_X = s_.cudnn_fe_bias_graph->tensor(CudnnFeTensor<NHWC>(y_dims, "x", data_type).Get());

      s_.cudnn_fe_B = s_.cudnn_fe_bias_graph->tensor(bias_tensor);
      s_.cudnn_fe_bias_Y = s_.cudnn_fe_bias_graph->pointwise(s_.cudnn_fe_bias_X, s_.cudnn_fe_B, bias_options);
      s_.cudnn_fe_bias_Y->set_output(true);

      CUDNN_FE_RETURN_IF_ERROR(s_.cudnn_fe_bias_graph->validate());
      CUDNN_FE_RETURN_IF_ERROR(s_.cudnn_fe_bias_graph->build_operation_graph(handle));
      CUDNN_FE_RETURN_IF_ERROR(s_.cudnn_fe_bias_graph->create_execution_plans({heur_mode}));
      CUDNN_FE_RETURN_IF_ERROR(s_.cudnn_fe_bias_graph->check_support(handle));
      CUDNN_FE_RETURN_IF_ERROR(s_.cudnn_fe_bias_graph->build_plans(handle));*/
    }
  }
  if (fuse_act && s_.cudnn_fe_act_attr.has_value()) {
    auto& activation_attr = s_.cudnn_fe_act_attr.value();
    s_.cudnn_fe_Y = s_.cudnn_fe_graph->pointwise(s_.cudnn_fe_Y, activation_attr);
  }

  s_.cudnn_fe_Y->set_dim(cudnn_fe_y_tensor.get_dim());
  s_.cudnn_fe_Y->set_stride(cudnn_fe_y_tensor.get_stride());
  s_.cudnn_fe_Y->set_output(true);

  try {
    CUDNN_FE_CALL_THROW(s_.cudnn_fe_graph->validate());
    CUDNN_FE_CALL_THROW(s_.cudnn_fe_graph->build_operation_graph(handle));
    CUDNN_FE_CALL_THROW(s_.cudnn_fe_graph->create_execution_plans({heur_mode}));
  } catch (const std::exception& ex) {
    std::string message = MakeString("Failed to initialize CUDNN Frontend", ex.what(),
                                     "with the cudnn frontend json:\n", s_.cudnn_fe_graph->print());
    return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::EP_FAIL, message);
  }

  if (!use_tf32) s_.cudnn_fe_graph->deselect_numeric_notes({cudnn_frontend::NumericalNote_t::TENSOR_CORE});

  try {
    CUDNN_FE_CALL_THROW(s_.cudnn_fe_graph->check_support(handle));
    CUDNN_FE_CALL_THROW(s_.cudnn_fe_graph->build_plans(handle));
  } catch (const std::exception& ex) {
    if (!fuse_bias && !fuse_act && use_tf32) {
      std::string message = MakeString("OP not supported by CUDNN Frontend", ex.what(),
                                       "with the cudnn frontend json:\n", s_.cudnn_fe_graph->print());
      return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::EP_FAIL, message);
    }

    // Try fallback.
    return CreateCudnnFeExecutionPlan(x_dims, w_dims, B, y_dims, handle, heur_mode,
                                      pads, strides, dilations, false, false, w_in_nhwc, true);
  }

  s_.workspace_bytes = s_.cudnn_fe_graph->get_workspace_size();
  return Status::OK();
}

#endif

template <typename T, bool Layout>
Status ConvTranspose<T, Layout>::UpdateState(OpKernelContext* context, bool dynamic_padding) const {
  constexpr bool channels_last = Layout == LAYOUT_NHWC;

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  bool has_bias = dynamic_padding ? num_inputs == 4 : num_inputs == 3;

  // set X
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  // X incl. x_dims is in NHWC Format iff. NHWC == true
  const auto x_dims = x_shape.AsShapeVector();

  s_.x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  s_.element_size = X->DataType()->Size();

  // set W
  bool w_in_nhwc;
  const Tensor* W;
  if (!W_) {
    W = context->Input<Tensor>(1);
    w_in_nhwc = false;
    // Dims and memory layout are in NCHW format
  } else {
    W = W_.get();
    w_in_nhwc = channels_last;
    // W got prepacked, therefore if NHWC == true, then dims and memory layout are in NHWC
  }
  const TensorShape& w_shape = W->Shape();
  onnxruntime::TensorShapeVector w_dims = w_shape.AsShapeVector();
  s_.w_data = reinterpret_cast<const CudaT*>(W->Data<T>());

  // set B
  // Always in NCHW format
  const Tensor* B = nullptr;
  if (has_bias) {
    B = context->Input<Tensor>(dynamic_padding ? 3 : 2);
    s_.b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
  } else {
    s_.b_data = nullptr;
  }

  const Tensor* Pads = dynamic_padding ? context->Input<Tensor>(2) : nullptr;

  bool input_dims_changed = (s_.last_x_dims != x_dims);
  bool w_dims_changed = (s_.last_w_dims != w_dims);
  if (input_dims_changed || w_dims_changed) {
    if (input_dims_changed)
      s_.last_x_dims = gsl::make_span(x_dims);

    if (w_dims_changed) {
      s_.last_w_dims = gsl::make_span(w_dims);
    }

    // The following code is from ConvTransposeAttributes::PrepareForCompute

    const int rank = static_cast<int>(X->Shape().NumDimensions());
    TensorShape input_shape = X->Shape().Slice(channels_last ? 1 : 2, channels_last ? rank - 1 : rank);
    const int64_t num_input_channels = channels_last ? X->Shape()[rank - 1] : X->Shape()[1];
    const int64_t N = X->Shape()[0];
    const int64_t num_output_channels_multiplier = w_in_nhwc ? w_shape[rank - 1] : w_shape[1];
    const int64_t num_output_channels = num_output_channels_multiplier * conv_transpose_attrs_.group;

    if (conv_transpose_attrs_.group <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "group count is <= 0",
                             " group: ", conv_transpose_attrs_.group);
    }

    if (X->Shape().NumDimensions() != w_shape.NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "X num_dims does not match W num_dims.",
                             " X: ", X->Shape().ToString().c_str(),
                             " W: ", w_shape.ToString().c_str());
    }

    if (w_shape[0] != num_input_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "filter number not equal to input channel number.",
                             " filter_number: ", w_shape[0],
                             " num_input_channels: ", num_input_channels);
    }

    // it looks like num_output_channels is really k*group similar to how in the conv case
    // num_input_channels is k*group. hence removing the check for num_output_channels here.

    if (num_input_channels % conv_transpose_attrs_.group != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input channels is not divisible by group.",
                             " num_input_channels: ", num_input_channels,
                             " group: ", conv_transpose_attrs_.group);
    }

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(conv_transpose_attrs_.ComputeKernelShape(w_shape, kernel_shape, w_in_nhwc));

    const size_t kernel_rank = kernel_shape.size();

    TensorShapeVector local_output_padding(conv_transpose_attrs_.output_padding);
    if (local_output_padding.empty()) {
      local_output_padding.resize(kernel_shape.size(), 0);
    }
    ConvPadVector pads;
    pads.reserve(2 * (input_shape.NumDimensions()));
    if (dynamic_padding) {
      for (int64_t i = 0; i < Pads->Shape().SizeFromDimension(0); ++i) {
        pads.push_back(Pads->Data<int64_t>()[i]);
      }
    } else {
      pads.assign(conv_transpose_attrs_.pads.begin(), conv_transpose_attrs_.pads.end());
    }
    if (pads.empty()) {
      pads.resize(kernel_shape.size() * 2, 0);
    }
    TensorShapeVector dilations(conv_transpose_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(kernel_shape.size(), 1);
    }
    TensorShapeVector strides(conv_transpose_attrs_.strides);
    if (strides.empty()) {
      strides.resize(kernel_shape.size(), 1);
    }

    TensorShapeVector y_dims;

    conv_transpose_attrs_.ComputePadsAndOutputShape(input_shape, num_output_channels, kernel_shape,
                                                    strides, dilations, local_output_padding, N, &pads, &y_dims, channels_last);
    TensorShape Yshape(y_dims);
    s_.Y = context->Output(0, Yshape);

    s_.y_data = reinterpret_cast<CudaT*>(s_.Y->MutableData<T>());
    const CUDAExecutionProvider* cuda_ep =
        static_cast<const CUDAExecutionProvider*>(this->Info().GetExecutionProvider());

    TensorShapeVector x_dims_cudnn{x_dims.begin(), x_dims.end()};
    TensorShapeVector y_dims_cudnn{y_dims.begin(), y_dims.end()};
    TensorShapeVector w_dims_cudnn{w_dims.begin(), w_dims.end()};

    if constexpr (channels_last) {
      x_dims_cudnn.insert(x_dims_cudnn.begin() + 1, *(x_dims_cudnn.end() - 1));
      y_dims_cudnn.insert(y_dims_cudnn.begin() + 1, *(y_dims_cudnn.end() - 1));
      x_dims_cudnn.erase(x_dims_cudnn.end() - 1);
      y_dims_cudnn.erase(y_dims_cudnn.end() - 1);

      if (w_in_nhwc) {
        w_dims_cudnn.insert(w_dims_cudnn.begin() + 1, *(w_dims_cudnn.end() - 1));
        w_dims_cudnn.erase(w_dims_cudnn.end() - 1);
      }
    }

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
        w_dims_cudnn.insert(w_dims_cudnn.begin() + 2, 1);
        pads.insert(pads.begin() + kernel_rank, 0);
        pads.insert(pads.begin(), 0);
        kernel_shape.insert(kernel_shape.begin(), 1);
        strides.insert(strides.begin(), 1);
        dilations.insert(dilations.begin(), 1);
      } else {
        x_dims_cudnn.push_back(1);
        y_dims_cudnn.push_back(1);
        w_dims_cudnn.push_back(1);
        pads.insert(pads.begin() + kernel_rank, 0);
        pads.insert(pads.end(), 0);
        kernel_shape.push_back(1);
        strides.push_back(1);
        dilations.push_back(1);
      }
    }

    // We must delay returning early until here so that the weight dims have been cached properly
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }

    auto handle = GetCudnnHandle(context);

    int cudnn_conv_algo = cuda_ep->GetCudnnConvAlgo();
#if !defined(__CUDACC__)
    cudnn_frontend::HeurMode_t heur_mode;
    switch (cudnn_conv_algo) {
      case 0:
        heur_mode = cudnn_frontend::HeurMode_t::B;
        break;
      case 1:
        heur_mode = cudnn_frontend::HeurMode_t::A;
        break;
      case 2:
        heur_mode = cudnn_frontend::HeurMode_t::FALLBACK;
        break;
      default:
        heur_mode = cudnn_frontend::HeurMode_t::A;
        break;
    }

    auto use_tf32 = cuda_ep->UseTF32();
    const auto fuse_bias = cuda_ep->IsFuseConvBias() || is_fused_node_;
    const auto fuse_act = is_fused_node_;

    ORT_RETURN_IF_ERROR(CreateCudnnFeExecutionPlan(x_dims_cudnn, w_dims_cudnn, B, y_dims_cudnn, handle, heur_mode,
                                                   std::vector<int64_t>(pads.begin(),
                                                                        pads.end()),
                                                   std::vector<int64_t>(strides.begin(),
                                                                        strides.end()),
                                                   std::vector<int64_t>(dilations.begin(),
                                                                        dilations.end()),
                                                   fuse_bias, fuse_act, w_in_nhwc, use_tf32));
#endif
  } else {
    // set Y
    s_.Y = context->Output(0, s_.y_dims);
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    s_.y_data = reinterpret_cast<CudaT*>(s_.Y->MutableData<T>());
  }
  return Status::OK();
}

template <typename T, bool Layout>
Status ConvTranspose<T, Layout>::DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const {
  std::lock_guard<OrtMutex> lock(s_.mutex);
  ORT_RETURN_IF_ERROR(UpdateState(context, dynamic_padding));
  if (s_.Y->Shape().Size() == 0) {
    return Status::OK();
  }
  const auto alpha = onnxruntime::cuda::Consts<CudaT>::One;
  auto cudnn_handle = GetCudnnHandle(context);
#if !defined(__CUDACC__)
  s_.variant_pack.insert_or_assign(s_.cudnn_fe_X, const_cast<void*>(s_.x_data));
  s_.variant_pack.insert_or_assign(s_.cudnn_fe_W, const_cast<void*>(s_.w_data));
  s_.variant_pack.insert_or_assign(s_.cudnn_fe_Y, s_.y_data);
  if (s_.bias_fused && s_.b_data != nullptr) {
    s_.variant_pack.insert_or_assign(s_.cudnn_fe_B, const_cast<void*>(s_.b_data));
  }
  if (s_.bias_fused && s_.z_data != nullptr) {
    s_.variant_pack.insert_or_assign(s_.cudnn_fe_Z, const_cast<void*>(s_.z_data));
    if (Layout == LAYOUT_NCHW && s_.z_data == s_.y_data) {
      // memset Z if it's required for a succesful fusion
      CUDA_RETURN_IF_ERROR(cudaMemset(s_.y_data, 0, s_.Y->SizeInBytes()));
    }
  }
  auto ws = GetWorkSpace(context->GetComputeStream());

  CUDNN_FE_RETURN_IF_ERROR(s_.cudnn_fe_graph->execute(cudnn_handle,
                                                      s_.variant_pack,
                                                      ws.get()));

  if (!s_.bias_fused && s_.z_data != nullptr) {
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(cudnn_handle, &alpha, s_.z_tensor, s_.z_data,
                                         &alpha, s_.y_tensor, s_.y_data));
  }
  if (!s_.bias_fused && s_.b_data != nullptr) {
    CUDNN_RETURN_IF_ERROR(cudnnAddTensor(cudnn_handle, &alpha, s_.b_tensor, s_.b_data,
                                         &alpha, s_.y_tensor, s_.y_data));

    /* For the standalone bias addition graph.
    s_.variant_pack_bias.insert_or_assign(s_.cudnn_fe_bias_X, s_.y_data);
    s_.variant_pack_bias.insert_or_assign(s_.cudnn_fe_bias_Y, s_.y_data);
    s_.variant_pack_bias.insert_or_assign(s_.cudnn_fe_B, const_cast<void*>(s_.b_data));
    CUDNN_FE_RETURN_IF_ERROR(s_.cudnn_fe_bias_graph->execute(cudnn_handle,
                                                             s_.variant_pack_bias,
                                                             GetWorkSpace(context->GetComputeStream()).get()));*/
  }
#endif

  return Status::OK();
}
#endif

template <typename T, bool Layout>
Status ConvTranspose<T, Layout>::ComputeInternal(OpKernelContext* context) const {
  return DoConvTranspose(context, false);
}

}  // namespace cuda
}  // namespace onnxruntime

#ifdef _WIN32
#pragma warning(pop)
#endif
