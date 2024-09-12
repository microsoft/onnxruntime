// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include <utility>
#include <algorithm>
#include <vector>

#include "core/common/status.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/common/span_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/shared_inc/cudnn_fe_call.h"

#if CUDNN_MAJOR < 9
// if compiled with cuDNN 8 we want to use the legacy cuDNN API
#include "conv_8.h"
#endif
namespace onnxruntime {
namespace cuda {

// Op Set 11 for Conv only update document to clearify default dilations and strides value.
// which are already convered by op set 11 cpu version, so simply add declaration.
#define REGISTER_KERNEL_TYPED(T, DOMAIN, NHWC)                                             \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Conv,                                                                                \
      DOMAIN,                                                                              \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, NHWC>);                                                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Conv,                                                                                \
      DOMAIN,                                                                              \
      11,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, NHWC>);

REGISTER_KERNEL_TYPED(float, kOnnxDomain, false)
REGISTER_KERNEL_TYPED(double, kOnnxDomain, false)
REGISTER_KERNEL_TYPED(MLFloat16, kOnnxDomain, false)

#ifdef ENABLE_CUDA_NHWC_OPS
REGISTER_KERNEL_TYPED(float, kMSInternalNHWCDomain, true)
REGISTER_KERNEL_TYPED(MLFloat16, kMSInternalNHWCDomain, true)
#endif

// First input (in this case X) is in case NHWC == true also in NHWC format, the other inputs in NCHW
template <typename T, bool NHWC>
Status Conv<T, NHWC>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                              bool& is_packed, PrePackedWeights* /*prepacked_weights*/) {
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
  }

  return Status::OK();
}

#if CUDNN_MAJOR >= 9
#if !defined(__CUDACC__)

template <typename T, bool Layout>
Status Conv<T, Layout>::CreateCudnnFeExecutionPlan(const onnxruntime::TensorShapeVector& x_dims,
                                                   const onnxruntime::TensorShapeVector& w_dims,
                                                   const Tensor* B,
                                                   const Tensor* Z,
                                                   const TensorShapeVector& y_dims,
                                                   cudnnContext* handle,
                                                   const cudnn_frontend::HeurMode_t heur_mode,
                                                   const std::vector<int64_t>& pads,
                                                   const std::vector<int64_t>& strides,
                                                   const std::vector<int64_t>& dilations,
                                                   const bool bias_expected,
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

  auto conv_options = cudnn_frontend::graph::Conv_fprop_attributes()
                          .set_pre_padding(std::vector<int64_t>(pads.begin(),
                                                                pads.begin() + pads.size() / 2))
                          .set_post_padding(std::vector<int64_t>(pads.begin() + pads.size() / 2, pads.end()))
                          .set_stride(strides)
                          .set_dilation(dilations);
  s_.cudnn_fe_conv_Y = s_.cudnn_fe_graph->conv_fprop(s_.cudnn_fe_X, s_.cudnn_fe_W, conv_options);
  auto cudnn_fe_y_tensor = CudnnFeTensor(y_dims, "y", data_type, Layout == LAYOUT_NHWC).Get();

  if (!bias_expected && B == nullptr) {
    s_.cudnn_fe_Y = s_.cudnn_fe_conv_Y;
  } else {
    int64_t bias_size;
    if (B != nullptr) {
      bias_size = B->Shape()[0];
    } else {
      bias_size = w_dims[0];
    }

    std::optional<cudnn_frontend::graph::Tensor_attributes> cudnn_fe_z_tensor;
    if (Z) {
      const auto& z_shape = Z->Shape().AsShapeVector();
      cudnn_fe_z_tensor = CudnnFeTensor(z_shape, "z", data_type, Layout == LAYOUT_NHWC).Get();
    } else if (fuse_bias && Layout == LAYOUT_NCHW) {
      // Z is required for NCHW precompiled kernels in cuDNN
      s_.z_data = s_.y_data;
      cudnn_fe_z_tensor = cudnn_fe_y_tensor;
    }

    if (fuse_bias) {
      std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> add_output;
      if (cudnn_fe_z_tensor.has_value()) {
        s_.cudnn_fe_Z = s_.cudnn_fe_graph->tensor(cudnn_fe_z_tensor.value());
        auto add_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::ADD);
        add_output = s_.cudnn_fe_graph->pointwise(s_.cudnn_fe_conv_Y, s_.cudnn_fe_Z, add_options);
      } else {
        add_output = s_.cudnn_fe_conv_Y;
      }

      onnxruntime::TensorShapeVector b_dims;
      for (size_t i = 0; i < x_dims.size(); i++) {
        b_dims.push_back(i == 1 ? bias_size : 1);
      }
      auto bias_tensor = CudnnFeTensor(b_dims, "b", data_type, Layout == LAYOUT_NHWC).Get();
      auto bias_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::ADD);
      s_.cudnn_fe_B = s_.cudnn_fe_graph->tensor(bias_tensor);
      s_.cudnn_fe_Y = s_.cudnn_fe_graph->pointwise(add_output, s_.cudnn_fe_B, bias_options);
    } else {
      s_.cudnn_fe_Y = s_.cudnn_fe_conv_Y;

      TensorShapeVector b_dims(y_dims.size(), 1);
      TensorShapeVector b_strides(y_dims.size(), 1);
      b_dims[1] = bias_size;
      b_strides[0] = bias_size;
      if (Z) {
        ORT_RETURN_IF_ERROR(s_.z_tensor.Set(Z->Shape().AsShapeVector(),
                                            CudnnTensor::GetDataType<CudaT>(),
                                            cudnn_fe_z_tensor->get_stride()));
      }
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
    return CreateCudnnFeExecutionPlan(x_dims, w_dims, B, Z, y_dims, handle, heur_mode,
                                      pads, strides, dilations, bias_expected, false, false, w_in_nhwc, true);
  }

  s_.workspace_bytes = s_.cudnn_fe_graph->get_workspace_size();
  return Status::OK();
}

#endif

template <typename T, bool Layout>
Status Conv<T, Layout>::UpdateState(OpKernelContext* context, bool bias_expected) const {
  constexpr bool channels_last = Layout;

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
    w_in_nhwc = W_already_nhwc;
    // Dims and memory layout are in NCHW format
  } else {
    W = W_.get();
    w_in_nhwc = true;
    // W got prepacked, therfore if NHWC == true, then dims and memory layout are in NHWC
  }
  const TensorShape& w_shape = W->Shape();
  onnxruntime::TensorShapeVector w_dims = w_shape.AsShapeVector();
  s_.w_data = reinterpret_cast<const CudaT*>(W->Data<T>());

  // set B
  // Always in NCHW format
  const Tensor* B = nullptr;
  if (context->InputCount() >= 3) {
    B = context->Input<Tensor>(2);
    s_.b_data = reinterpret_cast<const CudaT*>(B->Data<T>());
  } else {
    s_.b_data = nullptr;
  }

  // set Z
  const Tensor* Z = nullptr;
  if (context->InputCount() >= 4) {
    Z = context->Input<Tensor>(3);
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
    }

    ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W->Shape(), channels_last, w_in_nhwc));

    TensorShapeVector kernel_shape;
    ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape, w_in_nhwc));

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

    if constexpr (channels_last) {
      y_dims.push_back(N);
    } else {
      y_dims.insert(y_dims.begin(), {N, M});
    }

    constexpr size_t spatial_dim_start = channels_last ? 1 : 2;
    const size_t spatial_dim_end = spatial_dim_start + kernel_rank;
    TensorShape spatial_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);

    ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(spatial_shape, kernel_shape,
                                                            strides, dilations, pads, y_dims));
    if constexpr (channels_last) {
      y_dims.push_back(M);
    }

    s_.y_dims = gsl::make_span(y_dims);
    s_.Y = context->Output(0, TensorShape(s_.y_dims));

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
      // [N,C,D] to [N,C,1,D], especially on A100, and especially for ConvGrad.
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
        LOGS_DEFAULT(WARNING) << "OP " << CudaKernel::Node().OpType() << "(" << CudaKernel::Node().Name()
                              << ") running in Fallback mode. May be extremely slow.";
        break;
      default:
        heur_mode = cudnn_frontend::HeurMode_t::A;
        break;
    }

    const auto use_tf32 = cuda_ep->UseTF32();
    // fuse if this op is part of a FusedConv or if the EP is set to fuse ops
    const auto fuse_bias = cuda_ep->IsFuseConvBias() || is_fused_node_;
    const auto fuse_act = is_fused_node_;

    ORT_RETURN_IF_ERROR(CreateCudnnFeExecutionPlan(x_dims_cudnn, w_dims_cudnn, B, Z, y_dims_cudnn, handle, heur_mode,
                                                   std::vector<int64_t>(pads.begin(),
                                                                        pads.end()),
                                                   std::vector<int64_t>(strides.begin(),
                                                                        strides.end()),
                                                   std::vector<int64_t>(dilations.begin(),
                                                                        dilations.end()),
                                                   bias_expected, fuse_bias, fuse_act, w_in_nhwc, use_tf32));
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
Status Conv<T, Layout>::ComputeInternal(OpKernelContext* context) const {
  std::lock_guard<OrtMutex> lock(s_.mutex);
  ORT_RETURN_IF_ERROR(UpdateState(context));
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
    const gsl::span<const int64_t>& pads,
    const gsl::span<const int64_t>& strides,
    const gsl::span<const int64_t>& dilations,
    int groups,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t data_type,
    bool use_tf32) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateConvolutionDescriptor(&desc_));

  InlinedVector<int, kTensorShapeSmallBufferElementsSize> pad_dims(rank);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> stride_dims(rank);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> dilation_dims(rank);
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
  } else if (data_type == CUDNN_DATA_FLOAT && !use_tf32) {
    CUDNN_RETURN_IF_ERROR(cudnnSetConvolutionMathType(desc_, CUDNN_FMA_MATH));
  }

  return Status::OK();
}
#endif

#ifndef DISABLE_CONTRIB_OPS
// template instantiation for NhwcConv
template class Conv<float, true>;
template class Conv<MLFloat16, true>;
#endif

}  // namespace cuda
}  // namespace onnxruntime
