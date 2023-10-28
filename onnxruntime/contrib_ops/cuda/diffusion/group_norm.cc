// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/diffusion/group_norm.h"
#include "contrib_ops/cuda/diffusion/group_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define GROUP_NORM_TYPES float, MLFloat16

ONNX_OPERATOR_KERNEL_EX(
    GroupNorm, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<GROUP_NORM_TYPES>()), GroupNorm);

ONNX_OPERATOR_KERNEL_EX(
    SkipGroupNorm, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<GROUP_NORM_TYPES>()), GroupNorm);

using namespace ONNX_NAMESPACE;

namespace {

template <typename T>
struct DispatchGroupNorm {
  Status operator()(cudaStream_t stream,
                    Tensor* output,
                    Tensor* add_out,
                    const Tensor* input,
                    const Tensor* skip,
                    const Tensor* bias,
                    const Tensor* gamma,
                    const Tensor* beta,
                    void* workspace,
                    float epsilon,
                    int batch_size,
                    int num_channels,
                    int height,
                    int width,
                    int num_groups,
                    bool use_swish_activation,
                    bool broadcast_skip) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    return LaunchGroupNormKernel<CudaT>(
        stream,
        reinterpret_cast<CudaT*>(output->MutableData<T>()),
        add_out == nullptr ? nullptr : reinterpret_cast<CudaT*>(add_out->MutableData<T>()),
        reinterpret_cast<const CudaT*>(input->Data<T>()),
        skip == nullptr ? nullptr : reinterpret_cast<const CudaT*>(skip->Data<T>()),
        bias == nullptr ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>()),
        gamma->Data<float>(),
        beta->Data<float>(),
        workspace,
        epsilon,
        batch_size,
        num_channels,
        height,
        width,
        num_groups,
        use_swish_activation,
        broadcast_skip);
  }
};

}  // namespace

GroupNorm::GroupNorm(const OpKernelInfo& op_info) : CudaKernel(op_info) {
  has_skip_ = false;
  const std::string& op_name = op_info.GetKernelDef().OpName();
  if (op_name == "SkipGroupNorm") {
    has_skip_ = true;
  }

  epsilon_ = op_info.GetAttrOrDefault<float>("epsilon", 1e-5f);
  ORT_ENFORCE(epsilon_ >= 0);

  int64_t num_groups;
  ORT_ENFORCE(op_info.GetAttr("groups", &num_groups).IsOK());
  ORT_ENFORCE(num_groups >= 0);
  num_groups_ = static_cast<int>(num_groups);

  int64_t activation;
  ORT_ENFORCE(op_info.GetAttr("activation", &activation).IsOK());
  ORT_ENFORCE(activation == 0 || activation == 1);  // 0 is None, 1 is Swish
  use_swish_activation_ = (activation == 1);

  channels_last_ = (op_info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(1)) != 0);
}

Status GroupNorm::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* gamma = context->Input<Tensor>(1);
  const Tensor* beta = context->Input<Tensor>(2);
  Tensor* output = context->Output(0, input->Shape());

  if (!channels_last_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "only the channels_last layout is supported");
  }

  if (!gamma->IsDataType<float>() || !beta->IsDataType<float>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "GroupNorm only supports gamma and beta in float type");
  }

  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 4 dimensions, got ", input_dims.size());
  }

  // Input and output format is NHWC
  int batch_size = static_cast<int>(input_dims[0]);
  int num_channels = static_cast<int>(input_dims[3]);
  int height = static_cast<int>(input_dims[1]);
  int width = static_cast<int>(input_dims[2]);

  if (num_channels % num_groups_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "number of channels should be divisiable by num_groups");
  }

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of channels in gamma and input does not match");
  }

  const auto& beta_dims = beta->Shape().GetDims();
  if (beta_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have 1 dimension, got ", beta_dims.size());
  }
  if (beta_dims[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of channels in beta and input does not match");
  }

  if (context->GetUseDeterministicCompute()) {
    static std::once_flag log_warning;
    std::call_once(log_warning, []() {
      LOGS_DEFAULT(WARNING) << "GroupNorm has no deterministic CUDA kernel, its outputs may still be nondeterministic.";
    });
  }

  const Tensor* skip = nullptr;
  const Tensor* bias = nullptr;
  Tensor* add_out = nullptr;

  bool broadcast_skip = false;
  if (has_skip_) {
    skip = context->Input<Tensor>(3);
    bias = context->Input<Tensor>(4);
    add_out = context->Output(1, input->Shape());

    if (bias != nullptr) {  // Bias is optional
      // If provided, bias has shape (C).
      const auto& bias_dims = bias->Shape().GetDims();
      if (bias_dims.size() != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "bias is expected to have 1 dimension, got ", bias_dims.size());
      }
      if (bias_dims[0] != num_channels) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Number of channels in bias and input does not match");
      }
    }

    // Check whether skip can be broadcasted to input shape.
    if (skip->Shape() != input->Shape()) {
      const auto& dims = skip->Shape().GetDims();
      // The shape of ship can be (N, C) or (N, 1, 1, C) for broadcast.
      const bool b2 = (dims.size() == 2 && dims[0] == batch_size && dims[1] == num_channels);
      const bool b4 = (dims.size() == 4 && dims[0] == batch_size &&
                       dims[1] == 1 && dims[2] == 1 && dims[3] == num_channels);
      broadcast_skip = b2 || b4;
      if (!broadcast_skip) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "skip shape is expected to be (N, H, W, C) or (N, 1, 1, C) or (N, C)");
      }
    }
  }

  auto workspace = GetScratchBuffer<void>(GetGroupNormWorkspaceSizeInBytes(batch_size, num_groups_),
                                          context->GetComputeStream());

  utils::MLTypeCallDispatcher<GROUP_NORM_TYPES> dispatcher(input->GetElementType());
  return dispatcher.InvokeRet<Status, DispatchGroupNorm>(Stream(context), output, add_out, input, skip, bias,
                                                         gamma, beta, workspace.get(),
                                                         epsilon_,
                                                         batch_size,
                                                         num_channels,
                                                         height,
                                                         width,
                                                         num_groups_,
                                                         use_swish_activation_,
                                                         broadcast_skip);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
