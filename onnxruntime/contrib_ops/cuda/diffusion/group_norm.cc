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

using namespace ONNX_NAMESPACE;

namespace {
template <typename T>
struct DispatchGroupNorm {
  Status operator()(cudaStream_t stream,
                    Tensor* output,
                    const Tensor* input,
                    const Tensor* gamma,
                    const Tensor* beta,
                    void* workspace,
                    float epsilon,
                    int batch_size,
                    int num_channels,
                    int height,
                    int width,
                    int num_groups,
                    bool use_swish_activation) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    return LaunchGroupNormKernel<CudaT>(
        stream,
        reinterpret_cast<CudaT*>(output->MutableData<T>()),
        reinterpret_cast<const CudaT*>(input->Data<T>()),
        gamma->Data<float>(),
        beta->Data<float>(),
        workspace,
        epsilon,
        batch_size,
        num_channels,
        height,
        width,
        num_groups,
        use_swish_activation);
  }
};

}  // namespace

GroupNorm::GroupNorm(const OpKernelInfo& op_info) : CudaKernel(op_info) {
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
}

Status GroupNorm::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* gamma = context->Input<Tensor>(1);
  const Tensor* beta = context->Input<Tensor>(2);
  Tensor* output = context->Output(0, input->Shape());

  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 4 dimensions, got ", input_dims.size());
  }

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != input_dims[3]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of channels in gamma and input does not match");
  }

  const auto& beta_dims = beta->Shape().GetDims();
  if (beta_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have 1 dimension, got ", beta_dims.size());
  }
  if (beta_dims[0] != input_dims[3]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of channels in beta and input does not match");
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

  if (context->GetUseDeterministicCompute()) {
    static std::once_flag log_warning;
    std::call_once(log_warning, []() {
      LOGS_DEFAULT(WARNING) << "GroupNorm has no deterministic CUDA kernel, its outputs may still be nondeterministic.";
    });
  }

  auto workspace = GetScratchBuffer<void>(GetGroupNormWorkspaceSizeInBytes(), context->GetComputeStream());

  utils::MLTypeCallDispatcher<GROUP_NORM_TYPES> dispatcher(input->GetElementType());
  return dispatcher.InvokeRet<Status, DispatchGroupNorm>(Stream(context), output, input, gamma, beta, workspace.get(),
                                                         epsilon_,
                                                         batch_size,
                                                         num_channels,
                                                         height,
                                                         width,
                                                         num_groups_,
                                                         use_swish_activation_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
