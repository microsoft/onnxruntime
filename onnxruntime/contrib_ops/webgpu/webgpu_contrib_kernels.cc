// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/causal_conv_with_state.h"
#include "contrib_ops/webgpu/bert/group_query_attention.h"
#include "contrib_ops/webgpu/bert/linear_attention.h"

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

static const BuildKernelCreateInfoFn build_kernel_create_info_function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, Attention)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, BiasAdd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, CausalConvWithState)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, BiasGelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, BiasSplitGelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, FastGelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, GatherBlockQuantized)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, FusedConv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, Gelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, LinearAttention)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MatMulNBits)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MultiHeadAttention)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, QuickGelu)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, RotaryEmbedding)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, SkipLayerNormalization)>,
    // LayerNormalization used to be a contrib op that (incorrectly) used kOnnxDomain so we need to version it
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 16, LayerNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, SimplifiedLayerNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, SkipSimplifiedLayerNormalization)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MoE)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, QMoE)>};

Status RegisterWebGpuContribKernels(KernelRegistry& kernel_registry, bool enable_graph_capture) {
  for (auto& function_table_entry : build_kernel_create_info_function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  // Register GroupQueryAttention with conditional InputMemoryType based on graph capture
  ORT_RETURN_IF_ERROR(kernel_registry.Register(CreateGroupQueryAttentionKernelInfo(enable_graph_capture)));

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
