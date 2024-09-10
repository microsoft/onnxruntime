// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, Attention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, BiasAdd);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, BiasSplitGelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, FastGelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, FusedConv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, Gelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, GroupQueryAttention);
// LayerNormalization used to be a contrib op that (incorrectly) used kOnnxDomain so we need to version it
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 16, LayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MatMulNBits);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MultiHeadAttention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, QuickGelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, RotaryEmbedding);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, SkipLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, SimplifiedLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, SkipSimplifiedLayerNormalization);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterWebGpuContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
                                    // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, Attention)>,
                                    // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, BiasAdd)>,
                                    // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, BiasSplitGelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, FastGelu)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, FusedConv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, Gelu)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, GroupQueryAttention)>,
      // // LayerNormalization used to be a contrib op that (incorrectly) used kOnnxDomain so we need to version it
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 16, LayerNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MatMulNBits)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, MultiHeadAttention)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, QuickGelu)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1, RotaryEmbedding)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1,
      //                                                       SkipLayerNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1,
      //                                                       SimplifiedLayerNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSDomain, 1,
      //                                                       SkipSimplifiedLayerNormalization)>
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
