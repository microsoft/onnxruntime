// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/js/js_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace js {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, Attention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, Gelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, MultiHeadAttention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, BiasSplitGelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, BiasAdd);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, SkipLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, FusedConv);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterJsContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, Attention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, Gelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, MultiHeadAttention)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, BiasAdd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, BiasSplitGelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1,
                                                            SkipLayerNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSDomain, 1, FusedConv)>};

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
