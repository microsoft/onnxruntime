// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/paged_attention.h"

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// v1 registers only against T = float16, S = int32. See
// docs/design/webgpu_paged_attention.md §3 for the schema surface and
// §5 for the phased delivery plan.
ONNX_OPERATOR_KERNEL_EX(
    PagedAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()),
    PagedAttention);

PagedAttention::PagedAttention(const OpKernelInfo& info) : WebGpuKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0,
              "num_heads must be provided and > 0.");
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() &&
                  kv_num_heads > 0 && num_heads % kv_num_heads == 0,
              "kv_num_heads must be provided, > 0, and evenly divide num_heads.");
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
}

Status PagedAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& /*context*/) const {
  // v1 skeleton — implementation lands in Phase 1 per
  // docs/design/webgpu_paged_attention.md §5. Registering the op at all
  // gives us a specific error message instead of "no kernel found."
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "PagedAttention is not yet implemented on the WebGPU EP. "
                         "See docs/design/webgpu_paged_attention.md.");
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
