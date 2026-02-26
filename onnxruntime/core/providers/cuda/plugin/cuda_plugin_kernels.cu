// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_plugin_kernels.h"
#include "cuda_stream_plugin.h"
#include "cuda_kernel_adapter.h"
#include "core/common/narrow.h"
#include "core/providers/cuda/activation/activations.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/clip.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/math/unary_elementwise_ops.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/tensor/concat.h"
#include "core/providers/cuda/tensor/cast_op.h"
#include "core/providers/cuda/tensor/gather.h"
#include "core/providers/cuda/tensor/split.h"
#include "core/providers/cuda/tensor/where.h"
#include "contrib_ops/cuda/bert/decoder_masked_multihead_attention.h"
#include "contrib_ops/cuda/bert/embed_layer_norm.h"
#include "contrib_ops/cuda/bert/fast_gelu.h"
#include "contrib_ops/cuda/bert/gemma_rotary_emb.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cuda/bert/multihead_attention.h"
#include "contrib_ops/cuda/bert/rotary_embedding.h"
#include "contrib_ops/cuda/bert/skip_layer_norm.h"
#include "contrib_ops/cuda/bert/attention.h"
#include "contrib_ops/cuda/moe/moe.h"
#include "contrib_ops/cuda/quantization/gather_block_quantized.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.h"
#include "contrib_ops/cuda/quantization/moe_quantization.h"

#include <cstring>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Legacy plugin path helpers (not used in adapter mode)
// ---------------------------------------------------------------------------

#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
namespace {

CudaSyncStream* GetCudaSyncStream(const Ort::KernelContext& ctx) {
  void* stream = ctx.GetGPUComputeStream();
  if (!stream) return nullptr;
  return CudaSyncStream::FromCudaStream(static_cast<cudaStream_t>(stream));
}

}  // namespace

// Generic Adapter Kernel — wraps any cuda::CudaKernel-derived class.
// The adapter path uses the AdapterKernelImpl and GenericCreateKernel
// in cuda_kernel_adapter.h instead.

template <typename KernelT>
struct AdapterKernelImpl : public OrtKernelImpl {
  std::unique_ptr<KernelT> kernel;

  explicit AdapterKernelImpl(const OrtKernelInfo* info) : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;

    const auto& adapter_info = *reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
    kernel = std::make_unique<KernelT>(adapter_info);
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept {
    EXCEPTION_TO_STATUS_BEGIN

    auto* self = static_cast<AdapterKernelImpl*>(this_ptr);
    auto* adapter_ctx = reinterpret_cast<onnxruntime::OpKernelContext*>(context);
    Status status = self->kernel->Compute(adapter_ctx);
    if (!status.IsOK()) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, status.ErrorMessage().c_str());
    }
    return nullptr;

    EXCEPTION_TO_STATUS_END
  }

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<AdapterKernelImpl*>(this_ptr);
  }
};
#endif  // !ORT_CUDA_PLUGIN_USE_ADAPTER

OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* create_kernel_state,
                                    OrtKernelRegistry** out_registry) {
  return CreateCudaKernelRegistryFromOrtTables(ep_api, ep_name, create_kernel_state, out_registry);
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
