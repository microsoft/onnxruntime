
#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Initialize the singleton instance
AttentionKernelOptions AttentionKernelOptions::instance;

void AttentionKernelOptions::Initialize(int value) {
  if (value > 0) {
    use_flash_attention_ = (value & static_cast<int>(AttentionBackend::FLASH_ATTENTION)) > 0;
    use_efficient_attention_ = (value & static_cast<int>(AttentionBackend::EFFICIENT_ATTENTION)) > 0;
    use_trt_fused_attention_ = (value & static_cast<int>(AttentionBackend::TRT_FUSED_ATTENTION)) > 0;
    use_unfused_ = (value & static_cast<int>(AttentionBackend::MATH)) > 0;
    use_trt_flash_attention_ = (value & static_cast<int>(AttentionBackend::TRT_FLASH_ATTENTION)) > 0;
    use_trt_cross_attention_ = (value & static_cast<int>(AttentionBackend::TRT_CROSS_ATTENTION)) > 0;
    use_trt_causal_attention_ = (value & static_cast<int>(AttentionBackend::TRT_CAUSAL_ATTENTION)) > 0;
  } else {
    use_flash_attention_ = !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFlashAttention, false);
    use_efficient_attention_ = !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
    use_trt_fused_attention_ = !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedSelfAttention, false);
    use_unfused_ = true;
    use_trt_flash_attention_ = !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);
    use_trt_cross_attention_ = !ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedCrossAttention, false);
    use_trt_causal_attention_ = ParseEnvironmentVariableWithDefault<bool>(attention::kEnableFusedCausalAttention, false);
  }

  min_seq_len_for_flash_attention_packed_qkv_ = ParseEnvironmentVariableWithDefault<int>(
      attention::kMinSeqLenForFlashAttentionPackedQKV,
      attention::kDefaultMinSeqLenForFlashAttentionPackedQKV);

  min_seq_len_for_efficient_attention_fp32_ = ParseEnvironmentVariableWithDefault<int>(
      attention::kMinSeqLenForEfficientAttentionFp32,
      attention::kDefaultMinSeqLenForEfficientAttentionFp32);

  initialized_ = true;
}

const AttentionKernelOptions* AttentionKernelOptions::GetInstance(int sdpa_kernel, bool force_init) {
  if (force_init || !instance.initialized_) {
    instance.Initialize(sdpa_kernel);
  }

  return &instance;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
