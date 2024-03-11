/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_v2.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/flash_attention/fmha_flash_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

union __half2_uint32_t_union {
  half2 fp162;
  uint32_t u32;
};

void set_alpha_fp16(uint32_t& alpha, float norm) {
  __half2_uint32_t_union temp;
  temp.u32 = 0;
  temp.fp162 = __float2half2_rn(norm);
  alpha = temp.u32;
}

class FusedMHARunnerFP16v2::mhaImpl {
 public:
  mhaImpl(FusedMHARunnerFP16v2* interface)
      : interface(interface),
        sm(interface->mSm),
        xmmaKernel(getXMMAKernelsV2(DATA_TYPE_FP16, sm)) {
    ORT_ENFORCE((sm == kSM_70 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 || sm == kSM_89),
                "Unsupported architecture");

    flash_attention_kernel = nullptr;
    if (interface->mEnableFlashAttention) {
      flash_attention_kernel = get_flash_attention_kernels(DATA_TYPE_FP16, sm);
    }

    params.clear();
  }

  ~mhaImpl() {}

  void setup(const int seq_len, const int B) {
    // For bert and vit, use flash attention when sequence length is larger than the threshold.
    use_flash_attention = is_flash_attention(seq_len);

    params.force_unroll = use_flash_attention;

    size_t warps_m = 2;
    size_t warps_n = 2;
    size_t warps_k = 1;

    if (use_flash_attention) {
      warps_m = 4;
      warps_n = 1;
    } else {
      if (sm == 70) {
        if (seq_len == 64 || seq_len == 96) {
          warps_m = 2;
          warps_n = 2;
        } else if (seq_len == 128) {
          warps_m = 1;
          warps_n = 4;
        } else if (seq_len == 256 || seq_len == 384) {
          warps_m = 1;
          warps_n = 8;
        } else {
          ORT_ENFORCE(false, "Unsupported sequence length");
        }
      } else {
        if (seq_len == 32 || seq_len == 64 || seq_len == 96 || seq_len == 128) {
          warps_m = 2;
          warps_n = 2;
        } else if (seq_len == 192 || seq_len == 256) {
          warps_m = 1;
          warps_n = 4;
        } else if (seq_len == 384) {
          warps_m = 1;
          warps_n = 8;
        } else {
          ORT_ENFORCE(false, "Unsupported sequence length");
        }
      }
    }

    // The number of threads per CTA.
    threads_per_cta = warps_m * warps_n * warps_k * 32;
    // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
    xmmas_m = (seq_len + 16 * warps_m - 1) / (16 * warps_m);

    const float scale_bmm1 = interface->mScale;
    const float scale_softmax = 1.f;  // Seems to be only required for int8
    const float scale_bmm2 = 1.f;

    set_alpha_fp16(params.scale_bmm1, scale_bmm1);
    set_alpha_fp16(params.scale_softmax, scale_softmax);
    set_alpha_fp16(params.scale_bmm2, scale_bmm2);

    params.b = B;
    params.h = interface->mNumHeads;
    params.s = seq_len;
    params.d = interface->mHeadSize;

    params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
    params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
    params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(half);

    has_causal_mask = false;
  }

  void setup_causal_masked_fmha(const int seq_len, const int B) {
    const float scale_bmm1 = interface->mScale;
    const float scale_softmax = 1.f;  // Seems to be only required for int8
    const float scale_bmm2 = 1.f;

    set_alpha_fp16(params.scale_bmm1, scale_bmm1);
    set_alpha_fp16(params.scale_softmax, scale_softmax);
    set_alpha_fp16(params.scale_bmm2, scale_bmm2);

    params.b = B;
    params.h = interface->mNumHeads;
    params.s = seq_len;
    params.d = interface->mHeadSize;

    params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
    params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(half);

    // fallback to original fmha_v2 when head_size <= 64 and seq_len <- 128
    use_flash_attention = interface->mEnableFlashAttention;
    if (params.d <= 64 && params.s <= 128) {
      use_flash_attention = false;
      // get max sequence length
      if (params.s > 64) {
        params.s = 128;
      } else {
        params.s = 64;
      }
    }

    // set flags
    params.force_unroll = use_flash_attention;
    has_causal_mask = true;
  }

  void run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) {
    params.qkv_ptr = const_cast<void*>(input);
    params.o_ptr = output;
    params.cu_seqlens = static_cast<int*>(const_cast<void*>(cu_seqlens));

    if (use_flash_attention && flash_attention_kernel != nullptr && !has_causal_mask) {
      flash_attention_kernel->run(params, stream);
    } else {
      xmmaKernel->run(params, stream, use_flash_attention, has_causal_mask);
    }

    CUDA_CALL_THROW(cudaPeekAtLastError());
  }

  bool isValid(int s) const {
    if (is_flash_attention(s)) {
      return (flash_attention_kernel != nullptr) && flash_attention_kernel->isValid(s);
    }

    return xmmaKernel->isValid(s);
  }

  int getSFromMaxSeqLen(const int max_seq_len) const {
    if (is_flash_attention(max_seq_len)) {
      return max_seq_len;
    }

    int seq_len = max_seq_len;
    if (max_seq_len <= 32) {
      seq_len = (sm == 70) ? 64 : 32;
    } else if (max_seq_len <= 64) {
      seq_len = 64;
    } else if (max_seq_len <= 96) {
      seq_len = 96;
    } else if (max_seq_len <= 128) {
      seq_len = 128;
    } else if (max_seq_len <= 192) {
      seq_len = (sm == 70) ? 256 : 192;
    } else if (max_seq_len <= 256) {
      seq_len = 256;
    } else if (max_seq_len <= 384) {
      seq_len = 384;
    }

    return seq_len;
  }

 protected:
  bool is_flash_attention(const int seq_len) const {
    ORT_ENFORCE(interface->mHasCausalMask == false);
    return interface->mEnableFlashAttention && seq_len >= kMinSequenceLengthFlashAttention;
  }

 private:
  FusedMHARunnerFP16v2* interface;
  Fused_multihead_attention_params_v2 params;
  int sm;
  const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
  const FusedMultiHeadFlashAttentionKernel* flash_attention_kernel;
  size_t xmmas_m;
  size_t threads_per_cta;
  bool use_flash_attention = false;
  bool has_causal_mask = false;
};

FusedMHARunnerFP16v2::FusedMHARunnerFP16v2(const int numHeads,
                                           const int headSize,
                                           const int sm,
                                           bool causal_mask,
                                           bool enable_flash_attention,
                                           const float scale)
    : MHARunner(numHeads, headSize, 2, causal_mask, scale),
      mSm(sm),
      mEnableFlashAttention(enable_flash_attention),
      pimpl(new mhaImpl(this)) {
}

void FusedMHARunnerFP16v2::setup(const int seq_len, const int B) {
  MHARunner::setup(seq_len, B);
  if (mHasCausalMask) {
    pimpl->setup_causal_masked_fmha(seq_len, B);
  } else {
    pimpl->setup(seq_len, B);
  }
}

bool FusedMHARunnerFP16v2::is_supported(int sm, int head_size, int sequence_length,
                                        bool enable_flash_attention, bool causal) {
  if (causal) {
    if (!(sm == kSM_70 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 || sm == kSM_89)) {
      return false;
    }

    if (enable_flash_attention) {
      return head_size == 64 ||
             head_size == 32 ||
             head_size == 40 ||
             head_size == 80 ||
             head_size == 128 ||
             head_size == 144 ||
             head_size == 160 ||
             head_size == 256;
    }

    return (head_size == 64 || head_size == 32 || head_size == 40) && sequence_length <= 128;
  }

  bool use_flash = enable_flash_attention && sequence_length >= kMinSequenceLengthFlashAttention;
  if (use_flash && has_flash_attention_kernel(sm, head_size)) {
    return true;
  }

  if (!(sm == kSM_70 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 || sm == kSM_89)) {
    return false;
  }

  if (head_size != 64 && head_size != 32) {
    return false;
  }

  if (sm == kSM_70 && head_size == 32) {
    return false;
  }

  // Normal (not flash) fused kernel supports sequence length up to 384.
  constexpr int max_sequence_length = 384;
  return sequence_length <= max_sequence_length;
}

size_t FusedMHARunnerFP16v2::getWorkspaceSize() const {
  return 0;
}

void FusedMHARunnerFP16v2::run(const void* input, const void* cu_seqlens, void* output, cudaStream_t stream) {
  pimpl->run(input, cu_seqlens, output, stream);
}

bool FusedMHARunnerFP16v2::isValid(int s) const {
  return pimpl->isValid(s);
}

int FusedMHARunnerFP16v2::getSFromMaxSeqLen(const int max_seq_len) const {
  return pimpl->getSFromMaxSeqLen(max_seq_len);
}

std::unique_ptr<MHARunner> FusedMHARunnerFP16v2::Create(const int numHeads,
                                                                   const int headSize,
                                                                   const int sm,
                                                                   bool causal_mask,
                                                                   bool enable_flash_attention,
                                                                   const float scale) {
#ifdef _MSC_VER
  return std::make_unique<FusedMHARunnerFP16v2>(numHeads, headSize, sm, causal_mask, enable_flash_attention, scale);
#else
  // Linux build has error using make_unique: invalid application of ‘sizeof’ to incomplete type ‘onnxruntime::contrib::cuda::FusedMHARunnerFP16v2::mhaImpl
  std::unique_ptr<MHARunner> runner;
  runner.reset(new FusedMHARunnerFP16v2(numHeads, headSize, sm, causal_mask, enable_flash_attention, scale));
  return runner;
#endif
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
