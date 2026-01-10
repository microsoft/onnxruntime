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

// Modifications: Update interface and implmentation to be thread-safe
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

class FusedMHARunnerFP16v2::FmhaImpl {
 public:
  FmhaImpl(FusedMHARunnerFP16v2* interface, int sm)
      : interface_(interface),
        sm_(sm),
        xmma_kernel_(getXMMAKernelsV2(DATA_TYPE_FP16, sm)) {
    ORT_ENFORCE((sm == kSM_70 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86 || sm == kSM_89),
                "Unsupported architecture");

    flash_kernel_ = nullptr;
    if (interface_->enable_flash_attention_) {
      flash_kernel_ = get_flash_attention_kernels(DATA_TYPE_FP16, sm);
    }
  }

  ~FmhaImpl() {}

  void Setup(Fused_multihead_attention_params_v2& params,
             int sequence_length,  // normalized sequence length
             int batch_size,
             bool& use_flash_attention) const {
    use_flash_attention = UseFlashAttention(sequence_length);

    params.force_unroll = use_flash_attention;

    size_t warps_m = 2;
    size_t warps_n = 2;
    size_t warps_k = 1;

    if (use_flash_attention) {
      warps_m = 4;
      warps_n = 1;
    } else {
      if (sm_ == 70) {
        if (sequence_length == 64 || sequence_length == 96) {
          warps_m = 2;
          warps_n = 2;
        } else if (sequence_length == 128) {
          warps_m = 1;
          warps_n = 4;
        } else if (sequence_length == 256 || sequence_length == 384) {
          warps_m = 1;
          warps_n = 8;
        } else {
          ORT_ENFORCE(false, "Unsupported sequence length");
        }
      } else {
        if (sequence_length == 32 || sequence_length == 64 || sequence_length == 96 || sequence_length == 128) {
          warps_m = 2;
          warps_n = 2;
        } else if (sequence_length == 192 || sequence_length == 256) {
          warps_m = 1;
          warps_n = 4;
        } else if (sequence_length == 384) {
          warps_m = 1;
          warps_n = 8;
        } else {
          ORT_ENFORCE(false, "Unsupported sequence length");
        }
      }
    }

    // The number of threads per CTA.
    size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
    // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
    size_t xmmas_m = (sequence_length + 16 * warps_m - 1) / (16 * warps_m);

    const float scale_bmm1 = interface_->scale_;
    const float scale_softmax = 1.f;  // Seems to be only required for int8
    const float scale_bmm2 = 1.f;

    set_alpha_fp16(params.scale_bmm1, scale_bmm1);
    set_alpha_fp16(params.scale_softmax, scale_softmax);
    set_alpha_fp16(params.scale_bmm2, scale_bmm2);

    params.b = batch_size;
    params.h = interface_->num_heads_;
    params.s = sequence_length;
    params.d = interface_->head_size_;

    params.qkv_stride_in_bytes = 3 * interface_->num_heads_ * interface_->head_size_ * sizeof(half);
    params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
    params.o_stride_in_bytes = interface_->num_heads_ * interface_->head_size_ * sizeof(half);
  }

  void SetupCausal(Fused_multihead_attention_params_v2& params,
                   int sequence_length,  // normalized sequence length
                   int batch_size,
                   bool& use_flash_attention) const {
    const float scale_bmm1 = interface_->scale_;
    const float scale_softmax = 1.f;  // Seems to be only required for int8
    const float scale_bmm2 = 1.f;

    set_alpha_fp16(params.scale_bmm1, scale_bmm1);
    set_alpha_fp16(params.scale_softmax, scale_softmax);
    set_alpha_fp16(params.scale_bmm2, scale_bmm2);

    params.b = batch_size;
    params.h = interface_->num_heads_;
    params.s = sequence_length;
    params.d = interface_->head_size_;

    params.qkv_stride_in_bytes = 3 * interface_->num_heads_ * interface_->head_size_ * sizeof(half);
    params.o_stride_in_bytes = interface_->num_heads_ * interface_->head_size_ * sizeof(half);

    use_flash_attention = interface_->enable_flash_attention_;

    // fallback to original fmha_v2 when head_size <= 64 and sequence_length <= 128
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
  }

  void Run(Fused_multihead_attention_params_v2& params,
           const void* input,
           const void* cu_seqlens,
           void* output,
           cudaStream_t stream,
           bool use_flash_attention,
           bool has_causal_mask) const {
    params.qkv_ptr = const_cast<void*>(input);
    params.o_ptr = output;
    params.cu_seqlens = static_cast<int*>(const_cast<void*>(cu_seqlens));

    if (use_flash_attention && flash_kernel_ != nullptr && !has_causal_mask) {
      flash_kernel_->run(params, stream);
    } else {
      xmma_kernel_->run(params, stream, use_flash_attention, has_causal_mask);
    }

    CUDA_CALL_THROW(cudaPeekAtLastError());
  }

  bool IsValid(int sequence_length) const {
    if (UseFlashAttention(sequence_length)) {
      return (flash_kernel_ != nullptr) && flash_kernel_->isValid(sequence_length);
    }

    return xmma_kernel_->isValid(sequence_length);
  }

  int NormalizeSequenceLength(int max_seq_len) const {
    if (UseFlashAttention(max_seq_len)) {
      return max_seq_len;
    }

    int sequence_length = max_seq_len;
    if (max_seq_len <= 32) {
      sequence_length = (sm_ == 70) ? 64 : 32;
    } else if (max_seq_len <= 64) {
      sequence_length = 64;
    } else if (max_seq_len <= 96) {
      sequence_length = 96;
    } else if (max_seq_len <= 128) {
      sequence_length = 128;
    } else if (max_seq_len <= 192) {
      sequence_length = (sm_ == 70) ? 256 : 192;
    } else if (max_seq_len <= 256) {
      sequence_length = 256;
    } else if (max_seq_len <= 384) {
      sequence_length = 384;
    }

    return sequence_length;
  }

 protected:
  bool UseFlashAttention(int sequence_length) const {
    ORT_ENFORCE(interface_->is_causal_ == false);
    return interface_->enable_flash_attention_ && sequence_length >= kMinSequenceLengthFlashAttention;
  }

 private:
  FusedMHARunnerFP16v2* interface_;
  int sm_;
  const FusedMultiHeadAttentionXMMAKernelV2* xmma_kernel_;
  const FusedMultiHeadFlashAttentionKernel* flash_kernel_;
};

FusedMHARunnerFP16v2::FusedMHARunnerFP16v2(int num_heads,
                                           int head_size,
                                           int sm,
                                           bool causal,
                                           bool enable_flash_attention,
                                           float scale)
    : MHARunner(num_heads, head_size, causal, scale),
      enable_flash_attention_(enable_flash_attention),
      impl_(new FmhaImpl(this, sm)) {
}

bool FusedMHARunnerFP16v2::IsSupported(int sm, int head_size, int sequence_length,
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

void FusedMHARunnerFP16v2::Run(int batch_size,
                               int normalized_sequence_length,
                               const void* input,
                               const void* cu_seqlens,
                               void* output,
                               cudaStream_t stream) const {
  Fused_multihead_attention_params_v2 params;
  bool use_flash_attention = false;
  if (is_causal_) {
    impl_->SetupCausal(params, normalized_sequence_length, batch_size, use_flash_attention);
  } else {
    impl_->Setup(params, normalized_sequence_length, batch_size, use_flash_attention);
  }

  impl_->Run(params, input, cu_seqlens, output, stream, use_flash_attention, is_causal_);
}

bool FusedMHARunnerFP16v2::IsValid(int normalized_sequence_length) const {
  return impl_->IsValid(normalized_sequence_length);
}

int FusedMHARunnerFP16v2::NormalizeSequenceLength(int max_seq_len) const {
  return impl_->NormalizeSequenceLength(max_seq_len);
}

std::unique_ptr<MHARunner> FusedMHARunnerFP16v2::Create(int num_heads,
                                                        int head_size,
                                                        int sm,
                                                        bool causal,
                                                        bool enable_flash_attention,
                                                        const float scale) {
#ifdef _MSC_VER
  return std::make_unique<FusedMHARunnerFP16v2>(num_heads, head_size, sm, causal, enable_flash_attention, scale);
#else
  // Linux build has error using make_unique: invalid application of ‘sizeof’ to
  // incomplete type ‘onnxruntime::contrib::cuda::FusedMHARunnerFP16v2::FmhaImpl
  std::unique_ptr<MHARunner> runner;
  runner.reset(new FusedMHARunnerFP16v2(num_heads, head_size, sm, causal, enable_flash_attention, scale));
  return runner;
#endif
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
