/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_v2.h"

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"  // for set_alpha
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype) {
  ORT_ENFORCE(dtype == DATA_TYPE_FP16);
  half2 h2 = __float2half2_rn(norm);
  alpha = reinterpret_cast<const uint32_t&>(h2);
}

class FusedMHARunnerFP16v2::mhaImpl {
 public:
  explicit mhaImpl(FusedMHARunnerFP16v2* interface)
      : interface(interface), sm(interface->mSm), xmmaKernel(getXMMAKernelsV2(DATA_TYPE_FP16, sm)) {
    ORT_ENFORCE((sm == kSM_70 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86),
                "Unsupported architecture");
    params.clear();
  }

  ~mhaImpl() {}

  void setup(const int S, const int B) {
    size_t warps_m{};
    size_t warps_n{};
    size_t warps_k = 1;
    if (sm == 70) {
      if (S == 64 || S == 96) {
        warps_m = 2;
        warps_n = 2;
      } else if (S == 128) {
        warps_m = 1;
        warps_n = 4;
      } else if (S == 256 || S == 384) {
        warps_m = 1;
        warps_n = 8;
      } else {
        ORT_ENFORCE(false, "Unsupporte sequence length");
      }
    } else {
      if (S == 64 || S == 96 || S == 128) {
        warps_m = 2;
        warps_n = 2;
      } else if (S == 256 || S == 192) {
        warps_m = 1;
        warps_n = 4;
      } else if (S == 384 || S == 512) {
        warps_m = 1;
        warps_n = 8;
      } else {
        ORT_ENFORCE(false, "Unsupporte sequence length");
      }
    }
    // The number of threads per CTA.
    threads_per_cta = warps_m * warps_n * warps_k * 32;
    // The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension.
    xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);
    // The number of xmmas in the N dimension.
    xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);

    const float scale_bmm1 = interface->mRsqrtHeadSize;
    const float scale_softmax = 1.f;  // Seems to be only required for int8
    const float scale_bmm2 = 1.f;

    Data_type scale_type = DATA_TYPE_FP16;
    set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
    set_alpha(params.scale_softmax, scale_softmax, scale_type);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

    params.b = B;
    params.h = interface->mNumHeads;
    params.s = S;
    params.d = interface->mHeadSize;

    params.qkv_stride_in_bytes = 3 * interface->mNumHeads * interface->mHeadSize * sizeof(half);
    params.packed_mask_stride_in_bytes = xmmas_m * threads_per_cta * sizeof(uint32_t);
    params.o_stride_in_bytes = interface->mNumHeads * interface->mHeadSize * sizeof(half);
  }

  void run(const void* qkvPtr, const void* /*maskPtr*/, const void* seqLens,
           void* output, void* workspace, cudaStream_t stream) {
    params.qkv_ptr = const_cast<void*>(qkvPtr);

    // dummy input in V2/V3 because now we use cu_seqlens
    params.packed_mask_ptr = nullptr;

    params.o_ptr = output;

    params.cu_seqlens = static_cast<int*>(const_cast<void*>(seqLens));
    xmmaKernel->run(params, stream);
    CUDA_CALL_THROW(cudaPeekAtLastError());
  }

  bool isValid(int s) const {
    return xmmaKernel->isValid(s);
  }

 private:
  FusedMHARunnerFP16v2* interface;
  Fused_multihead_attention_params_v2 params;
  int sm;
  const FusedMultiHeadAttentionXMMAKernelV2* xmmaKernel;
  size_t xmmas_m;
  size_t xmmas_n;
  size_t threads_per_cta;
};

FusedMHARunnerFP16v2::FusedMHARunnerFP16v2(const int numHeads, const int headSize, const int sm)
    : MHARunner(numHeads, headSize, 2), mSm(sm), pimpl(new mhaImpl(this)) {
}

void FusedMHARunnerFP16v2::setup(const int S, const int B) {
  MHARunner::setup(S, B);
  pimpl->setup(S, B);
}

size_t FusedMHARunnerFP16v2::getWorkspaceSize() const {
  return 0;
}

bool FusedMHARunnerFP16v2::isValid(int s) const {
  return pimpl->isValid(s);
}

void FusedMHARunnerFP16v2::run(const void* qkvPtr, const void* maskPtr, const void* seqLens,
                               void* output, void* workspace, cudaStream_t stream) {
  return pimpl->run(qkvPtr, maskPtr, seqLens, output, workspace, stream);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
