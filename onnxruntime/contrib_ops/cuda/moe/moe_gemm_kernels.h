/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include <cuda_runtime_api.h>

namespace fastertransformer {

enum class ActivationType {
    Gelu,
    Relu,
    Silu,
    GeGLU,
    ReGLU,
    SiGLU,
    Identity,
    InvalidType
};

// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig {
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // Warp configs for M=128
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape128x32x64
};

enum class SplitKStyle {
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    // SPLIT_K_PARALLEL // Not supported yet
};

struct CutlassGemmConfig {
    CutlassTileConfig tile_config    = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle       split_k_style  = SplitKStyle::NO_SPLIT_K;
    int               split_k_factor = -1;
    int               stages         = -1;
};

template<typename T, /*The type used for activations/scales/compute*/
         typename WeightType /* The type for the MoE weights */>
class MoeGemmRunner {
public:
    MoeGemmRunner();

    void moe_gemm_bias_act(const T*          A,
                           const WeightType* B,
                           const T*          weight_scales,
                           const T*          biases,
                           T*                C,
                           int64_t*          total_rows_before_expert,
                           int64_t           total_rows,
                           int64_t           gemm_n,
                           int64_t           gemm_k,
                           int               num_experts,
                           ActivationType    activation_type,
                           cudaStream_t      stream);

    void moe_gemm(const T*          A,
                  const WeightType* B,
                  const T*          weight_scales,
                  T*                C,
                  int64_t*          total_rows_before_expert,
                  int64_t           total_rows,
                  int64_t           gemm_n,
                  int64_t           gemm_k,
                  int               num_experts,
                  cudaStream_t      stream);

private:
    template<typename EpilogueTag>
    void dispatch_to_arch(const T*          A,
                          const WeightType* B,
                          const T*          weight_scales,
                          const T*          biases,
                          T*                C,
                          int64_t*          total_rows_before_expert,
                          int64_t           total_rows,
                          int64_t           gemm_n,
                          int64_t           gemm_k,
                          int               num_experts,
                          CutlassGemmConfig gemm_config,
                          cudaStream_t      stream,
                          int*              occupancy = nullptr);

    template<typename EpilogueTag>
    void run_gemm(const T*          A,
                  const WeightType* B,
                  const T*          weight_scales,
                  const T*          biases,
                  T*                C,
                  int64_t*          total_rows_before_expert,
                  int64_t           total_rows,
                  int64_t           gemm_n,
                  int64_t           gemm_k,
                  int               num_experts,
                  cudaStream_t      stream);

private:
    int sm_;
    int multi_processor_count_;
};

}  // namespace fastertransformer
