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

#include "contrib_ops/cuda/moe/cutlass_extensions/gemm_configs.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace ort_fastertransformer {

enum class ActivationType { Gelu, Relu, Silu, GeGLU, ReGLU, SiGLU, Identity, InvalidType };

template <typename T, /*The type used for activations/scales/compute*/
          typename WeightType /* The type for the MoE weights */>
class MoeGemmRunner {
  public:
    MoeGemmRunner();

    void initialize(int sm);

    void try_find_best_config(int num_experts, int hidden_size, int inter_size, int num_rows) {
        // TODO: find a general way e.g a config file
        bool is_weight_only = !std::is_same<T, WeightType>::value;
        bool only_simt_configs = std::is_same<T, float>::value;
        if (sm_ == 80 && !is_weight_only && !only_simt_configs && num_experts == 8 && hidden_size == 4096 &&
            inter_size == 7168) {
            CutlassGemmConfig temp_best_config;
            temp_best_config.split_k_style = SplitKStyle::NO_SPLIT_K;
            temp_best_config.split_k_factor = 1;
            if (num_rows > 528) {
                temp_best_config.tile_config = CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64;
                temp_best_config.stages = 4;
            } else if (num_rows > 464) {
                temp_best_config.tile_config = CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64;
                temp_best_config.stages = 3;
            } else if (num_rows > 192) {
                temp_best_config.tile_config = CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64;
                temp_best_config.stages = 4;
            } else if (num_rows > 96) {
                temp_best_config.tile_config = CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64;
                temp_best_config.stages = 4;
            } else if (num_rows > 64) {
                temp_best_config.tile_config = CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64;
                temp_best_config.stages = 4;
            } else if (num_rows > 32) {
                temp_best_config.tile_config = CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64;
                temp_best_config.stages = 3;
            } else {
                temp_best_config.tile_config = CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64;
                temp_best_config.stages = 3;
            }
            best_config_ = std::move(temp_best_config);
        }
    }

    void moe_gemm_bias_act(const T *A, const WeightType *B, const T *weight_scales, const T *biases, T *C,
                           int64_t *total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                           int num_experts, ActivationType activation_type, cudaStream_t stream);

    void moe_gemm(const T *A, const WeightType *B, const T *weight_scales, const T *biases, T *C,
                  int64_t *total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                  int num_experts, cudaStream_t stream);

  private:
    template <typename EpilogueTag>
    void dispatch_to_arch(const T *A, const WeightType *B, const T *weight_scales, const T *biases, T *C,
                          int64_t *total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                          int num_experts, CutlassGemmConfig gemm_config, cudaStream_t stream,
                          int *occupancy = nullptr);

    template <typename EpilogueTag>
    void run_gemm(const T *A, const WeightType *B, const T *weight_scales, const T *biases, T *C,
                  int64_t *total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                  int num_experts, cudaStream_t stream);

  private:
    int sm_;
    int multi_processor_count_;
    std::optional<CutlassGemmConfig> best_config_{};
};

} // namespace ort_fastertransformer
