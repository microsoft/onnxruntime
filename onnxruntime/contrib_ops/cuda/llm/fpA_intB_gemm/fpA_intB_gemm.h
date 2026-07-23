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

#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/weight_only_quant_op.h"
#include "core/common/safeint.h"
#include <cuda_runtime_api.h>
#include <optional>
#include <vector>

namespace tkc = onnxruntime::llm::cutlass_extensions;

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {

// TRT Activation Type does not have Gelu or Silu
enum class ActivationType {
  Gelu,
  Relu,
  Silu,
  Identity,
  InvalidType
};

/*
  This runner only supports:
  T in {half, __nv_bfloat} WeightType in {int8_t, cutlass::uint4b_t}

  Activations, biases, scales and outputs are all assumed to be row-major.

  However, it is assumed that B is in a special format governed by cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.
  In this case, B must be preprocessed using the cutlass weight only quant preprocessors. The weight preprocessor
  will instantiate the layout and preprocess based on the instantiation, so layout changes should only require
  modifications to mix_gemm_B_layout.h.
*/

class CutlassFpAIntBGemmRunnerInterface {
 public:
  CutlassFpAIntBGemmRunnerInterface() {}

  virtual ~CutlassFpAIntBGemmRunnerInterface() {}

  virtual void gemm(void const* A, void const* B, void const* weight_scales, void* C, int m, int n, int k,
                    tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) = 0;

  virtual void gemm(void const* A, void const* B, void const* weight_scales, float const alpha, void* C, int m, int n,
                    int k, tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
                    cudaStream_t stream) = 0;

  virtual void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
                    void const* biases, void* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig,
                    char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) = 0;

  virtual void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
                    void const* biases, float const alpha, void* C, int m, int n, int k, int const group_size,
                    tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) = 0;

  // Returns desired workspace size in bytes.
  virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

  virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

  // Overrides the SM architecture used for tactic/config enumeration, workspace sizing and kernel
  // dispatch. By default the runner targets the detected device SM. On SM90 the half/bf16
  // weight-only path dispatches the SM80 (Ampere) mixed-GEMM kernel (which now runs on Hopper, see
  // GemmFpAIntB::operator()), so MatMulNBits forces the runner to SM80 to keep the enumerated
  // tactics (tile_config_sm80) and workspace sizing consistent with the dispatched kernel.
  // Default: no-op (keep detected SM).
  virtual void setArch(int /*sm*/) {}

  // Opts in to the native SM90 (Hopper TMA/WGMMA) mixed-GEMM kernel instead of the SM80
  // compatibility path. Only meaningful when the runner targets SM90 (setArch is left at 90) and
  // the weights were prepacked for the Hopper layout. Default: no-op (keep the SM80 kernel).
  virtual void setUseSm90Native(bool /*use*/) {}

  // These tile constants are public (not protected) so the free/namespace-scope helper
  // ComputeFpAIntBGemmWorkspaceSize below - which is not a member of this class - can read them.
  // They are template/dtype-independent (they describe the launch grid, not the element type).
  static constexpr int SPLIT_K_LIMIT = 7;
  static constexpr int MIN_M_TILE = 16;
  static constexpr int MIN_N_TILE = 64;

  static constexpr int MAX_M_TILE_SM90 = 128;
  static constexpr int MAX_N_TILE_SM90 = 256;
};

// Shared, stateless workspace-size formula for the fpA_intB / weight-only CUTLASS GEMM. This is the
// single source of truth funnelled through by all three call sites (Phase-A memory roadmap, issue
// microsoft/onnxruntime#29775):
//   - CutlassFpAIntBGemmRunner<...>::getWorkspaceSize (runtime, via the constructed runner);
//   - MatMulNBits::DeclareWorkspaceRequirements (Level 2, instance-level, after CreateKernels);
//   - EstimateMatMulNBitsWorkspace (Level 1, partition-time, before any kernel instance exists).
// It is pure arithmetic: no CUDA calls, no device state, no tensor data - it depends only on
// m, n, sm and multi_processor_count (k is unused, matching the CUTLASS runner). Every intermediate
// is computed with SafeInt<size_t> so adversarial (untrusted-model) dimensions cannot silently
// overflow; on overflow this returns std::nullopt instead of throwing.
inline std::optional<size_t> ComputeFpAIntBGemmWorkspaceSize(int m, int n, int /*k*/, int sm,
                                                             int multi_processor_count) {
  try {
    using Interface = CutlassFpAIntBGemmRunnerInterface;
#ifndef EXCLUDE_SM_90
    if (sm == 90) {
      // For Hopper, reserve a large workspace for the potential stream-K partial-sum reduction.
      // sk_tiles <= 2 * ctas_per_wave (== 2 * multi_processor_count); sk_units <= multi_processor_count;
      // the final scaled sk_tiles is at most 2 * max_sk_tiles + max_sk_units. Each entry is one float.
      SafeInt<size_t> mpc(multi_processor_count);
      SafeInt<size_t> max_sk_tiles = mpc * 2;
      SafeInt<size_t> max_sk_units = mpc;
      SafeInt<size_t> max_sk_tiles_with_separate_reduction = max_sk_tiles * 2 + max_sk_units;
      SafeInt<size_t> bytes = max_sk_tiles_with_separate_reduction *
                              static_cast<size_t>(Interface::MAX_M_TILE_SM90) *
                              static_cast<size_t>(Interface::MAX_N_TILE_SM90) * sizeof(float);
      return static_cast<size_t>(bytes);
    }
#else
    (void)sm;
    (void)multi_processor_count;
#endif
    // These are the min tile sizes for each config, which launch the maximum number of blocks.
    // ceil_div(a, b) == (a + b - 1) / b; compute the numerator with SafeInt too so it cannot
    // overflow before the division.
    SafeInt<size_t> max_grid_m = (SafeInt<size_t>(m) + (Interface::MIN_M_TILE - 1)) / Interface::MIN_M_TILE;
    SafeInt<size_t> max_grid_n = (SafeInt<size_t>(n) + (Interface::MIN_N_TILE - 1)) / Interface::MIN_N_TILE;
    // We need 4 bytes per block in the worst case, launched split_k_limit deep in the z dimension.
    SafeInt<size_t> bytes = max_grid_m * max_grid_n * Interface::SPLIT_K_LIMIT * 4;
    return static_cast<size_t>(bytes);
  } catch (const OnnxRuntimeException&) {
    return std::nullopt;
  }
}

template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp,
          typename ScaleZeroType = ActivationType, typename BiasType = ActivationType, typename OutputType = ActivationType>
class CutlassFpAIntBGemmRunner : public virtual CutlassFpAIntBGemmRunnerInterface {
 public:
  CutlassFpAIntBGemmRunner();
  ~CutlassFpAIntBGemmRunner();

  void gemm(void const* A, void const* B, void const* weight_scales, void* C, int m, int n, int k,
            tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
            cudaStream_t stream) override;

  void gemm(void const* A, void const* B, void const* weight_scales, float const alpha, void* C, int m, int n, int k,
            tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
            cudaStream_t stream) override;

  void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
            void const* biases, void* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig,
            char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) override;

  void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
            void const* biases, float const alpha, void* C, int m, int n, int k, int const group_size,
            tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
            cudaStream_t stream) override;

  // Disabled since the fused GEMM, activation kernels will not be used in v1.

  // void gemm_bias_act(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C, int m, int n,
  //     int k, ActivationType activation_type, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t
  //     stream);

  // Returns desired workspace size in bytes.
  size_t getWorkspaceSize(int const m, int const n, int const k) override;

  std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

  void setArch(int sm) override { sm_ = sm; }

  void setUseSm90Native(bool use) override { use_sm90_native_ = use; }

 private:
  template <typename EpilogueTag>
  void dispatch_to_arch(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
                        ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
                        int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
                        const size_t workspace_bytes, cudaStream_t stream, int* occupancy = nullptr);

 private:
  int sm_;
  int multi_processor_count_;
  bool use_sm90_native_{false};
};

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
