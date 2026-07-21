// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulNBits operator, it is basically
// matmul float with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//
#pragma once
#include <atomic>
#include <optional>
#include <string>
#include <vector>
#include "core/common/safeint.h"
#include "core/common/string_utils.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_profiler.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

// Environment variables to force the chunked dequant+GEMM path for testing.
// ORT_MATMULNBITS_FORCE_CHUNKED=1 bypasses the scratch-size and min-N guards.
// ORT_MATMULNBITS_CHUNK_SIZE overrides the default chunk size (32768).
constexpr const char* kForceChunkedEnvVar = "ORT_MATMULNBITS_FORCE_CHUNKED";
constexpr const char* kChunkSizeEnvVar = "ORT_MATMULNBITS_CHUNK_SIZE";
constexpr int64_t kDefaultChunkTargetRows = 32768;

#if USE_FPA_INTB_GEMM
using onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
using onnxruntime::llm::kernels::weight_only::GemmDims;
using onnxruntime::llm::kernels::weight_only::GemmIdCore;
using onnxruntime::llm::kernels::weight_only::GemmPluginProfilerManager;
using onnxruntime::llm::kernels::weight_only::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using GemmProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>;

// Environment variable to enable/disable the fpA_intB path: unset/0/off to disable, other value to enable.
// This only affects nodes whose weights are NOT prepacked (see the constructor).
constexpr const char* kFpAIntBGemmOption = "ORT_FPA_INTB_GEMM";

constexpr int64_t kMatMulNBitsWeightNotPrepacked = 0;
constexpr int64_t kMatMulNBitsWeightPrepackedSm80 = 1;
constexpr int64_t kMatMulNBitsWeightPrepackedSm90 = 2;

// Session-option config keys. These are readable by BOTH the built-in CUDA EP and the CUDA plugin
// EP: every kernel is created via KernelRegistryManager::CreateKernel, which injects the
// session-level ConfigOptions, and the plugin CUDA EP wraps a CUDAExecutionProvider that reuses this
// same kernel. Each key overrides its ORT_* environment-variable equivalent (config wins).
//   ep.cuda.fpa_intb_gemm       <-> ORT_FPA_INTB_GEMM       (0/off, 1/on)
//   ep.cuda.fpa_intb_profile_m  <-> ORT_FPA_INTB_PROFILE_M  (initial profile M buckets)
constexpr const char* kConfigFpAIntBGemm = "ep.cuda.fpa_intb_gemm";
constexpr const char* kConfigFpAIntBProfileM = "ep.cuda.fpa_intb_profile_m";

// Resolves a setting from the session config first (per-session, EP-agnostic), then the environment
// variable, else empty. Session config wins so a model/session can override a process-wide env var.
inline std::string ResolveFpAIntBConfigOrEnv(const OpKernelInfo& info, const char* config_key,
                                             const char* env_key) {
  const auto from_config = info.GetConfigOptions().GetConfigEntry(config_key);
  if (from_config.has_value()) {
    return *from_config;
  }
  return ParseEnvironmentVariableWithDefault<std::string>(env_key, "");
}

// Parses the fpA_intB enable flag. "on" enables the full fpA_intB path (the CUTLASS GEMM and, where
// supported, the GEMV decode kernel; they share one weight layout and cannot be split). Accepts
// (case-insensitive): "" / "0" / "off" -> disabled; otherwise, enabled (a non-zero numeric value
// still enables, for backward compatibility).
inline bool ParseFpAIntBEnabled(const std::string& value) {
  const std::string lowered = onnxruntime::utils::GetLowercaseString(onnxruntime::utils::TrimString(value));
  if (lowered.empty() || lowered == "0" || lowered == "off") {
    return false;
  }
  return true;
}

// Effective SM architecture that the fpA_intB CUTLASS runner uses for workspace sizing AFTER
// InitGemmProfiler's setArch() call (matmul_nbits.cc). This is the SINGLE source of the effective-arch
// rule: MatMulNBits::FpAIntBPackingSmForKernel() delegates to this function, so Level 1 (the
// EstimateMatMulNBitsWorkspace estimate) and Level 2 / the runtime are compiler-guaranteed to agree.
// The runner targets native SM90 only when the device is SM90 AND the weights were prepacked for the
// Hopper layout (weight_prepacked == 2); every other case runs the SM80-compat kernel, whose
// workspace formula ignores sm.
inline int EffectiveFpAIntBWorkspaceSm(int device_sm, int64_t weight_prepacked) {
  return (device_sm == 90 && weight_prepacked == kMatMulNBitsWeightPrepackedSm90) ? 90 : 80;
}

// Single source of truth for the fpA_intB / CUTLASS weight-only-GEMM eligibility decision. Reads
// only node attributes + input-0 dtype + device SM (no kernel instance required). Called from BOTH
// the MatMulNBits constructor (to compute has_fpA_intB_gemm_) and EstimateMatMulNBitsWorkspace
// (Level 1), so the two can never disagree about whether a node takes the fpA_intB path. Returns
// true iff the node is eligible for the fpA_intB path.
//
// input0_elem_type is an onnx::TensorProto_DataType (FLOAT16 / BFLOAT16 are eligible; FLOAT is not).
// fpa_intb_option is the resolved ORT_FPA_INTB_GEMM / ep.cuda.fpa_intb_gemm flag (0 = off). A
// prepacked weight (weight_prepacked != 0) forces the fpA_intB path on regardless of the option.
bool CheckFpAIntBEligibility(int32_t input0_elem_type, int64_t N, int64_t K,
                             int64_t nbits, int64_t block_size,
                             int64_t weight_prepacked, bool has_g_idx,
                             int device_sm, int fpa_intb_option);

// Level 1 partition-time workspace estimate for a MatMulNBits node, callable during GetCapability()
// before any kernel instance exists. Returns nullopt when the node is not fpA_intB-eligible, when
// the leading (M) dimension of input A is not statically known, or when the size formula overflows.
std::optional<size_t> EstimateMatMulNBitsWorkspace(const Node& node, const cudaDeviceProp& device_prop);
#endif

template <typename T>
class MatMulNBits final : public CudaKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));

    constexpr int kInputIndexScale = 2;
    constexpr int kInputIndexZeroPoints = 3;
    constexpr int kInputIndexGroupIndex = 4;
    constexpr int kInputIndexBias = 5;

#ifdef BUILD_CUDA_EP_AS_PLUGIN
    // PLUGIN BUILD ADAPTATION: The adapter Node does not expose InputDefs(),
    // so we cannot check whether optional inputs (zero_points, g_idx, bias)
    // truly exist at construction time. Instead, we check input count here
    // and verify actual tensor presence in ComputeInternal.
    ORT_UNUSED_PARAMETER(kInputIndexScale);  // only used in non-plugin path for type checking
    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints;
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex;
    has_bias_ = info.GetInputCount() > kInputIndexBias;
    // is_zero_points_scale_same_type_ defaults to false; checked at runtime in plugin path.
#else
    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints && info.node().InputDefs()[kInputIndexZeroPoints]->Exists();
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex && info.node().InputDefs()[kInputIndexGroupIndex]->Exists();
    has_bias_ = info.GetInputCount() > kInputIndexBias && info.node().InputDefs()[kInputIndexBias]->Exists();

    if (has_zero_points_) {
      int32_t zero_point_type = info.node().InputDefs()[kInputIndexZeroPoints]->TypeAsProto()->tensor_type().elem_type();
      int32_t scale_type = info.node().InputDefs()[kInputIndexScale]->TypeAsProto()->tensor_type().elem_type();
      is_zero_points_scale_same_type_ = (zero_point_type == scale_type);
    }
#endif
    sm_ = this->GetDeviceProp().major * 10 + this->GetDeviceProp().minor;

    force_chunked_ = ParseEnvironmentVariableWithDefault<int>(kForceChunkedEnvVar, 0) != 0;
    chunk_target_rows_ = ParseEnvironmentVariableWithDefault<int64_t>(kChunkSizeEnvVar, kDefaultChunkTargetRows);
    if (chunk_target_rows_ < 1) {
      chunk_target_rows_ = kDefaultChunkTargetRows;
    }

#if USE_FPA_INTB_GEMM
    weight_prepacked_ = info.GetAttrOrDefault<int64_t>("weight_prepacked", kMatMulNBitsWeightNotPrepacked);
    ORT_ENFORCE(weight_prepacked_ == kMatMulNBitsWeightNotPrepacked ||
                    weight_prepacked_ == kMatMulNBitsWeightPrepackedSm80 ||
                    weight_prepacked_ == kMatMulNBitsWeightPrepackedSm90,
                "weight_prepacked must be 0 (not prepacked), 1 (SM80 layout), or 2 (SM90 layout), but got ",
                weight_prepacked_);
    if (weight_prepacked_ == kMatMulNBitsWeightPrepackedSm90) {
      // The native SM90 (Hopper TMA/WGMMA) mixed-GEMM kernel requires a compute-capability 9.0
      // device and a block_size that is a multiple of the Hopper K tile (128 / sizeof(half) = 64).
      // block_size=32 is only supported by the SM80/Ampere-class kernel + GEMV path.
      ORT_ENFORCE(sm_ == 90,
                  "weight_prepacked=2 (SM90 layout) requires a compute capability 9.0 (Hopper) device, but got sm ", sm_);
      ORT_ENFORCE(block_size_ == 64 || block_size_ == 128,
                  "weight_prepacked=2 (SM90 layout) supports block_size 64 or 128 only, but got ", block_size_);
    }

    if constexpr (std::is_same<T, MLFloat16>::value || std::is_same<T, BFloat16>::value) {
      const bool prepacked = weight_prepacked_ != kMatMulNBitsWeightNotPrepacked;
      // The enable flag (session config ep.cuda.fpa_intb_gemm, else ORT_FPA_INTB_GEMM env) only
      // chooses the path for weights that are NOT prepacked. A prepacked weight is already stored in
      // the fpA_intB layout, so the choice was made at export time and cannot be turned off here.
      const int fpa_intb_option =
          ParseFpAIntBEnabled(ResolveFpAIntBConfigOrEnv(info, kConfigFpAIntBGemm, kFpAIntBGemmOption)) ? 1 : 0;
      // Route the fpA_intB path decision through the single shared eligibility function so the
      // constructor and the Level-1 EstimateMatMulNBitsWorkspace estimate can never disagree.
      const bool fpa_intb_eligible = CheckFpAIntBEligibility(
          onnxruntime::utils::ToTensorProtoElementType<T>(), N_, K_, nbits_, block_size_,
          weight_prepacked_, has_g_idx_, sm_, fpa_intb_option);
      // Note: a fused bias (input[5]) is fully supported by the fpA_intB GEMV, CUTLASS SM80/SM90
      // GEMM (EpilogueOpBias), and the tactic profiler, so bias-bearing nodes (e.g. gpt-oss
      // qkv_proj/o_proj) are eligible. Only g_idx/reorder remains unsupported by this path.
      if (fpa_intb_eligible) {
        // The CUTLASS GEMM and the GEMV decode kernel consume the same fpA_intB weight layout, so
        // enable GEMV whenever it is supported; a node cannot mix fpA_intB and legacy layouts.
        using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
        KernelType cuda_kernel_type;
        if constexpr (std::is_same<T, MLFloat16>::value) {
          cuda_kernel_type = (nbits_ == 8) ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
        } else if constexpr (std::is_same<T, BFloat16>::value) {
          cuda_kernel_type = (nbits_ == 8) ? KernelType::BF16Int8Groupwise : KernelType::BF16Int4Groupwise;
        }
        if (onnxruntime::llm::kernels::fpA_intB_gemv::is_supported(sm_, cuda_kernel_type)) {
          has_fpA_intB_gemv_ = true;
        }

        InitGemmProfiler(FpAIntBPackingSmForKernel());

        // Initial profile M buckets from session config (ep.cuda.fpa_intb_profile_m) with
        // ORT_FPA_INTB_PROFILE_M env fallback; empty -> profiler uses its default bucket set.
        std::vector<int> profile_m = WeightOnlyGroupwiseQuantGemmPluginProfiler::ParseProfileMList(
            ResolveFpAIntBConfigOrEnv(info, kConfigFpAIntBProfileM,
                                      onnxruntime::llm::kernels::weight_only::kEnvProfileM));
        gemmProfiler_->setProfileMOverride(profile_m);

        int max_m = profile_m.empty() ? onnxruntime::llm::kernels::weight_only::kDefaultProfileMaxM
                                      : profile_m.back();
        RunGemmProfile(has_fpA_intB_gemv_, 1, max_m);
        has_fpA_intB_gemm_ = true;
      }

      if (prepacked) {
        ORT_ENFORCE(has_fpA_intB_gemm_,
                    "weight_prepacked requires the fpA_intB path, but it is unsupported for this node "
                    "(check bits, block_size, N/K alignment, g_idx, and compute capability >= 7.5)");
        ORT_ENFORCE(weight_prepacked_ == RequiredWeightPrepackedFormat(),
                    "weight_prepacked=", weight_prepacked_, " does not match the format required by the selected fpA_intB kernel: ",
                    RequiredWeightPrepackedFormat());
      }
    } else {
      ORT_ENFORCE(weight_prepacked_ == kMatMulNBitsWeightNotPrepacked,
                  "weight_prepacked requires fp16 or bf16 input A so the CUDA fpA_intB path can consume input B");
    }

#ifndef NDEBUG
    printf("n=%d, k=%d, block_size=%d, bits=%d, zp_bits=%d, g_idx=%d, bias=%d, gemv=%d, gemm=%d\n",
           int(N_), int(K_), int(block_size_), int(nbits_),
           has_zero_points_ ? (is_zero_points_scale_same_type_ ? int(sizeof(T)) * 8 : int(nbits_)) : int(0),
           int(has_g_idx_ ? 1 : 0), int(has_bias_ ? 1 : 0),
           int(has_fpA_intB_gemv_), int(has_fpA_intB_gemm_));
#endif
#else
    const int64_t weight_prepacked = info.GetAttrOrDefault<int64_t>("weight_prepacked", static_cast<int64_t>(0));
    ORT_ENFORCE(weight_prepacked == 0,
                "weight_prepacked requires an ONNX Runtime build with onnxruntime_USE_FPA_INTB_GEMM=ON");
#endif
  }

  Status ComputeInternal(OpKernelContext* context) const override;
#if USE_FPA_INTB_GEMM
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

#ifndef BUILD_CUDA_EP_AS_PLUGIN
  // Level 2 (Phase-A memory roadmap, issue microsoft/onnxruntime#29775): instance-level workspace
  // estimate, callable after CreateKernels(). Uses the same constructed runner state that
  // ComputeInternal() uses, so it equals the real runtime request when the queried input-A shape
  // equals the runtime input shape. Declared only for the in-tree hierarchy; the plugin build
  // inherits the adapter OpKernel's default no-op. See DeclareWorkspaceRequirements in op_kernel.h.
  Status DeclareWorkspaceRequirements(
      gsl::span<const TensorShape> input_shapes,
      /*out*/ InlinedVector<WorkspaceRequirement>& requirements) const override;
#endif

  // TEST INSTRUMENTATION ONLY - not a runtime API. Records the workspace size the CUTLASS runner
  // requested on the most recent ComputeInternal() call so a test can verify the Level-2 estimate
  // against the real runtime request. This atomic is not correlated to a specific Run() when
  // concurrent Run()s share one kernel instance; it is only meant for this pilot's single-threaded
  // tests. Do not build anything on top of it.
  //
  // STALENESS: the value is only updated on the fpA_intB CUTLASS GEMM branch of ComputeInternal().
  // It is therefore only meaningful immediately after a call that took that branch; a subsequent
  // call that takes the GEMV (cuda-kernel) path or the non-fpA_intB path leaves it holding the old
  // value from the previous GEMM call. It is NOT reset between calls.
  size_t LastComputeWorkspaceBytes() const { return last_compute_workspace_bytes_.load(); }
#endif

 private:
#if USE_FPA_INTB_GEMM
  int FpAIntBPackingSmForKernel() const;
  int64_t RequiredWeightPrepackedFormat() const;

  void InitGemmProfiler(int sm);
  void RunGemmProfile(bool hasWeightOnlyCudaKernel, int min_m, int max_m);

  Status PrePack_B(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream, bool& is_packed);
  Status PrePack_Scale(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream);
  Status PrePack_ZeroPoint(const Tensor& tensor, AllocatorPtr alloc, cudaStream_t stream);
#endif

  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  int sm_{0};
  bool column_wise_quant_blk_{true};

  bool has_g_idx_{false};
  bool has_bias_{false};
  bool has_zero_points_{false};
  bool is_zero_points_scale_same_type_{false};
  bool force_chunked_{false};
  int64_t chunk_target_rows_{kDefaultChunkTargetRows};

#if USE_FPA_INTB_GEMM
  bool has_fpA_intB_gemv_{false};
  bool has_fpA_intB_gemm_{false};
  int64_t weight_prepacked_{kMatMulNBitsWeightNotPrepacked};

  bool is_prepacked_weight_{false};
  bool is_prepacked_scale_{false};
  bool is_prepacked_zero_point_{false};

  WeightOnlyGemmRunnerPtr weightOnlyGemmRunner_{nullptr};
  mutable GemmProfilerPtr gemmProfiler_{nullptr};
  GemmIdCore gemmId_{};

  IAllocatorUniquePtr<void> fpA_intB_weight_buffer_;
  IAllocatorUniquePtr<void> fpA_intB_scale_buffer_;
  IAllocatorUniquePtr<void> fpA_intB_zero_buffer_;

  // TEST INSTRUMENTATION ONLY (see LastComputeWorkspaceBytes above).
  mutable std::atomic<size_t> last_compute_workspace_bytes_{0};
#endif
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
