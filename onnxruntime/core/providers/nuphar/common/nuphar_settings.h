// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/settings.h"

namespace onnxruntime {

// forward declaration
struct NupharExecutionProviderInfo;

namespace nuphar {
constexpr static const char* kNupharDumpPartition = "nuphar_dump_partition";
constexpr static const char* kNupharDumpFusedNodes = "nuphar_dump_fused_nodes";
constexpr static const char* kNupharMatmulExec = "nuphar_matmul_exec";
constexpr static const char* kNupharCachePath = "nuphar_cache_path";
constexpr static const char* kNupharCacheVersion = "nuphar_cache_version";
constexpr static const char* kNupharCacheSoName = "nuphar_cache_so_name";
constexpr static const char* kNupharCacheModelChecksum = "nuphar_cache_model_checksum";
constexpr static const char* kNupharCacheForceNoJIT = "nuphar_cache_force_no_jit";
// force to use IMatMulExternMKL/IMatMul16ExternMKL
constexpr static const char* kNupharIMatMulForceMkl = "nuphar_imatmul_force_mkl";

constexpr static const char* kNupharMatMulExec_ExternCpu = "extern_cpu";

constexpr static const char* kNupharForceNoTensorize = "nuphar_force_no_tensorize";

constexpr static const char* kNupharTensorize_IGEMM_Tile_M = "nuphar_igemm_tile_m";
constexpr static const char* kNupharTensorize_IGEMM_Tile_N = "nuphar_igemm_tile_n";
constexpr static const char* kNupharTensorize_IGEMM_Tile_K = "nuphar_igemm_tile_k";

constexpr static const char* kNupharTensorize_IGEMM_Permute = "nuphar_igemm_permute";
constexpr static const char* kNupharTensorize_IGEMM_Permute_Inner = "inner";
constexpr static const char* kNupharTensorize_IGEMM_Permute_Outer = "outer";
constexpr static const char* kNupharTensorize_IGEMM_Permute_All = "all";
constexpr static const char* kNupharTensorize_IGEMM_Split_Last_Tile = "nuphar_igemm_split_last_tile";

constexpr static const char* kNupharFastMath = "nuphar_fast_math";                         // fast math
constexpr static const char* kNupharFastMath_Polynormial = "polynormial_math";             // generic polynormial fast math for exp and log
constexpr static const char* kNupharFastMath_ShortPolynormial = "short_polynormial_math";  // generic shorter polynormial fast math for exp and log

constexpr static const char* kNupharFastActivation = "nuphar_fast_activation";  // fast activation
constexpr static const char* kNupharActivations_DeepCpu = "deep_cpu_activation";

// Option to control nuphar code generation target (avx / avx2 / avx512)
constexpr static const char* kNupharCodeGenTarget = "nuphar_codegen_target";

// cache version number (MAJOR.MINOR.PATCH) following https://semver.org/
// 1. MAJOR version when you make incompatible changes that old cache files no longer work,
// 2. MINOR version when you add functionality in a backwards - compatible manner, and
// 3. PATCH version when you make backwards - compatible bug fixes.
// NOTE this version needs to be updated when generated code may change
constexpr static const char* kNupharCacheVersion_Current = "1.0.0";

constexpr static const char* kNupharCacheSoName_Default = "jit.so";

void CreateNupharCodeGenSettings(const NupharExecutionProviderInfo& info);

}  // namespace nuphar
}  // namespace onnxruntime
