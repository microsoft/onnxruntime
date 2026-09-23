// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Persistent, hardware-keyed tactic cache for weight-only GEMM autotuning.
//
// This is the shared foundation described in
// docs/contrib_ops/cuda/gemm_profiler_cache.md (Phase 1). It provides:
//   * HardwareSignature: GPU model / SM / toolkit / build guard used to name and
//     validate cache files so tactics are never reused across incompatible setups.
//   * (De)serialization of CutlassGemmConfig to/from a fixed set of integer columns.
//   * TSV read/write helpers with percent-encoding for free-text fields.
//   * MatMulNBitsTacticCache: a typed, disk-backed cache keyed by the fpA_intB
//     packed problem shape and quantization mode, with file locking, atomic write,
//     and in-memory merge.
//
// Persistence is opt-in. When no cache directory or explicit prefix is configured,
// the cache stays in memory only and never touches disk.
//
// The declarations below are always available in CUDA builds; the implementation in
// gemm_tactic_cache.cc is compiled only when USE_FPA_INTB_GEMM is defined and is only
// referenced from the fpA_intB profiling paths that are themselves so guarded.
//
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/status.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"

namespace onnxruntime::llm::gemm_cache {

using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;

// Cache file format version. Bump when the on-disk schema changes incompatibly.
constexpr const char* kCacheMagic = "ort_cuda_gemm_tactic_cache";
constexpr const char* kCacheFormatVersion = "v1";

constexpr const char* kTableMatMulNBits = "matmulnbits_fpa_intb";

// Environment variables (see docs section 9).
constexpr const char* kEnvCacheDir = "ORT_CUDA_GEMM_TACTIC_CACHE_DIR";
constexpr const char* kEnvCachePrefix = "ORT_CUDA_GEMM_TACTIC_CACHE_PREFIX";

// Session-option config keys (equivalents of the env vars). These can be set via
// SessionOptions::AddConfigEntry, or through onnxruntime-genai's session_options in
// genai_config.json (any unrecognized session_options key is forwarded to AddConfigEntry).
constexpr const char* kSessionConfigCacheDir = "ep.cuda.gemm_tactic_cache_dir";
constexpr const char* kSessionConfigCachePrefix = "ep.cuda.gemm_tactic_cache_prefix";

// Hardware / build signature used both to name cache files and as a stored guard.
// Reuse is rejected on mismatch of any strict field (see StrictMatches).
struct HardwareSignature {
  std::string device_name;       // cudaDeviceProp.name, e.g. "NVIDIA A100-SXM4-80GB"
  int sm = 0;                    // major*10 + minor, e.g. 80, 89, 90
  int multiprocessor_count = 0;  // diagnostic only
  int cuda_runtime = 0;          // CUDART_VERSION
  int cuda_driver = 0;           // cudaDriverGetVersion(), diagnostic only
  std::string ort_version;       // ORT_VERSION (strict-match field)
  std::string ort_git_commit;    // parsed from ORT_BUILD_INFO or "unknown" (diagnostic only)
  std::string ort_build_config;  // "Release" / "Debug" (diagnostic only)

  // Computes the signature for the current CUDA device and build.
  static HardwareSignature Compute();

  // Strict reuse guard: device_name, sm, cuda_runtime, and ort_version must all match.
  // multiprocessor_count, cuda_driver, ort_git_commit, and ort_build_config are recorded for
  // diagnostics only and are NOT part of the guard: a tactic is just a config selected among
  // getConfigs(), so a cross-commit/cross-build reuse can at worst pick a slightly suboptimal
  // (never incorrect) tactic, and loadPersistentCache re-validates every CUTLASS tactic against
  // the current runner and drops any that no longer applies.
  bool StrictMatches(const HardwareSignature& other) const;

  // A filesystem-safe token derived from device_name + sm used as a default cache prefix.
  std::string FilePrefixToken() const;
};

// TSV free-text encoding: percent-encodes '%', '\t', '\n', '\r' so a field never
// breaks the tab/newline structure. Decode reverses it.
std::string TsvEncode(const std::string& s);
std::string TsvDecode(const std::string& s);

// Fixed integer column layout for a (possibly-absent) tactic, mirroring the doc:
//   valid_config sm_version tile80 tile90 tile100 tile120
//   split_k_style split_k stages cluster mainloop epilogue tma enable_cuda_kernel
constexpr int kNumConfigColumns = 14;

// Appends the 14 config columns (as decimal strings) for `config` to `row`.
// A missing tactic (std::nullopt) is written with valid_config=0 and defaults elsewhere.
void AppendConfigColumns(std::vector<std::string>& row, const std::optional<CutlassGemmConfig>& config);

// Parses the 14 config columns starting at `columns[begin]`. Returns std::nullopt
// (outer) if the columns are malformed; the inner optional is std::nullopt when
// valid_config==0 (a profiled bucket with no valid tactic).
std::optional<std::optional<CutlassGemmConfig>> ParseConfigColumns(
    const std::vector<std::string>& columns, size_t begin);

// Problem key for MatMulNBits fpA_intB. Many nodes with the same shape collapse to
// one entry. n_16b is the value used by GemmIdCore (N after casting int weight to fp16).
struct MatMulNBitsKey {
  int n_16b = 0;
  int k = 0;
  std::string activation_dtype;  // "half" / "bfloat16"
  std::string weight_type;       // "uint4b_t" / "uint8_t"
  int bits = 0;
  int block_size = 0;
  bool has_zero_points = false;
  std::string zero_point_dtype;  // e.g. "uint4b_t" / activation dtype / "none"
  bool gemv_enabled = false;
  int packing_sm = 0;

  bool operator==(const MatMulNBitsKey& o) const;
};

struct MatMulNBitsKeyHash {
  std::size_t operator()(const MatMulNBitsKey& k) const;
};

// Disk-backed tactic cache for MatMulNBits fpA_intB. Thread-safe.
//
// Lookup/merge semantics:
//   * Get(key, m) -> outer nullopt means "not cached, must profile"; inner optional
//     is the tactic (inner nullopt means "profiled, no valid tactic").
//   * Put(key, m, config) records a bucket in memory.
//   * Flush() atomically merges the in-memory table into the on-disk file under a
//     file lock so concurrent sessions tuning different shapes do not lose updates.
class MatMulNBitsTacticCache {
 public:
  // Returns a configured cache if persistence is enabled, otherwise nullptr (callers keep the
  // in-process-only behavior). Resolution order for the file location:
  //   1. explicit session-config `prefix`/`dir` arguments (when non-empty),
  //   2. ORT_CUDA_GEMM_TACTIC_CACHE_PREFIX / ORT_CUDA_GEMM_TACTIC_CACHE_DIR env vars.
  // A non-empty prefix wins over a dir. The returned cache has already attempted to load
  // matching rows from disk.
  static std::shared_ptr<MatMulNBitsTacticCache> MaybeCreate(const std::string& config_dir = "",
                                                             const std::string& config_prefix = "");

  // Test/tooling entry point: build a cache bound to an explicit file path.
  MatMulNBitsTacticCache(std::string file_path, HardwareSignature signature);

  // Ordered map from M-bucket to its (possibly-absent) tactic for one problem key.
  using BucketMap = std::map<int, std::optional<CutlassGemmConfig>>;

  const std::string& FilePath() const { return file_path_; }
  const HardwareSignature& Signature() const { return signature_; }

  std::optional<std::optional<CutlassGemmConfig>> Get(const MatMulNBitsKey& key, int m_bucket) const;

  // Returns all cached M-buckets for `key` (empty if the key is not cached).
  BucketMap GetAll(const MatMulNBitsKey& key) const;

  void Put(const MatMulNBitsKey& key, int m_bucket, const std::optional<CutlassGemmConfig>& config);

  // Loads rows from file_path_ whose header signature strictly matches signature_.
  // A missing file or mismatched signature leaves the in-memory table untouched and
  // returns OK (treated as an empty/rejected cache).
  onnxruntime::common::Status Load();

  // Atomically merges the in-memory table with the current on-disk file and writes
  // it back. No-op (OK) if there is nothing to persist.
  onnxruntime::common::Status Flush();

 private:
  onnxruntime::common::Status WriteAllLocked(
      const std::unordered_map<MatMulNBitsKey, BucketMap, MatMulNBitsKeyHash>& table) const;

  std::string file_path_;
  HardwareSignature signature_;

  mutable std::mutex mutex_;
  std::unordered_map<MatMulNBitsKey, BucketMap, MatMulNBitsKeyHash> table_;
  bool dirty_ = false;
  size_t generation_ = 0;
};

}  // namespace onnxruntime::llm::gemm_cache
