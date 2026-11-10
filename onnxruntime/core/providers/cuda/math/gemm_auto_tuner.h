// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// ORT_ENABLE_SMALL_N_GEMV=1/0 forces the small-N GEMV on/off for eligible shapes and bypasses tuning.
constexpr const char* kSmallNGemvEnvVar = "ORT_ENABLE_SMALL_N_GEMV";
// Fallback for session config ep.cuda.enable_gemm_auto_tune.
constexpr const char* kGemmAutoTuneEnvVar = "ORT_CUDA_GEMM_AUTO_TUNE";
constexpr const char* kGemmGraphReplayTuneEnvVar = "ORT_CUDA_GEMM_GRAPH_REPLAY_TUNING";

// A candidate other than the default (cuBLAS) must be at least this much faster to be selected.
constexpr float kGemmAutoTuneMinSpeedup = 1.05f;

enum class GemmDispatchPolicy : uint8_t {
  kCublas,      // always cuBLAS
  kSmallNGemv,  // small-N GEMV whenever the shape is eligible
  kAutoTune,    // per shape, the fastest measured kernel
};

// Parses an on/off option: "1"/"true"/"on" or "0"/"false"/"off" (case-insensitive). Unset or empty
// yields nullopt; any other value throws.
std::optional<bool> ParseGemmOnOffOption(const std::optional<std::string>& value, const char* name);

// Precedence: the small-N env var forces a kernel, then the session config, then the auto-tune env var.
GemmDispatchPolicy ResolveGemmDispatchPolicy(const std::optional<std::string>& small_n_gemv_env,
                                             const std::optional<std::string>& auto_tune_config,
                                             const std::optional<std::string>& auto_tune_env);

enum class GemmKernel : uint8_t {
  kCublas = 0,
  kSmallNGemv = 1,
  kTinyGemm2 = 2,
};

constexpr uint8_t GemmKernelBit(GemmKernel kernel) { return static_cast<uint8_t>(1u << static_cast<int>(kernel)); }

const char* GemmKernelName(GemmKernel kernel);

enum class GemmDataType : uint8_t {
  kFloat16,
  kBFloat16,
};

struct GemmTuneKey {
  std::array<char, 16> device_uuid{};
  GemmDataType data_type{GemmDataType::kFloat16};
  int m{0};
  int n{0};
  int k{0};
  // Operand alignment picks a different small-N kernel, so it is part of the key.
  bool small_n_vectorized{false};
  // GemmKernelBit of every candidate that was eligible, since alignment can change the set.
  uint8_t candidates{0};
  bool tinygemm2_b_is_constant{false};
  bool cuda_graph_replay{false};

  bool operator==(const GemmTuneKey& other) const {
    return device_uuid == other.device_uuid && data_type == other.data_type && m == other.m && n == other.n &&
           k == other.k && small_n_vectorized == other.small_n_vectorized && candidates == other.candidates &&
           tinygemm2_b_is_constant == other.tinygemm2_b_is_constant && cuda_graph_replay == other.cuda_graph_replay;
  }
};

struct GemmTuneKeyHash {
  size_t operator()(const GemmTuneKey& key) const;
};

std::array<char, 16> GetDeviceUuid(const cudaDeviceProp& device_prop);

struct GemmTuneCandidate {
  GemmKernel kernel;
  // Enqueues one run on the stream being tuned.
  std::function<Status()> run;
};

// Index of the fastest candidate. Candidate 0 is the default and is replaced only by one that is at
// least `min_speedup` times faster.
size_t PickFastestGemmCandidate(const std::vector<float>& times_ms, float min_speedup);

// Scratch size used to evict L2 between timed runs.
size_t GemmAutoTuneFlushBytes(const cudaDeviceProp& device_prop);

Status IsCudaStreamCapturing(cudaStream_t stream, bool& capturing);

// Scratch used to put L2 in a decode-like state before each timed run: everything is evicted by reading
// `flush_buffer`, then `hot_buffer` (the activation, which the previous op just produced) is read back in.
struct GemmTuneL2State {
  void* flush_buffer{nullptr};
  size_t flush_bytes{0};
  const void* hot_buffer{nullptr};
  size_t hot_bytes{0};
};

// Median time per run of every candidate from `l2`, using stream launches or CUDA graph replay.
// Replay captures L2 conditioning outside the event-timed regions; the empty region's overhead is
// subtracted in both modes. Synchronizes the stream and rejects an already-capturing stream.
Status TimeGemmCandidates(cudaStream_t stream, const std::vector<GemmTuneCandidate>& candidates,
                          const GemmTuneL2State& l2, std::vector<float>& times_ms, bool cuda_graph_replay = false);

// Process-wide map from shape/device and eligible launch configuration to the selected kernel.
// The first insertion wins, so every caller with the same tuning key runs the same kernel.
class GemmAutoTuneCache {
 public:
  static GemmAutoTuneCache& Instance();

  std::optional<GemmKernel> Lookup(const GemmTuneKey& key) const;
  GemmKernel Insert(const GemmTuneKey& key, GemmKernel kernel);
  size_t Size() const;
  void Clear();

  // Serializes measurements so concurrent tuning does not distort timings.
  std::mutex& TuningMutex() { return tuning_mutex_; }

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<GemmTuneKey, GemmKernel, GemmTuneKeyHash> kernels_;
  std::mutex tuning_mutex_;
};

// Times `candidates` (candidates[0] is the default), caches the winner for `key`, and returns it.
// The caller must not call this while `stream` is capturing a CUDA graph.
Status TuneGemmKernel(const GemmTuneKey& key, cudaStream_t stream, const std::vector<GemmTuneCandidate>& candidates,
                      const GemmTuneL2State& l2, GemmKernel& selected);

// Spins the stream for `nanoseconds` so later launches queue up behind it (defined in gemm_auto_tuner_impl.cu).
Status LaunchGpuDelay(cudaStream_t stream, uint64_t nanoseconds);

// Reads `bytes` of `buffer` through L2 (evicting other lines when the buffer is larger than L2).
Status LaunchL2Read(cudaStream_t stream, const void* buffer, size_t bytes, int num_sms);

}  // namespace cuda
}  // namespace onnxruntime
