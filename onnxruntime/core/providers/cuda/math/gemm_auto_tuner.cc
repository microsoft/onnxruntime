// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/gemm_auto_tuner.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "core/common/string_utils.h"

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kWarmupRuns = 2;
constexpr int kTimedRuns = 10;
// Enough head start for the host to enqueue one timed run of a multi-launch candidate.
constexpr uint64_t kDelayPerTimedRunNs = 50'000;
constexpr size_t kMaxFlushBytes = size_t{256} << 20;

class CudaEventPool {
 public:
  CudaEventPool() = default;
  CudaEventPool(const CudaEventPool&) = delete;
  CudaEventPool& operator=(const CudaEventPool&) = delete;
  ~CudaEventPool() {
    for (cudaEvent_t event : events_) {
      cudaEventDestroy(event);
    }
  }

  Status Create(size_t count) {
    events_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      cudaEvent_t event{};
      CUDA_RETURN_IF_ERROR(cudaEventCreate(&event));
      events_.push_back(event);
    }
    return Status::OK();
  }

  cudaEvent_t operator[](size_t index) const { return events_[index]; }

 private:
  std::vector<cudaEvent_t> events_;
};

class CudaTuneGraph {
 public:
  explicit CudaTuneGraph(cudaStream_t stream) : stream_(stream) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaTuneGraph);
  ~CudaTuneGraph() {
    if (capturing_) {
      cudaStreamEndCapture(stream_, &graph_);
    }
    if (executable_) {
      cudaGraphExecDestroy(executable_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  Status Begin() {
    CUDA_RETURN_IF_ERROR(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    capturing_ = true;
    return Status::OK();
  }

  Status Instantiate() {
    const cudaError_t status = cudaStreamEndCapture(stream_, &graph_);
    capturing_ = false;
    CUDA_RETURN_IF_ERROR(status);
    CUDA_RETURN_IF_ERROR(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));
    return Status::OK();
  }

  Status Launch() {
    CUDA_RETURN_IF_ERROR(cudaGraphLaunch(executable_, stream_));
    return Status::OK();
  }

 private:
  cudaStream_t stream_{};
  cudaGraph_t graph_{};
  cudaGraphExec_t executable_{};
  bool capturing_{false};
};

}  // namespace

std::optional<bool> ParseGemmOnOffOption(const std::optional<std::string>& value, const char* name) {
  if (!value.has_value()) {
    return std::nullopt;
  }
  const std::string lowered = utils::GetLowercaseString(utils::TrimString(*value));
  if (lowered.empty()) {
    return std::nullopt;
  }
  if (lowered == "1" || lowered == "true" || lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "off") {
    return false;
  }
  ORT_THROW("Invalid value '", *value, "' for ", name, ": expected 0 or 1.");
}

GemmDispatchPolicy ResolveGemmDispatchPolicy(const std::optional<std::string>& small_n_gemv_env,
                                             const std::optional<std::string>& auto_tune_config,
                                             const std::optional<std::string>& auto_tune_env) {
  const std::optional<bool> force_small_n = ParseGemmOnOffOption(small_n_gemv_env, kSmallNGemvEnvVar);
  if (force_small_n.has_value()) {
    return *force_small_n ? GemmDispatchPolicy::kSmallNGemv : GemmDispatchPolicy::kCublas;
  }
  std::optional<bool> auto_tune = ParseGemmOnOffOption(auto_tune_config, "ep.cuda.enable_gemm_auto_tune");
  if (!auto_tune.has_value()) {
    auto_tune = ParseGemmOnOffOption(auto_tune_env, kGemmAutoTuneEnvVar);
  }
  return auto_tune.value_or(false) ? GemmDispatchPolicy::kAutoTune : GemmDispatchPolicy::kCublas;
}

const char* GemmKernelName(GemmKernel kernel) {
  switch (kernel) {
    case GemmKernel::kCublas:
      return "cublas";
    case GemmKernel::kSmallNGemv:
      return "small_n_gemv";
    case GemmKernel::kTinyGemm2:
      return "tinygemm2";
  }
  return "unknown";
}

size_t GemmTuneKeyHash::operator()(const GemmTuneKey& key) const {
  size_t seed = 0;
  auto combine = [&seed](size_t value) { seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2); };
  for (char byte : key.device_uuid) {
    combine(static_cast<unsigned char>(byte));
  }
  combine(static_cast<size_t>(key.data_type));
  combine(static_cast<size_t>(key.m));
  combine(static_cast<size_t>(key.n));
  combine(static_cast<size_t>(key.k));
  combine(static_cast<size_t>(key.small_n_vectorized));
  combine(static_cast<size_t>(key.candidates));
  combine(static_cast<size_t>(key.tinygemm2_b_is_constant));
  combine(static_cast<size_t>(key.cuda_graph_replay));
  return seed;
}

std::array<char, 16> GetDeviceUuid(const cudaDeviceProp& device_prop) {
  std::array<char, 16> uuid{};
  static_assert(sizeof(device_prop.uuid.bytes) == 16, "unexpected cudaUUID_t size");
  std::memcpy(uuid.data(), device_prop.uuid.bytes, uuid.size());
  return uuid;
}

size_t PickFastestGemmCandidate(const std::vector<float>& times_ms, float min_speedup) {
  ORT_ENFORCE(!times_ms.empty(), "No GEMM candidate was timed.");
  size_t best = 0;
  for (size_t i = 1; i < times_ms.size(); ++i) {
    if (times_ms[i] < times_ms[best]) {
      best = i;
    }
  }
  if (best != 0 && times_ms[0] < times_ms[best] * min_speedup) {
    best = 0;
  }
  return best;
}

size_t GemmAutoTuneFlushBytes(const cudaDeviceProp& device_prop) {
  const size_t l2_bytes = device_prop.l2CacheSize > 0 ? static_cast<size_t>(device_prop.l2CacheSize) : 0;
  return std::min(2 * l2_bytes, kMaxFlushBytes);
}

Status IsCudaStreamCapturing(cudaStream_t stream, bool& capturing) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &status));
  capturing = status != cudaStreamCaptureStatusNone;
  return Status::OK();
}

Status TimeGemmCandidates(cudaStream_t stream, const std::vector<GemmTuneCandidate>& candidates,
                          const GemmTuneL2State& l2, std::vector<float>& times_ms, bool cuda_graph_replay) {
  ORT_RETURN_IF(candidates.empty(), "No GEMM candidate to time.");
  bool capturing = false;
  ORT_RETURN_IF_ERROR(IsCudaStreamCapturing(stream, capturing));
  ORT_RETURN_IF(capturing, "Cannot time GEMM candidates during stream capture.");
  // Slot 0 times an empty region to measure the fixed event overhead.
  const size_t num_slots = candidates.size() + 1;

  int num_sms = 0;
  if (l2.flush_bytes > 0 || l2.hot_bytes > 0) {
    int device = 0;
    CUDA_RETURN_IF_ERROR(cudaGetDevice(&device));
    CUDA_RETURN_IF_ERROR(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  }

  for (const auto& candidate : candidates) {
    for (int i = 0; i < kWarmupRuns; ++i) {
      ORT_RETURN_IF_ERROR(candidate.run());
    }
  }

  CudaEventPool events;
  ORT_RETURN_IF_ERROR(events.Create(2 * num_slots * kTimedRuns));
  CudaTuneGraph graph(stream);

  if (cuda_graph_replay) {
    ORT_RETURN_IF_ERROR(graph.Begin());
  } else {
    ORT_RETURN_IF_ERROR(LaunchGpuDelay(stream, kDelayPerTimedRunNs * num_slots * kTimedRuns));
  }
  for (int run = 0; run < kTimedRuns; ++run) {
    for (size_t slot = 0; slot < num_slots; ++slot) {
      ORT_RETURN_IF_ERROR(LaunchL2Read(stream, l2.flush_buffer, l2.flush_bytes, num_sms));
      ORT_RETURN_IF_ERROR(LaunchL2Read(stream, l2.hot_buffer, l2.hot_bytes, num_sms));
      const size_t index = 2 * (static_cast<size_t>(run) * num_slots + slot);
      if (cuda_graph_replay) {
        CUDA_RETURN_IF_ERROR(cudaEventRecordWithFlags(events[index], stream, cudaEventRecordExternal));
      } else {
        CUDA_RETURN_IF_ERROR(cudaEventRecord(events[index], stream));
      }
      if (slot > 0) {
        ORT_RETURN_IF_ERROR(candidates[slot - 1].run());
      }
      if (cuda_graph_replay) {
        CUDA_RETURN_IF_ERROR(cudaEventRecordWithFlags(events[index + 1], stream, cudaEventRecordExternal));
      } else {
        CUDA_RETURN_IF_ERROR(cudaEventRecord(events[index + 1], stream));
      }
    }
  }
  if (cuda_graph_replay) {
    ORT_RETURN_IF_ERROR(graph.Instantiate());
    for (int replay = 0; replay <= kWarmupRuns; ++replay) {
      ORT_RETURN_IF_ERROR(graph.Launch());
    }
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  }
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(events[2 * num_slots * kTimedRuns - 1]));

  std::vector<float> medians(num_slots);
  std::vector<float> samples(kTimedRuns);
  for (size_t slot = 0; slot < num_slots; ++slot) {
    for (int run = 0; run < kTimedRuns; ++run) {
      const size_t index = 2 * (static_cast<size_t>(run) * num_slots + slot);
      CUDA_RETURN_IF_ERROR(cudaEventElapsedTime(&samples[run], events[index], events[index + 1]));
    }
    std::nth_element(samples.begin(), samples.begin() + kTimedRuns / 2, samples.end());
    medians[slot] = samples[kTimedRuns / 2];
  }
  times_ms.resize(candidates.size());
  for (size_t c = 0; c < candidates.size(); ++c) {
    times_ms[c] = std::max(medians[c + 1] - medians[0], 0.0f);
  }
  return Status::OK();
}

GemmAutoTuneCache& GemmAutoTuneCache::Instance() {
  static GemmAutoTuneCache cache;
  return cache;
}

std::optional<GemmKernel> GemmAutoTuneCache::Lookup(const GemmTuneKey& key) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = kernels_.find(key);
  if (it == kernels_.end()) {
    return std::nullopt;
  }
  return it->second;
}

GemmKernel GemmAutoTuneCache::Insert(const GemmTuneKey& key, GemmKernel kernel) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return kernels_.emplace(key, kernel).first->second;
}

size_t GemmAutoTuneCache::Size() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return kernels_.size();
}

void GemmAutoTuneCache::Clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  kernels_.clear();
}

Status TuneGemmKernel(const GemmTuneKey& key, cudaStream_t stream, const std::vector<GemmTuneCandidate>& candidates,
                      const GemmTuneL2State& l2, GemmKernel& selected) {
  GemmAutoTuneCache& cache = GemmAutoTuneCache::Instance();
  std::lock_guard<std::mutex> tuning_lock(cache.TuningMutex());
  // Another thread may have tuned this key while we waited.
  if (const auto cached = cache.Lookup(key)) {
    selected = *cached;
    return Status::OK();
  }

  std::vector<float> times_ms;
  ORT_RETURN_IF_ERROR(TimeGemmCandidates(stream, candidates, l2, times_ms, key.cuda_graph_replay));
  const size_t best = PickFastestGemmCandidate(times_ms, kGemmAutoTuneMinSpeedup);
  selected = cache.Insert(key, candidates[best].kernel);

  std::ostringstream timings;
  for (size_t i = 0; i < candidates.size(); ++i) {
    timings << (i == 0 ? "" : ", ") << GemmKernelName(candidates[i].kernel) << "=" << times_ms[i] * 1000.0f << "us";
  }
  LOGS_DEFAULT(VERBOSE) << "GEMM auto-tune " << (key.data_type == GemmDataType::kFloat16 ? "fp16" : "bf16")
                        << " M=" << key.m << " N=" << key.n << " K=" << key.k
                        << " timing=" << (key.cuda_graph_replay ? "cuda_graph_replay" : "stream") << ": " << timings.str()
                        << " -> " << GemmKernelName(selected);
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
