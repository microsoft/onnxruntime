// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Platform-independent aggregation helpers for the WebGPU plugin EP telemetry events.
// Kept header-only and free of ORT/Dawn dependencies so they can be unit tested from the provider test binary.

namespace onnxruntime {
namespace webgpu {
namespace ep {
namespace telemetry {

// Operator domains whose operator names are defined by ONNX or ONNX Runtime. Operators from any other domain
// are model-defined strings and are reported only as a "custom" bucket.
inline bool IsWellKnownOpDomain(std::string_view domain) {
  return domain.empty() ||
         domain == "ai.onnx" ||
         domain == "ai.onnx.ml" ||
         domain == "com.microsoft" ||
         domain == "com.microsoft.experimental" ||
         domain == "com.microsoft.nchwc" ||
         domain == "com.ms.internal.nhwc";
}

inline std::string OpTypeTelemetryKey(std::string_view domain, std::string_view op_type) {
  if (!IsWellKnownOpDomain(domain)) {
    return "custom";
  }
  std::string key{domain.empty() ? std::string_view{"ai.onnx"} : domain};
  key += ':';
  key += op_type;
  return key;
}

// Counts occurrences of operator types with a bounded number of distinct keys.
class OpTypeCounter {
 public:
  static constexpr size_t kMaxDistinctKeys = 128;
  static constexpr std::string_view kOverflowKey = "other";

  void Add(std::string_view domain, std::string_view op_type) {
    std::string key = OpTypeTelemetryKey(domain, op_type);
    auto it = counts_.find(key);
    if (it == counts_.end()) {
      if (counts_.size() >= kMaxDistinctKeys) {
        key = std::string{kOverflowKey};
      }
      it = counts_.emplace(std::move(key), 0).first;
    }
    ++it->second;
  }

  size_t DistinctCount() const { return counts_.size(); }
  bool Empty() const { return counts_.empty(); }

  // Formats up to max_entries "key=count" pairs, most frequent first (ties broken by key), separated by ','.
  std::string ToString(size_t max_entries) const {
    std::vector<std::pair<std::string_view, uint32_t>> entries;
    entries.reserve(counts_.size());
    for (const auto& [key, count] : counts_) {
      entries.emplace_back(key, count);
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.second != rhs.second ? lhs.second > rhs.second : lhs.first < rhs.first;
    });

    std::string result;
    const size_t count = std::min(max_entries, entries.size());
    for (size_t i = 0; i < count; ++i) {
      if (i != 0) {
        result += ',';
      }
      result += entries[i].first;
      result += '=';
      result += std::to_string(entries[i].second);
    }
    return result;
  }

 private:
  std::map<std::string, uint32_t, std::less<>> counts_;
};

// Emission schedule for periodic summaries: starts at 2 seconds and doubles up to 10 minutes, matching the
// host's RuntimePerf/EpDeviceUsage cadence.
class SummaryIntervalScheduler {
 public:
  static constexpr int64_t kInitialIntervalUs = 2LL * 1000 * 1000;
  static constexpr int64_t kMaxIntervalUs = 10LL * 60 * 1000 * 1000;

  bool IsDue(int64_t elapsed_since_last_emit_us) const { return elapsed_since_last_emit_us >= interval_us_; }

  void OnEmitted() { interval_us_ = std::min(interval_us_ * 2, kMaxIntervalUs); }

  int64_t IntervalUs() const { return interval_us_; }

 private:
  int64_t interval_us_ = kInitialIntervalUs;
};

// Per-EP-instance run counters accumulated between WebGpuRunSummary events.
struct RunCounters {
  uint32_t runs = 0;
  uint32_t run_start_failures = 0;
  uint32_t run_end_failures = 0;
  uint32_t graph_replays = 0;
  uint32_t graph_replay_failures = 0;
  int64_t total_run_duration_us = 0;
  int64_t max_run_duration_us = 0;

  void AddRun(int64_t duration_us, bool run_end_succeeded) {
    ++runs;
    total_run_duration_us += duration_us;
    max_run_duration_us = std::max(max_run_duration_us, duration_us);
    if (!run_end_succeeded) {
      ++run_end_failures;
    }
  }

  void AddGraphReplay(bool succeeded) {
    ++graph_replays;
    if (!succeeded) {
      ++graph_replay_failures;
    }
  }

  bool HasData() const { return runs != 0 || run_start_failures != 0 || graph_replays != 0; }
};

}  // namespace telemetry
}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
