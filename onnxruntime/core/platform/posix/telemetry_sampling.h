// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string_view>

namespace onnxruntime::telemetry_internal {

// Percentage of model-session events retained. Process-scoped events remain at 100%.
// 1DS popSample is metadata only; ShouldSampleSession performs the actual client-side sampling.
inline constexpr double kModelSessionSampleRatePercent = 100.0;

static_assert(kModelSessionSampleRatePercent >= 0.0 &&
              kModelSessionSampleRatePercent <= 100.0);

inline uint64_t HashSamplingKey(std::string_view app_session_guid, uint32_t session_id) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char c : app_session_guid) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  for (unsigned int shift = 0; shift < 32; shift += 8) {
    hash ^= static_cast<unsigned char>(session_id >> shift);
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline bool ShouldSampleSession(std::string_view app_session_guid, uint32_t session_id,
                                double sample_rate_percent = kModelSessionSampleRatePercent) {
  if (!(sample_rate_percent > 0.0)) {
    return false;
  }
  if (sample_rate_percent >= 100.0) {
    return true;
  }

  // One million buckets support rates down to 0.0001% while keeping the decision stable for every
  // event in a model session. The random process GUID prevents sequential session IDs from biasing
  // the sample across processes.
  constexpr uint64_t kBucketCount = 1'000'000;
  const auto threshold = static_cast<uint64_t>(
      static_cast<long double>(sample_rate_percent) * kBucketCount / 100.0L);
  return HashSamplingKey(app_session_guid, session_id) % kBucketCount < threshold;
}

}  // namespace onnxruntime::telemetry_internal
