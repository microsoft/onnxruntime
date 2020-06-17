/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/// \file
/// \brief The internal representation of user-provided settings.

#ifndef MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
#define MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H

#include <chrono>
#include <cmath>
#include <string>

#include "logging.h"
#include "test_settings.h"

namespace mlperf {

namespace logging {
class AsyncSummary;
}

namespace loadgen {

using AsyncSummary = logging::AsyncSummary;

std::string ToString(TestScenario scenario);
std::string ToString(TestMode mode);

/// \brief takes the user-friendly TestSettings and normalizes it
/// for consumption by the loadgen.
/// \details It does things like remove scenario-specific naming and introduce
/// the concept of target_duration used to pre-generate queries.
struct TestSettingsInternal {
  explicit TestSettingsInternal(const TestSettings &requested_settings,
                                size_t qsl_performance_sample_count);
  void LogEffectiveSettings() const;
  void LogAllSettings() const;
  void LogSummary(AsyncSummary &summary) const;

  const TestSettings requested;
  const TestScenario scenario;  // Copied here for convenience.
  const TestMode mode;          // Copied here for convenience.

  uint64_t samples_per_query;
  double target_qps;
  std::chrono::nanoseconds target_latency{0};
  double target_latency_percentile;  // Single, multistream, and server modes.
  uint64_t max_async_queries;

  // Target duration is used to generate queries of a minimum duration before
  // the test run.
  std::chrono::milliseconds target_duration{0};

  // Min duration/query_count/sample_count are used to validate the test
  // duration at the end of the run.
  std::chrono::milliseconds min_duration{0};
  std::chrono::milliseconds max_duration{0};
  uint64_t min_query_count;
  uint64_t max_query_count;
  uint64_t min_sample_count;  // Offline only.

  uint64_t qsl_rng_seed;
  uint64_t sample_index_rng_seed;
  uint64_t schedule_rng_seed;
  uint64_t accuracy_log_rng_seed;
  double accuracy_log_probability;
  bool print_timestamps;
  bool performance_issue_unique;
  bool performance_issue_same;
  uint64_t performance_issue_same_index;
  uint64_t performance_sample_count;
};

/// \brief A namespace of collections of FindPeakPerformance helper functions,
/// mainly about binary search.
namespace find_peak_performance {

constexpr char const *kNotSupportedMsg =
    "Finding peak performance is only supported in MultiStream, "
    "MultiStreamFree, and Server scenarios.";

template <TestScenario scenario>
TestSettingsInternal MidOfBoundaries(
    const TestSettingsInternal &lower_bound_settings,
    const TestSettingsInternal &upper_bound_settings) {
  TestSettingsInternal mid_settings = lower_bound_settings;
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    assert(lower_bound_settings.samples_per_query <
           upper_bound_settings.samples_per_query);
    mid_settings.samples_per_query = lower_bound_settings.samples_per_query +
                                     (upper_bound_settings.samples_per_query -
                                      lower_bound_settings.samples_per_query) /
                                         2;
  } else if (scenario == TestScenario::Server) {
    assert(lower_bound_settings.target_qps < upper_bound_settings.target_qps);
    mid_settings.target_qps =
        lower_bound_settings.target_qps +
        (upper_bound_settings.target_qps - lower_bound_settings.target_qps) / 2;
  } else {
    LogDetail([](AsyncDetail &detail) { detail(kNotSupportedMsg); });
  }
  return mid_settings;
}

template <TestScenario scenario>
bool IsFinished(const TestSettingsInternal &lower_bound_settings,
                const TestSettingsInternal &upper_bound_settings) {
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    return lower_bound_settings.samples_per_query + 1 >=
           upper_bound_settings.samples_per_query;
  } else if (scenario == TestScenario::Server) {
    uint8_t precision = lower_bound_settings.requested
                            .server_find_peak_qps_decimals_of_precision;
    double l =
        std::floor(lower_bound_settings.target_qps * std::pow(10, precision));
    double u =
        std::floor(upper_bound_settings.target_qps * std::pow(10, precision));
    return l + 1 >= u;
  } else {
    LogDetail([](AsyncDetail &detail) { detail(kNotSupportedMsg); });
    return true;
  }
}

template <TestScenario scenario>
std::string ToStringPerformanceField(const TestSettingsInternal &settings) {
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    return std::to_string(settings.samples_per_query);
  } else if (scenario == TestScenario::Server) {
    return std::to_string(settings.target_qps);
  } else {
    LogDetail([](AsyncDetail &detail) { detail(kNotSupportedMsg); });
    return ToString(settings.scenario);
  }
}

template <TestScenario scenario>
void WidenPerformanceField(TestSettingsInternal *settings) {
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    settings->samples_per_query = settings->samples_per_query * 2;
  } else if (scenario == TestScenario::Server) {
    settings->target_qps =
        settings->target_qps *
        (1 + settings->requested.server_find_peak_qps_boundary_step_size);
  } else {
    LogDetail([](AsyncDetail &detail) { detail(kNotSupportedMsg); });
  }
}

}  // namespace find_peak_performance
}  // namespace loadgen
}  // namespace mlperf

#endif  // MLPERF_LOADGEN_TEST_SETTINGS_INTERNAL_H
