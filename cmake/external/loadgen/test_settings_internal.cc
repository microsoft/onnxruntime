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

#include "test_settings_internal.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "logging.h"
#include "utils.h"

namespace mlperf {
namespace loadgen {

TestSettingsInternal::TestSettingsInternal(
    const TestSettings &requested_settings, size_t qsl_performance_sample_count)
    : requested(requested_settings),
      scenario(requested.scenario),
      mode(requested.mode),
      samples_per_query(1),
      target_qps(1),
      max_async_queries(0),
      target_duration(std::chrono::milliseconds(requested.min_duration_ms)),
      min_duration(std::chrono::milliseconds(requested.min_duration_ms)),
      max_duration(std::chrono::milliseconds(requested.max_duration_ms)),
      min_query_count(requested.min_query_count),
      max_query_count(requested.max_query_count),
      min_sample_count(0),
      qsl_rng_seed(requested.qsl_rng_seed),
      sample_index_rng_seed(requested.sample_index_rng_seed),
      schedule_rng_seed(requested.schedule_rng_seed),
      accuracy_log_rng_seed(requested.accuracy_log_rng_seed),
      accuracy_log_probability(requested.accuracy_log_probability),
      print_timestamps(requested.print_timestamps),
      performance_issue_unique(requested.performance_issue_unique),
      performance_issue_same(requested.performance_issue_same),
      performance_issue_same_index(requested.performance_issue_same_index),
      performance_sample_count(0) {
  // Target QPS, target latency, and max_async_queries.
  switch (requested.scenario) {
    case TestScenario::SingleStream:
      target_qps = static_cast<double>(std::nano::den) /
                   requested.single_stream_expected_latency_ns;
      max_async_queries = 1;
      target_latency_percentile =
          requested.single_stream_target_latency_percentile;
      break;
    case TestScenario::MultiStream: {
      max_async_queries = requested.multi_stream_max_async_queries;
      target_qps = requested.multi_stream_target_qps;
      double target_latency_seconds =
          max_async_queries / requested.multi_stream_target_qps;
      target_latency =
          SecondsToDuration<std::chrono::nanoseconds>(target_latency_seconds);
      target_latency_percentile =
          requested.multi_stream_target_latency_percentile;
      break;
    }
    case TestScenario::MultiStreamFree:
      max_async_queries = requested.multi_stream_max_async_queries;
      target_qps = requested.multi_stream_target_qps;
      target_latency =
          std::chrono::nanoseconds(requested.multi_stream_target_latency_ns);
      target_latency_percentile =
          requested.multi_stream_target_latency_percentile;
      break;
    case TestScenario::Server:
      if (requested.server_target_qps >= 0.0) {
        target_qps = requested.server_target_qps;
      } else {
        LogDetail([server_target_qps = requested.server_target_qps,
                   target_qps = target_qps](AsyncDetail &detail) {
          detail.Error("Invalid value for server_target_qps requested.",
                       "requested", server_target_qps, "using", target_qps);
        });
      }
      target_latency =
          std::chrono::nanoseconds(requested.server_target_latency_ns);
      target_latency_percentile = requested.server_target_latency_percentile;
      max_async_queries = requested.server_max_async_queries;
      break;
    case TestScenario::Offline:
      if (requested.offline_expected_qps >= 0.0) {
        target_qps = requested.offline_expected_qps;
      } else {
        LogDetail([offline_expected_qps = requested.offline_expected_qps,
                   target_qps = target_qps](AsyncDetail &detail) {
          detail.Error("Invalid value for offline_expected_qps requested.",
                       "requested", offline_expected_qps, "using", target_qps);
        });
      }
      max_async_queries = 1;
      break;
  }

  // Performance Sample Count: TestSettings override QSL ->
  // PerformanceSampleCount
  performance_sample_count = (requested.performance_sample_count_override == 0)
                                 ? qsl_performance_sample_count
                                 : requested.performance_sample_count_override;

  // Samples per query.
  if (requested.scenario == TestScenario::MultiStream ||
      requested.scenario == TestScenario::MultiStreamFree) {
    samples_per_query = requested.multi_stream_samples_per_query;
  }

  // In the offline scenario, coalesce all queries into a single query.
  if (requested.scenario == TestScenario::Offline) {
    // TODO: Should the spec require a max duration for large query counts?
    // kSlack is used to make sure we generate enough samples for the SUT
    // to take longer than than the minimum test duration required by the
    // MLPerf spec.
    constexpr double kSlack = 1.1;
    uint64_t target_sample_count =
        kSlack * DurationToSeconds(target_duration) * target_qps;
    samples_per_query =
        (requested.performance_issue_unique || requested.performance_issue_same)
            ? performance_sample_count
            : std::max<uint64_t>(min_query_count, target_sample_count);
    min_query_count = 1;
    target_duration = std::chrono::milliseconds(0);
  }

  min_sample_count = min_query_count * samples_per_query;

  // Validate TestSettings
  if (requested.performance_issue_same &&
      (requested.performance_issue_same_index >= performance_sample_count)) {
    LogDetail(
        [performance_issue_same_index = requested.performance_issue_same_index,
         performance_sample_count =
             performance_sample_count](AsyncDetail &detail) {
          detail.Error(
              "Sample Idx to be repeated in performance_issue_same mode"
              " cannot be greater than loaded performance_sample_count.",
              "performance_issue_same_index", performance_issue_same_index,
              "performance_sample_count", performance_sample_count);
        });
  }

  if (requested.performance_issue_unique && requested.performance_issue_same) {
    LogDetail([performance_issue_unique = requested.performance_issue_unique,
               performance_issue_same =
                   requested.performance_issue_same](AsyncDetail &detail) {
      detail.Error(
          "Performance_issue_unique and performance_issue_same, both"
          " cannot be true at the same time.",
          "performance_issue_unique", performance_issue_unique,
          "performance_issue_same", performance_issue_same);
    });
  }
}

std::string ToString(TestScenario scenario) {
  switch (scenario) {
    case TestScenario::SingleStream:
      return "Single Stream";
    case TestScenario::MultiStream:
      return "Multi Stream";
    case TestScenario::MultiStreamFree:
      return "Multi Stream Free";
    case TestScenario::Server:
      return "Server";
    case TestScenario::Offline:
      return "Offline";
  }
  assert(false);
  return "InvalidScenario";
}

std::string ToString(TestMode mode) {
  switch (mode) {
    case TestMode::SubmissionRun:
      return "Submission";
    case TestMode::AccuracyOnly:
      return "Accuracy";
    case TestMode::PerformanceOnly:
      return "Performance";
    case TestMode::FindPeakPerformance:
      return "Find Peak Performance";
  }
  assert(false);
  return "InvalidMode";
}

void LogRequestedTestSettings(const TestSettings &s) {
  LogDetail([s](AsyncDetail &detail) {
    detail("");
    detail("Requested Settings:");
    detail("Scenario : " + ToString(s.scenario));
    detail("Test mode : " + ToString(s.mode));

    // Scenario-specific
    switch (s.scenario) {
      case TestScenario::SingleStream:
        detail("single_stream_expected_latency_ns : ",
               s.single_stream_expected_latency_ns);
        detail("single_stream_target_latency_percentile : ",
               s.single_stream_target_latency_percentile);
        break;
      case TestScenario::MultiStream:
      case TestScenario::MultiStreamFree:
        detail("multi_stream_target_qps : ", s.multi_stream_target_qps);
        detail("multi_stream_target_latency_ns : ",
               s.multi_stream_target_latency_ns);
        detail("multi_stream_target_latency_percentile : ",
               s.multi_stream_target_latency_percentile);
        detail("multi_stream_samples_per_query : ",
               s.multi_stream_samples_per_query);
        detail("multi_stream_max_async_queries : ",
               s.multi_stream_max_async_queries);
        break;
      case TestScenario::Server:
        detail("server_target_qps : ", s.server_target_qps);
        detail("server_target_latency_ns : ", s.server_target_latency_ns);
        detail("server_target_latency_percentile : ",
               s.server_target_latency_percentile);
        detail("server_coalesce_queries : ", s.server_coalesce_queries);
        detail("server_find_peak_qps_decimals_of_precision : ",
               s.server_find_peak_qps_decimals_of_precision);
        detail("server_find_peak_qps_boundary_step_size : ",
               s.server_find_peak_qps_boundary_step_size);
        detail("server_max_async_queries : ", s.server_max_async_queries);
        break;
      case TestScenario::Offline:
        detail("offline_expected_qps : ", s.offline_expected_qps);
        break;
    }

    // Overrides
    detail("min_duration_ms : ", s.min_duration_ms);
    detail("max_duration_ms : ", s.max_duration_ms);
    detail("min_query_count : ", s.min_query_count);
    detail("max_query_count : ", s.max_query_count);
    detail("qsl_rng_seed : ", s.qsl_rng_seed);
    detail("sample_index_rng_seed : ", s.sample_index_rng_seed);
    detail("schedule_rng_seed : ", s.schedule_rng_seed);
    detail("accuracy_log_rng_seed : ", s.accuracy_log_rng_seed);
    detail("accuracy_log_probability : ", s.accuracy_log_probability);
    detail("print_timestamps : ", s.print_timestamps);
    detail("performance_issue_unique : ", s.performance_issue_unique);
    detail("performance_issue_same : ", s.performance_issue_same);
    detail("performance_issue_same_index : ", s.performance_issue_same_index);
    detail("performance_sample_count_override : ",
           s.performance_sample_count_override);
    detail("");
  });
}

void TestSettingsInternal::LogEffectiveSettings() const {
  LogDetail([s = *this](AsyncDetail &detail) {
    detail("");
    detail("Effective Settings:");

    detail("Scenario : " + ToString(s.scenario));
    detail("Test mode : " + ToString(s.mode));

    detail("samples_per_query : ", s.samples_per_query);
    detail("target_qps : ", s.target_qps);
    detail("target_latency (ns): ", s.target_latency.count());
    detail("target_latency_percentile : ", s.target_latency_percentile);
    detail("max_async_queries : ", s.max_async_queries);
    detail("target_duration (ms): ", s.target_duration.count());
    detail("min_duration (ms): ", s.min_duration.count());
    detail("max_duration (ms): ", s.max_duration.count());
    detail("min_query_count : ", s.min_query_count);
    detail("max_query_count : ", s.max_query_count);
    detail("min_sample_count : ", s.min_sample_count);
    detail("qsl_rng_seed : ", s.qsl_rng_seed);
    detail("sample_index_rng_seed : ", s.sample_index_rng_seed);
    detail("schedule_rng_seed : ", s.schedule_rng_seed);
    detail("accuracy_log_rng_seed : ", s.accuracy_log_rng_seed);
    detail("accuracy_log_probability : ", s.accuracy_log_probability);
    detail("print_timestamps : ", s.print_timestamps);
    detail("performance_issue_unique : ", s.performance_issue_unique);
    detail("performance_issue_same : ", s.performance_issue_same);
    detail("performance_issue_same_index : ", s.performance_issue_same_index);
    detail("performance_sample_count : ", s.performance_sample_count);
  });
}

void TestSettingsInternal::LogAllSettings() const {
  LogEffectiveSettings();
  LogRequestedTestSettings(requested);
}

void TestSettingsInternal::LogSummary(AsyncSummary &summary) const {
  summary("samples_per_query : ", samples_per_query);
  summary("target_qps : ", target_qps);
  summary("target_latency (ns): ", target_latency.count());
  summary("max_async_queries : ", max_async_queries);
  summary("min_duration (ms): ", min_duration.count());
  summary("max_duration (ms): ", max_duration.count());
  summary("min_query_count : ", min_query_count);
  summary("max_query_count : ", max_query_count);
  summary("qsl_rng_seed : ", qsl_rng_seed);
  summary("sample_index_rng_seed : ", sample_index_rng_seed);
  summary("schedule_rng_seed : ", schedule_rng_seed);
  summary("accuracy_log_rng_seed : ", accuracy_log_rng_seed);
  summary("accuracy_log_probability : ", accuracy_log_probability);
  summary("print_timestamps : ", print_timestamps);
  summary("performance_issue_unique : ", performance_issue_unique);
  summary("performance_issue_same : ", performance_issue_same);
  summary("performance_issue_same_index : ", performance_issue_same_index);
  summary("performance_sample_count : ", performance_sample_count);
}

}  // namespace loadgen

/// \todo The TestSettings::FromConfig definition belongs in a test_settings.cc
/// file which doesn't yet exist. To avoid churn so close to the submission
/// deadline, adding a test_settings.cc file has been deferred to v0.6.
int TestSettings::FromConfig(const std::string &path, const std::string &model,
                             const std::string &scenario) {
  // TODO: move this method to a new file test_settings.cc
  std::map<std::string, std::string> kv;

  // lookup key/value pairs from config
  auto lookupkv = [&](const std::string &model, const std::string &scenario,
                      const std::string &key, uint64_t *val_l, double *val_d,
                      double multiplier = 1.0) {
    std::map<std::string, std::string>::iterator it;
    std::string found;
    // lookup exact key first
    it = kv.find(model + "." + scenario + "." + key);
    if (it != kv.end()) {
      found = it->second;
    } else {
      // lookup key with model wildcard
      it = kv.find("*." + scenario + "." + key);
      if (it != kv.end()) {
        found = it->second;
      } else {
        it = kv.find(model + ".*." + key);
        if (it != kv.end()) {
          found = it->second;
        } else {
          it = kv.find("*.*." + key);
          if (it != kv.end()) {
            found = it->second;
          } else {
            return false;
          }
        }
      }
    }
    // if we get here, found will be set
    if (val_l) {
      *val_l = strtoull(found.c_str(), nullptr, 0) * static_cast<uint64_t>(multiplier);
    }
    if (val_d) *val_d = strtod(found.c_str(), nullptr) * multiplier;
    return true;
  };

  // dirt simple config parser
  std::ifstream fss(path);
  std::string line;
  int line_nr = 0;
  int errors = 0;
  if (!fss.is_open()) {
    LogDetail([p = path](AsyncDetail &detail) {
      detail.Error("can't open file ", p);
    });
    return -ENOENT;
  }
  while (std::getline(fss, line)) {
    line_nr++;
    std::istringstream iss(line);
    std::string s, k;
    int looking_for = 0;  // 0=key, 1=equal, 2=value
    while (iss >> s) {
      if (s == "#" && looking_for != 2) {
        // done with this line
        break;
      }
      if (looking_for == 2) {
        // got key and value
        const char *start = s.c_str();
        char *stop;
        (void)strtoul(start, &stop, 0);
        if (start + s.size() == stop) {
          kv[k] = s;
          continue;
        }
        (void)strtod(start, &stop);
        if (start + s.size() == stop) {
          kv[k] = s;
          continue;
        }
        errors++;
        LogDetail([l = line_nr](AsyncDetail &detail) {
          detail.Error("value needs to be integer or double, line=", l);
        });
        break;
      }
      if (looking_for == 1 && s != "=") {
        errors++;
        LogDetail([l = line_nr](AsyncDetail &detail) {
          detail.Error("expected 'key=value', line=", l);
        });
        break;
      }
      if (looking_for == 0) k = s;
      looking_for++;
    }
  }
  if (errors != 0) return -EINVAL;

  uint64_t val;

  // keys that apply to all scenarios
  if (lookupkv(model, scenario, "mode", &val, nullptr)) {
    switch (val) {
      case 0:
        mode = TestMode::SubmissionRun;
        break;
      case 1:
        mode = TestMode::AccuracyOnly;
        break;
      case 2:
        mode = TestMode::PerformanceOnly;
        break;
      case 3:
        mode = TestMode::FindPeakPerformance;
        break;
      default:
        LogDetail([](AsyncDetail &detail) {
          detail.Error("Invalid value passed to Mode key in config.");
        });
        break;
    }
  }
  lookupkv(model, scenario, "min_duration", &min_duration_ms, nullptr);
  lookupkv(model, scenario, "max_duration", &max_duration_ms, nullptr);
  lookupkv(model, scenario, "min_query_count", &min_query_count, nullptr);
  lookupkv(model, scenario, "max_query_count", &max_query_count, nullptr);
  lookupkv(model, scenario, "qsl_rng_seed", &qsl_rng_seed, nullptr);
  lookupkv(model, scenario, "sample_index_rng_seed", &sample_index_rng_seed,
           nullptr);
  lookupkv(model, scenario, "schedule_rng_seed", &schedule_rng_seed, nullptr);
  lookupkv(model, scenario, "accuracy_log_rng_seed", &accuracy_log_rng_seed,
           nullptr);
  lookupkv(model, scenario, "accuracy_log_probability", nullptr,
           &accuracy_log_probability, 0.01);
  if (lookupkv(model, scenario, "print_timestamps", &val, nullptr))
    print_timestamps = (val == 0) ? false : true;
  if (lookupkv(model, scenario, "performance_issue_unique", &val, nullptr))
    performance_issue_unique = (val == 0) ? false : true;
  if (lookupkv(model, scenario, "performance_issue_same", &val, nullptr))
    performance_issue_same = (val == 0) ? false : true;
  lookupkv(model, scenario, "performance_issue_same_index",
           &performance_issue_same_index, nullptr);
  lookupkv(model, scenario, "performance_sample_count_override",
           &performance_sample_count_override, nullptr);

  // keys that apply to SingleStream
  lookupkv(model, "SingleStream", "target_latency_percentile", nullptr,
           &single_stream_target_latency_percentile, 0.01);
  lookupkv(model, "SingleStream", "target_latency",
           &single_stream_expected_latency_ns, nullptr, 1000 * 1000);

  // keys that apply to MultiStream
  lookupkv(model, "MultiStream", "target_latency_percentile", nullptr,
           &multi_stream_target_latency_percentile, 0.01);
  lookupkv(model, "MultiStream", "target_qps", nullptr,
           &multi_stream_target_qps);
  if (lookupkv(model, "MultiStream", "samples_per_query", &val, nullptr))
    multi_stream_samples_per_query = static_cast<int>(val);
  if (lookupkv(model, "MultiStream", "max_async_queries", &val, nullptr))
    multi_stream_max_async_queries = static_cast<int>(val);

  // keys that apply to Server
  lookupkv(model, "Server", "target_latency_percentile", nullptr,
           &server_target_latency_percentile, 0.01);
  lookupkv(model, "Server", "target_latency", &server_target_latency_ns,
           nullptr, 1000 * 1000);
  lookupkv(model, "Server", "target_qps", nullptr, &server_target_qps);
  if (lookupkv(model, "Server", "coalesce_queries", &val, nullptr))
    server_coalesce_queries = (val == 0) ? false : true;
  if (lookupkv(model, "Server", "max_async_queries", &val, nullptr))
    server_max_async_queries = int(val);

  // keys that apply to Offline
  lookupkv(model, "Offline", "target_qps", 0, &offline_expected_qps);

  return 0;
}

}  // namespace mlperf
