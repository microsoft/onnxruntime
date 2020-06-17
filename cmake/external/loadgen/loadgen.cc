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

#include "loadgen.h"

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "logging.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "test_settings_internal.h"
#include "utils.h"

namespace mlperf {

/// \brief Loadgen implementation details.
namespace loadgen {

struct SampleMetadata;
class QueryMetadata;

/// \brief Every query and sample within a call to StartTest gets a unique
/// sequence id for easy cross reference, and a random number which is used to
/// determine accuracy logging when it is enabled.
struct SequenceGen {
  uint64_t NextQueryId() { return query_id++; }
  uint64_t NextSampleId() { return sample_id++; }
  uint64_t CurrentSampleId() { return sample_id; }
  double NextAccLogRng() { return accuracy_log_dist(accuracy_log_rng); }
  void InitAccLogRng(uint64_t accuracy_log_rng_seed) {
    accuracy_log_rng = std::mt19937(accuracy_log_rng_seed);
  }

 private:
  uint64_t query_id = 0;
  uint64_t sample_id = 0;
  std::mt19937 accuracy_log_rng;
  std::uniform_real_distribution<double> accuracy_log_dist =
      std::uniform_real_distribution<double>(0, 1);
};

/// \brief A random set of samples in the QSL that should fit in RAM when
/// loaded together.
struct LoadableSampleSet {
  std::vector<QuerySampleIndex> set;
  const size_t sample_distribution_end;  // Excludes padding in multi-stream.
};

/// \brief An interface for a particular scenario + mode to implement for
/// extended hanlding of sample completion.
struct ResponseDelegate {
  virtual ~ResponseDelegate() = default;
  virtual void SampleComplete(SampleMetadata*, QuerySampleResponse*,
                              PerfClock::time_point) = 0;
  virtual void QueryComplete() = 0;
};

/// \brief Used by the loadgen to coordinate response data and completion.
struct SampleMetadata {
  QueryMetadata* query_metadata;
  uint64_t sequence_id;
  QuerySampleIndex sample_index;
  double accuracy_log_val;
};

/// \brief Maintains data and timing info for a query and all its samples.
class QueryMetadata {
 public:
  QueryMetadata(const std::vector<QuerySampleIndex>& query_sample_indices,
                std::chrono::nanoseconds scheduled_delta,
                ResponseDelegate* response_delegate, SequenceGen* sequence_gen)
      : scheduled_delta(scheduled_delta),
        response_delegate(response_delegate),
        sequence_id(sequence_gen->NextQueryId()),
        wait_count_(query_sample_indices.size()) {
    samples_.reserve(query_sample_indices.size());
    for (QuerySampleIndex qsi : query_sample_indices) {
      samples_.push_back({this, sequence_gen->NextSampleId(), qsi,
                          sequence_gen->NextAccLogRng()});
    }
    query_to_send.reserve(query_sample_indices.size());
    for (auto& s : samples_) {
      query_to_send.push_back(
          {reinterpret_cast<ResponseId>(&s), s.sample_index});
    }
  }

  QueryMetadata(QueryMetadata&& src)
      : query_to_send(std::move(src.query_to_send)),
        scheduled_delta(src.scheduled_delta),
        response_delegate(src.response_delegate),
        sequence_id(src.sequence_id),
        wait_count_(src.samples_.size()),
        samples_(std::move(src.samples_)) {
    // The move constructor should only be called while generating a
    // vector of QueryMetadata, before it's been used.
    // Assert that wait_count_ is in its initial state.
    assert(src.wait_count_.load() == samples_.size());
    // Update the "parent" of each sample to be this query; the old query
    // address will no longer be valid.
    // TODO: Only set up the sample parenting once after all the queries have
    //       been created, rather than re-parenting on move here.
    for (size_t i = 0; i < samples_.size(); i++) {
      SampleMetadata* s = &samples_[i];
      s->query_metadata = this;
      query_to_send[i].id = reinterpret_cast<ResponseId>(s);
    }
  }

  void NotifyOneSampleCompleted(PerfClock::time_point timestamp) {
    size_t old_count = wait_count_.fetch_sub(1, std::memory_order_relaxed);
    if (old_count == 1) {
      all_samples_done_time = timestamp;
      all_samples_done_.set_value();
      response_delegate->QueryComplete();
    }
  }

  void WaitForAllSamplesCompleted() { all_samples_done_.get_future().wait(); }

  PerfClock::time_point WaitForAllSamplesCompletedWithTimestamp() {
    all_samples_done_.get_future().wait();
    return all_samples_done_time;
  }

 public:
  std::vector<QuerySample> query_to_send;
  const std::chrono::nanoseconds scheduled_delta;
  ResponseDelegate* const response_delegate;
  const uint64_t sequence_id;

  // Performance information.

  size_t scheduled_intervals = 0;  // Number of intervals between queries, as
                                   // actually scheduled during the run.
                                   // For the multi-stream scenario only.
  PerfClock::time_point scheduled_time;
  PerfClock::time_point issued_start_time;
  PerfClock::time_point all_samples_done_time;

 private:
  std::atomic<size_t> wait_count_;
  std::promise<void> all_samples_done_;
  std::vector<SampleMetadata> samples_;
};

/// \brief Generates nanoseconds from a start time to multiple end times.
/// TODO: This isn't very useful anymore. Remove it.
struct DurationGeneratorNs {
  const PerfClock::time_point start;
  int64_t delta(PerfClock::time_point end) const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
        .count();
  }
};

/// \brief ResponseDelegate implementation templated by scenario and mode.
template <TestScenario scenario, TestMode mode>
struct ResponseDelegateDetailed : public ResponseDelegate {
  std::atomic<size_t> queries_completed{0};
  double accuracy_log_offset = 0.0f;
  double accuracy_log_prob = 0.0f;

  void SampleComplete(SampleMetadata* sample, QuerySampleResponse* response,
                      PerfClock::time_point complete_begin_time) override {
    // Using a raw pointer here should help us hit the std::function
    // small buffer optimization code path when we aren't copying data.
    // For some reason, using std::unique_ptr<std::vector> wasn't moving
    // into the lambda; even with C++14.
    std::vector<uint8_t>* sample_data_copy = nullptr;
    double accuracy_log_val =
        sample->accuracy_log_val + accuracy_log_offset < 1.0
            ? sample->accuracy_log_val + accuracy_log_offset
            : sample->accuracy_log_val + accuracy_log_offset - 1.0;
    if (mode == TestMode::AccuracyOnly ||
        accuracy_log_val <= accuracy_log_prob) {
      // TODO: Verify accuracy with the data copied here.
      uint8_t* src_begin = reinterpret_cast<uint8_t*>(response->data);
      uint8_t* src_end = src_begin + response->size;
      sample_data_copy = new std::vector<uint8_t>(src_begin, src_end);
    }
    Log([sample, complete_begin_time, sample_data_copy](AsyncLog& log) {
      QueryMetadata* query = sample->query_metadata;
      DurationGeneratorNs sched{query->scheduled_time};

      if (scenario == TestScenario::Server) {
        // Trace the server scenario as a stacked graph via counter events.
        DurationGeneratorNs issued{query->issued_start_time};
        log.TraceCounterEvent("Latency", query->scheduled_time, "issue_delay",
                              sched.delta(query->issued_start_time),
                              "issue_to_done",
                              issued.delta(complete_begin_time));
      }

      // While visualizing overlapping samples in offline mode is not
      // practical, sample completion is still recorded for auditing purposes.
      log.TraceSample("Sample", sample->sequence_id, query->scheduled_time,
                      complete_begin_time, "sample_seq", sample->sequence_id,
                      "query_seq", query->sequence_id, "sample_idx",
                      sample->sample_index, "issue_start_ns",
                      sched.delta(query->issued_start_time), "complete_ns",
                      sched.delta(complete_begin_time));

      if (sample_data_copy) {
        log.LogAccuracy(sample->sequence_id, sample->sample_index,
                        LogBinaryAsHexString{sample_data_copy});
        delete sample_data_copy;
      }

      // Record the latency at the end, since it will unblock the issuing
      // thread and potentially destroy the metadata being used above.
      QuerySampleLatency latency = sched.delta(complete_begin_time);
      log.RecordSampleCompletion(sample->sequence_id, complete_begin_time,
                                 latency);
    });
  }

  void QueryComplete() override {
    // We only need to track outstanding queries in the server scenario to
    // detect when the SUT has fallen too far behind.
    if (scenario == TestScenario::Server) {
      queries_completed.fetch_add(1, std::memory_order_relaxed);
    }
  }
};

/// \brief Selects the query timestamps for all scenarios except Server.
template <TestScenario scenario>
auto ScheduleDistribution(double qps) {
  return [period = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::duration<double>(1.0 / qps))](auto& /*gen*/) {
    return period;
  };
}

/// \brief Selects the query timestamps for the Server scenario.
template <>
auto ScheduleDistribution<TestScenario::Server>(double qps) {
  // Poisson arrival process corresponds to exponentially distributed
  // interarrival times.
  return [dist = std::exponential_distribution<>(qps)](auto& gen) mutable {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

/// \brief Selects samples for the accuracy mode.
template <TestMode mode>
auto SampleDistribution(size_t sample_count, size_t stride, std::mt19937* rng) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < sample_count; i += stride) {
    indices.push_back(i);
  }
  std::shuffle(indices.begin(), indices.end(), *rng);
  return [indices = std::move(indices), i = size_t(0)](auto& /*gen*/) mutable {
    return indices.at(i++);
  };
}

/// \brief Selects samples for the performance mode.
template <>
auto SampleDistribution<TestMode::PerformanceOnly>(size_t sample_count,
                                                   size_t /*stride*/,
                                                   std::mt19937* /*rng*/) {
  return [dist = std::uniform_int_distribution<>(0, sample_count - 1)](
             auto& gen) mutable { return dist(gen); };
}

/// \brief Generates queries for the requested settings, templated by
/// scenario and mode.
/// \todo Make GenerateQueries faster.
/// QueryMetadata is expensive to move; either reserve queries in advance
/// so the queries vector doesn't need to grow. And/or parent samples to their
/// queries only after all queries have been generated.
/// \todo For the server scenario only, scale the query timeline at the end so
/// the QPS as scheduled is equal to the QPS as requested.
template <TestScenario scenario, TestMode mode>
std::vector<QueryMetadata> GenerateQueries(
    const TestSettingsInternal& settings,
    const LoadableSampleSet& loaded_sample_set, SequenceGen* sequence_gen,
    ResponseDelegate* response_delegate) {
  auto tracer =
      MakeScopedTracer([](AsyncTrace& trace) { trace("GenerateQueries"); });

  auto& loaded_samples = loaded_sample_set.set;

  // Generate 2x more samples than we think we'll need given the expected
  // QPS in case the SUT is faster than expected.
  // We should exit before issuing all queries.
  // Does not apply to the server scenario since the duration only
  // depends on the ideal scheduled time, not the actual issue time.
  const int duration_multiplier = scenario == TestScenario::Server ? 1 : 2;
  std::chrono::microseconds gen_duration =
      duration_multiplier * settings.target_duration;
  size_t min_queries = settings.min_query_count;

  size_t samples_per_query = settings.samples_per_query;
  if (mode == TestMode::AccuracyOnly && scenario == TestScenario::Offline) {
    samples_per_query = loaded_sample_set.sample_distribution_end;
  }

  // We should not exit early in accuracy mode.
  if (mode == TestMode::AccuracyOnly || settings.performance_issue_unique ||
      settings.performance_issue_same) {
    gen_duration = std::chrono::microseconds(0);
    // Integer truncation here is intentional.
    // For MultiStream, loaded samples is properly padded.
    // For Offline, we create a 'remainder' query at the end of this function.
    min_queries = loaded_samples.size() / samples_per_query;
  }

  std::vector<QueryMetadata> queries;

  // Using the std::mt19937 pseudo-random number generator ensures a modicum of
  // cross platform reproducibility for trace generation.
  std::mt19937 sample_rng(settings.sample_index_rng_seed);
  std::mt19937 schedule_rng(settings.schedule_rng_seed);

  constexpr bool kIsMultiStream = scenario == TestScenario::MultiStream ||
                                  scenario == TestScenario::MultiStreamFree;
  const size_t sample_stride = kIsMultiStream ? samples_per_query : 1;

  auto sample_distribution = SampleDistribution<mode>(
      loaded_sample_set.sample_distribution_end, sample_stride, &sample_rng);
  // Use the unique sample distribution same as in AccuracyMode to
  // to choose samples when either flag performance_issue_unique
  // or performance_issue_same is set.
  auto sample_distribution_unique = SampleDistribution<TestMode::AccuracyOnly>(
      loaded_sample_set.sample_distribution_end, sample_stride, &sample_rng);

  auto schedule_distribution =
      ScheduleDistribution<scenario>(settings.target_qps);

  std::vector<QuerySampleIndex> samples(samples_per_query);
  std::chrono::nanoseconds timestamp(0);
  std::chrono::nanoseconds prev_timestamp(0);
  // Choose a single sample to repeat when in performance_issue_same mode
  QuerySampleIndex same_sample = settings.performance_issue_same_index;

  while (prev_timestamp < gen_duration || queries.size() < min_queries) {
    if (kIsMultiStream) {
      QuerySampleIndex sample_i = settings.performance_issue_unique
                                      ? sample_distribution_unique(sample_rng)
                                      : settings.performance_issue_same
                                            ? same_sample
                                            : sample_distribution(sample_rng);
      for (auto& s : samples) {
        // Select contiguous samples in the MultiStream scenario.
        // This will not overflow, since GenerateLoadableSets adds padding at
        // the end of the loadable sets in the MultiStream scenario.
        // The padding allows the starting samples to be the same for each
        // query as the value of samples_per_query increases.
        s = loaded_samples[sample_i++];
      }
    } else if (scenario == TestScenario::Offline) {
      // For the Offline + Performance scenario, we also want to support
      // contiguous samples. In this scenario the query can be much larger than
      // what fits into memory. We simply repeat loaded_samples N times, plus a
      // remainder to ensure we fill up samples. Note that this eliminates
      // randomization.
      size_t num_loaded_samples = loaded_samples.size();
      size_t num_full_repeats = samples_per_query / num_loaded_samples;
      uint64_t remainder = samples_per_query % (num_loaded_samples);
      if (settings.performance_issue_same) {
        std::fill(samples.begin(), samples.begin() + num_loaded_samples,
                  loaded_samples[same_sample]);
      } else {
        for (size_t i = 0; i < num_full_repeats; ++i) {
          std::copy(loaded_samples.begin(), loaded_samples.end(),
                    samples.begin() + i * num_loaded_samples);
        }

        std::copy(loaded_samples.begin(), loaded_samples.begin() + remainder,
                  samples.begin() + num_full_repeats * num_loaded_samples);
      }
    } else {
      for (auto& s : samples) {
        s = loaded_samples[settings.performance_issue_unique
                               ? sample_distribution_unique(sample_rng)
                               : settings.performance_issue_same
                                     ? same_sample
                                     : sample_distribution(sample_rng)];
      }
    }
    queries.emplace_back(samples, timestamp, response_delegate, sequence_gen);
    prev_timestamp = timestamp;
    timestamp += schedule_distribution(schedule_rng);
  }

  // See if we need to create a "remainder" query for offline+accuracy to
  // ensure we issue all samples in loaded_samples. Offline doesn't pad
  // loaded_samples like MultiStream does.
  if (scenario == TestScenario::Offline && mode == TestMode::AccuracyOnly) {
    size_t remaining_samples = loaded_samples.size() % samples_per_query;
    if (remaining_samples != 0) {
      samples.resize(remaining_samples);
      for (auto& s : samples) {
        s = loaded_samples[sample_distribution(sample_rng)];
      }
      queries.emplace_back(samples, timestamp, response_delegate, sequence_gen);
    }
  }

  LogDetail([count = queries.size(), spq = settings.samples_per_query,
             duration = timestamp.count()](AsyncDetail& detail) {
    detail("GeneratedQueries: ", "queries", count, "samples per query", spq,
           "duration", duration);
  });

  return queries;
}

/// \brief A base template that should never be used since each scenario has
/// its own specialization.
template <TestScenario scenario>
struct QueryScheduler {
  static_assert(scenario != scenario, "Unhandled TestScenario");
};

/// \brief Schedules queries for issuance in the single stream scenario.
template <>
struct QueryScheduler<TestScenario::SingleStream> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto tracer = MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
    if (prev_query != nullptr) {
      prev_query->WaitForAllSamplesCompleted();
    }
    prev_query = next_query;

    auto now = PerfClock::now();
    next_query->scheduled_time = now;
    next_query->issued_start_time = now;
    return now;
  }

  QueryMetadata* prev_query = nullptr;
};

/// \brief Schedules queries for issuance in the multi stream scenario.
template <>
struct QueryScheduler<TestScenario::MultiStream> {
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point start)
      : qps(settings.target_qps),
        max_async_queries(settings.max_async_queries),
        start_time(start) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    {
      prev_queries.push(next_query);
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
      if (prev_queries.size() > max_async_queries) {
        prev_queries.front()->WaitForAllSamplesCompleted();
        prev_queries.pop();
      }
    }

    {
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Scheduling"); });
      // TODO(brianderson): Skip ticks based on the query complete time,
      //     before the query synchronization + notification thread hop,
      //     rather than after.
      PerfClock::time_point now = PerfClock::now();
      auto i_period_old = i_period;
      PerfClock::time_point tick_time;
      do {
        i_period++;
        tick_time =
            start_time + SecondsToDuration<PerfClock::duration>(i_period / qps);
        Log([tick_time](AsyncLog& log) {
          log.TraceAsyncInstant("QueryInterval", 0, tick_time);
        });
      } while (tick_time < now);
      next_query->scheduled_intervals = i_period - i_period_old;
      next_query->scheduled_time = tick_time;
      std::this_thread::sleep_until(tick_time);
    }

    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  size_t i_period = 0;
  double qps;
  const size_t max_async_queries;
  PerfClock::time_point start_time;
  std::queue<QueryMetadata*> prev_queries;
};

/// \brief Schedules queries for issuance in the single stream free scenario.
template <>
struct QueryScheduler<TestScenario::MultiStreamFree> {
  QueryScheduler(const TestSettingsInternal& settings,
                 const PerfClock::time_point /*start*/)
      : max_async_queries(settings.max_async_queries) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    bool schedule_time_needed = true;
    {
      prev_queries.push(next_query);
      auto tracer =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Waiting"); });
      if (prev_queries.size() > max_async_queries) {
        next_query->scheduled_time =
            prev_queries.front()->WaitForAllSamplesCompletedWithTimestamp();
        schedule_time_needed = false;
        prev_queries.pop();
      }
    }

    auto now = PerfClock::now();
    if (schedule_time_needed) {
      next_query->scheduled_time = now;
    }
    next_query->issued_start_time = now;
    return now;
  }

  const size_t max_async_queries;
  std::queue<QueryMetadata*> prev_queries;
};

/// \brief Schedules queries for issuance in the server scenario.
template <>
struct QueryScheduler<TestScenario::Server> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point start)
      : start(start) {}

  // TODO: Coalesce all queries whose scheduled timestamps have passed.
  PerfClock::time_point Wait(QueryMetadata* next_query) {
    auto tracer =
        MakeScopedTracer([](AsyncTrace& trace) { trace("Scheduling"); });

    auto scheduled_time = start + next_query->scheduled_delta;
    next_query->scheduled_time = scheduled_time;
    std::this_thread::sleep_until(scheduled_time);

    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  const PerfClock::time_point start;
};

/// \brief Schedules queries for issuance in the offline scenario.
template <>
struct QueryScheduler<TestScenario::Offline> {
  QueryScheduler(const TestSettingsInternal& /*settings*/,
                 const PerfClock::time_point start)
      : start(start) {}

  PerfClock::time_point Wait(QueryMetadata* next_query) {
    next_query->scheduled_time = start;
    auto now = PerfClock::now();
    next_query->issued_start_time = now;
    return now;
  }

  const PerfClock::time_point start;
};

/// \brief Provides performance results that are independent of scenario
/// and other context.
/// \todo Move to results.h/cc
struct PerformanceResult {
  std::vector<QuerySampleLatency> sample_latencies;
  std::vector<QuerySampleLatency> query_latencies;  // MultiStream only.
  std::vector<size_t> query_intervals;              // MultiStream only.
  size_t queries_issued;
  double max_latency;
  double final_query_scheduled_time;         // seconds from start.
  double final_query_issued_time;            // seconds from start.
  double final_query_all_samples_done_time;  // seconds from start.
};

/// \brief Issues a series of pre-generated queries.
// TODO: Templates for scenario and mode are overused, given the loadgen
//       no longer generates queries on the fly. Should we reduce the
//       use of templates?
template <TestScenario scenario, TestMode mode>
PerformanceResult IssueQueries(SystemUnderTest* sut,
                               const TestSettingsInternal& settings,
                               const LoadableSampleSet& loaded_sample_set,
                               SequenceGen* sequence_gen) {
  ResponseDelegateDetailed<scenario, mode> response_logger;
  std::uniform_real_distribution<double> accuracy_log_offset_dist =
      std::uniform_real_distribution<double>(0.0, 1.0);
  std::mt19937 accuracy_log_offset_rng(settings.accuracy_log_rng_seed);
  response_logger.accuracy_log_offset =
      accuracy_log_offset_dist(accuracy_log_offset_rng);
  response_logger.accuracy_log_prob = settings.accuracy_log_probability;

  auto sequence_id_start = sequence_gen->CurrentSampleId();
  std::vector<QueryMetadata> queries = GenerateQueries<scenario, mode>(
      settings, loaded_sample_set, sequence_gen, &response_logger);
  auto sequence_id_end = sequence_gen->CurrentSampleId();
  size_t max_latencies_to_record = sequence_id_end - sequence_id_start;

  GlobalLogger().RestartLatencyRecording(sequence_id_start,
                                         max_latencies_to_record);

  size_t queries_issued = 0;

  auto start_for_power = std::chrono::system_clock::now();
  const PerfClock::time_point start = PerfClock::now();
  PerfClock::time_point last_now = start;
  QueryScheduler<scenario> query_scheduler(settings, start);

  // We can never run out of generated queries in the server scenario,
  // since the duration depends on the scheduled query time and not
  // the actual issue time.
  bool ran_out_of_generated_queries = scenario != TestScenario::Server;
  size_t expected_latencies = 0;
  for (auto& query : queries) {
    auto tracer1 =
        MakeScopedTracer([](AsyncTrace& trace) { trace("SampleLoop"); });
    last_now = query_scheduler.Wait(&query);

    // Issue the query to the SUT.
    {
      auto tracer3 =
          MakeScopedTracer([](AsyncTrace& trace) { trace("IssueQuery"); });
      sut->IssueQuery(query.query_to_send);
    }

    expected_latencies += query.query_to_send.size();
    queries_issued++;

    if (mode == TestMode::AccuracyOnly) {
      // TODO: Rate limit in accuracy mode so accuracy mode works even
      //       if the expected/target performance is way off.
      continue;
    }

    auto duration = (last_now - start);
    if (scenario == TestScenario::Server) {
      if (settings.max_async_queries != 0) {
        size_t queries_outstanding =
            queries_issued -
            response_logger.queries_completed.load(std::memory_order_relaxed);
        if (queries_outstanding > settings.max_async_queries) {
          LogDetail([queries_issued, queries_outstanding](AsyncDetail& detail) {
            detail.Error("Ending early: Too many outstanding queries.",
                         "issued", queries_issued, "outstanding",
                         queries_outstanding);
          });
          break;
        }
      }
    } else {
      if (queries_issued >= settings.min_query_count &&
          duration >= settings.target_duration) {
        LogDetail([](AsyncDetail& detail) {
          detail(
              "Ending naturally: Minimum query count and test duration met.");
        });
        ran_out_of_generated_queries = false;
        break;
      }
    }

    if (settings.max_query_count != 0 &&
        queries_issued >= settings.max_query_count) {
      LogDetail([queries_issued](AsyncDetail& detail) {
        detail.Error("Ending early: Max query count reached.", "query_count",
                     queries_issued);
      });
      ran_out_of_generated_queries = false;
      break;
    }

    if (settings.max_duration.count() != 0 &&
        duration > settings.max_duration) {
      LogDetail([duration](AsyncDetail& detail) {
        detail.Error("Ending early: Max test duration reached.", "duration_ns",
                     duration.count());
      });
      ran_out_of_generated_queries = false;
      break;
    }
  }

  // Let the SUT know it should not expect any more queries.
  sut->FlushQueries();

  if (mode == TestMode::PerformanceOnly && ran_out_of_generated_queries) {
    LogDetail([](AsyncDetail& detail) {
      detail.Error(
          "Ending early: Ran out of generated queries to issue before the "
          "minimum query count and test duration were reached.");
      detail(
          "Please update the relevant expected latency or target qps in the "
          "TestSettings so they are more accurate.");
    });
  }

  // Wait for tail queries to complete and collect all the latencies.
  // We have to keep the synchronization primitives alive until the SUT
  // is done with them.
  auto& final_query = queries[queries_issued - 1];
  std::vector<QuerySampleLatency> sample_latencies(
      GlobalLogger().GetLatenciesBlocking(expected_latencies));

  // Log contention counters after every test as a sanity check.
  GlobalLogger().LogContentionAndAllocations();

  // This properly accounts for the fact that the max completion time may not
  // belong to the final query. It also excludes any time spent postprocessing
  // in the loadgen itself after final completion, which may be significant
  // in the offline scenario.
  PerfClock::time_point max_completion_time =
      GlobalLogger().GetMaxCompletionTime();
  auto sut_active_duration = max_completion_time - start;
  LogDetail([start_for_power, sut_active_duration](AsyncDetail& detail) {
    auto end_for_power =
        start_for_power +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            sut_active_duration);
    detail("POWER_BEGIN: ", "mode", ToString(mode), "time",
           DateTimeStringForPower(start_for_power));
    detail("POWER_END: ", "mode", ToString(mode), "time",
           DateTimeStringForPower(end_for_power));
  });

  double max_latency =
      QuerySampleLatencyToSeconds(GlobalLogger().GetMaxLatencySoFar());
  double final_query_scheduled_time =
      DurationToSeconds(final_query.scheduled_delta);
  double final_query_issued_time =
      DurationToSeconds(final_query.issued_start_time - start);
  double final_query_all_samples_done_time =
      DurationToSeconds(final_query.all_samples_done_time - start);

  std::vector<QuerySampleLatency> query_latencies;
  std::vector<size_t> query_intervals;
  if (scenario == TestScenario::MultiStream ||
      scenario == TestScenario::MultiStreamFree) {
    query_latencies.resize(queries_issued);
    query_intervals.resize(queries_issued);
    for (size_t i = 0; i < queries_issued; i++) {
      query_latencies[i] = DurationGeneratorNs{queries[i].scheduled_time}.delta(
          queries[i].all_samples_done_time);
      if (i < queries_issued - settings.max_async_queries) {
        // For all queries except the last few, take into account actual
        // skipped intervals to the next query.
        query_intervals[i] =
            queries[i + settings.max_async_queries].scheduled_intervals;
      } else {
        // For the last queries, use query latency to guess if imaginary
        // queries issued at the end would have skipped intervals.
        query_intervals[i] =
            std::ceil(settings.target_qps *
                      QuerySampleLatencyToSeconds(query_latencies[i]));
      }
    }
  }

  return PerformanceResult{std::move(sample_latencies),
                           std::move(query_latencies),
                           std::move(query_intervals),
                           queries_issued,
                           max_latency,
                           final_query_scheduled_time,
                           final_query_issued_time,
                           final_query_all_samples_done_time};
}

/// \brief Wraps PerformanceResult with relevant context to change how
/// it's interpreted and reported.
/// \todo Move to results.h/cc
struct PerformanceSummary {
  std::string sut_name;
  TestSettingsInternal settings;
  PerformanceResult pr;

  // Set by ProcessLatencies.
  size_t sample_count = 0;
  QuerySampleLatency sample_latency_min = 0;
  QuerySampleLatency sample_latency_max = 0;
  QuerySampleLatency sample_latency_mean = 0;

  /// \brief The latency at a given percentile.
  struct PercentileEntry {
    const double percentile;
    QuerySampleLatency sample_latency = 0;
    QuerySampleLatency query_latency = 0;  // MultiStream only.
    size_t query_intervals = 0;            // MultiStream only.
  };
  // Latency target percentile
  PercentileEntry target_latency_percentile{settings.target_latency_percentile};
  PercentileEntry latency_percentiles[6] = {{.50}, {.90}, {.95},
                                            {.97}, {.99}, {.999}};

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
  // MSVC complains if there is no explicit constructor.
  // (target_latency_percentile above depends on construction with settings)
  PerformanceSummary(
      const std::string& sut_name_arg, const TestSettingsInternal& settings_arg,
      const PerformanceResult& pr_arg)
      : sut_name(sut_name_arg), settings(settings_arg), pr(pr_arg){};
#endif
  void ProcessLatencies();

  bool MinDurationMet(std::string* recommendation);
  bool MinQueriesMet();
  bool MinSamplesMet();
  bool HasPerfConstraints();
  bool PerfConstraintsMet(std::string* recommendation);
  void Log(AsyncSummary& summary);
};

void PerformanceSummary::ProcessLatencies() {
  if (pr.sample_latencies.empty()) {
    return;
  }

  sample_count = pr.sample_latencies.size();

  QuerySampleLatency accumulated_latency = 0;
  for (auto latency : pr.sample_latencies) {
    accumulated_latency += latency;
  }
  sample_latency_mean = accumulated_latency / sample_count;

  std::sort(pr.sample_latencies.begin(), pr.sample_latencies.end());

  target_latency_percentile.sample_latency =
      pr.sample_latencies[sample_count * target_latency_percentile.percentile];
  sample_latency_min = pr.sample_latencies.front();
  sample_latency_max = pr.sample_latencies.back();
  for (auto& lp : latency_percentiles) {
    assert(lp.percentile >= 0.0);
    assert(lp.percentile < 1.0);
    lp.sample_latency = pr.sample_latencies[sample_count * lp.percentile];
  }

  // MultiStream only after this point.
  if (settings.scenario != TestScenario::MultiStream &&
      settings.scenario != TestScenario::MultiStreamFree) {
    return;
  }

  // Calculate per-query stats.
  size_t query_count = pr.queries_issued;
  assert(pr.query_latencies.size() == query_count);
  assert(pr.query_intervals.size() == query_count);
  std::sort(pr.query_latencies.begin(), pr.query_latencies.end());
  std::sort(pr.query_intervals.begin(), pr.query_intervals.end());
  target_latency_percentile.query_latency =
      pr.query_latencies[query_count * target_latency_percentile.percentile];
  target_latency_percentile.query_intervals =
      pr.query_intervals[query_count * target_latency_percentile.percentile];
  for (auto& lp : latency_percentiles) {
    lp.query_latency = pr.query_latencies[query_count * lp.percentile];
    lp.query_intervals = pr.query_intervals[query_count * lp.percentile];
  }
}

bool PerformanceSummary::MinDurationMet(std::string* recommendation) {
  recommendation->clear();
  const double min_duration = DurationToSeconds(settings.min_duration);
  bool min_duration_met = false;
  switch (settings.scenario) {
    case TestScenario::Offline:
      min_duration_met = pr.max_latency >= min_duration;
      break;
    case TestScenario::Server:
      min_duration_met = pr.final_query_scheduled_time >= min_duration;
      break;
    case TestScenario::SingleStream:
    case TestScenario::MultiStream:
    case TestScenario::MultiStreamFree:
      min_duration_met = pr.final_query_issued_time >= min_duration;
      break;
  }
  if (min_duration_met) {
    return true;
  }

  switch (settings.scenario) {
    case TestScenario::SingleStream:
      *recommendation =
          "Decrease the expected latency so the loadgen pre-generates more "
          "queries.";
      break;
    case TestScenario::MultiStream:
      *recommendation =
          "MultiStream should always meet the minimum duration. "
          "Please file a bug.";
      break;
    case TestScenario::MultiStreamFree:
      *recommendation =
          "Increase the target QPS so the loadgen pre-generates more queries.";
      break;
    case TestScenario::Server:
      *recommendation =
          "Increase the target QPS so the loadgen pre-generates more queries.";
      break;
    case TestScenario::Offline:
      *recommendation =
          "Increase expected QPS so the loadgen pre-generates a larger "
          "(coalesced) query.";
      break;
  }
  return false;
}

bool PerformanceSummary::MinQueriesMet() {
  return pr.queries_issued >= settings.min_query_count;
}

bool PerformanceSummary::MinSamplesMet() {
  return sample_count >= settings.min_sample_count;
}

bool PerformanceSummary::HasPerfConstraints() {
  return settings.scenario == TestScenario::MultiStream ||
         settings.scenario == TestScenario::MultiStreamFree ||
         settings.scenario == TestScenario::Server;
}

bool PerformanceSummary::PerfConstraintsMet(std::string* recommendation) {
  recommendation->clear();
  bool perf_constraints_met = true;
  switch (settings.scenario) {
    case TestScenario::SingleStream:
      break;
    case TestScenario::MultiStream:
      ProcessLatencies();
      if (target_latency_percentile.query_intervals >= 2) {
        *recommendation = "Reduce samples per query to improve latency.";
        perf_constraints_met = false;
      }
      break;
    case TestScenario::MultiStreamFree:
      ProcessLatencies();
      if (target_latency_percentile.query_latency >
          settings.target_latency.count()) {
        *recommendation = "Reduce samples per query to improve latency.";
        perf_constraints_met = false;
      }
      break;
    case TestScenario::Server:
      ProcessLatencies();
      if (target_latency_percentile.sample_latency >
          settings.target_latency.count()) {
        *recommendation = "Reduce target QPS to improve latency.";
        perf_constraints_met = false;
      }
      break;
    case TestScenario::Offline:
      break;
  }
  return perf_constraints_met;
}

void PerformanceSummary::Log(AsyncSummary& summary) {
  ProcessLatencies();

  summary(
      "================================================\n"
      "MLPerf Results Summary\n"
      "================================================");
  summary("SUT name : ", sut_name);
  summary("Scenario : ", ToString(settings.scenario));
  summary("Mode     : ", ToString(settings.mode));

  switch (settings.scenario) {
    case TestScenario::SingleStream: {
      summary(DoubleToString(target_latency_percentile.percentile * 100, 0) +
                  "th percentile latency (ns) : ",
              target_latency_percentile.sample_latency);
      break;
    }
    case TestScenario::MultiStream: {
      summary("Samples per query : ", settings.samples_per_query);
      break;
    }
    case TestScenario::MultiStreamFree: {
      double samples_per_second = pr.queries_issued *
                                  settings.samples_per_query /
                                  pr.final_query_all_samples_done_time;
      summary("Samples per second : ", samples_per_second);
      break;
    }
    case TestScenario::Server: {
      // Subtract 1 from sample count since the start of the final sample
      // represents the open end of the time range: i.e. [begin, end).
      // This makes sense since:
      // a) QPS doesn't apply if there's only one sample; it's pure latency.
      // b) If you have precisely 1k QPS, there will be a sample exactly on
      //    the 1 second time point; but that would be the 1001th sample in
      //    the stream. Given the first 1001 queries, the QPS is
      //    1000 queries / 1 second.
      double qps_as_scheduled =
          (sample_count - 1) / pr.final_query_scheduled_time;
      summary("Scheduled samples per second : ",
              DoubleToString(qps_as_scheduled));
      break;
    }
    case TestScenario::Offline: {
      double samples_per_second = sample_count / pr.max_latency;
      summary("Samples per second: ", samples_per_second);
      break;
    }
  }

  std::string min_duration_recommendation;
  std::string perf_constraints_recommendation;

  bool min_duration_met = MinDurationMet(&min_duration_recommendation);
  bool min_queries_met = MinQueriesMet() && MinSamplesMet();
  bool perf_constraints_met =
      PerfConstraintsMet(&perf_constraints_recommendation);
  bool all_constraints_met =
      min_duration_met && min_queries_met && perf_constraints_met;
  summary("Result is : ", all_constraints_met ? "VALID" : "INVALID");
  if (HasPerfConstraints()) {
    summary("  Performance constraints satisfied : ",
            perf_constraints_met ? "Yes" : "NO");
  }
  summary("  Min duration satisfied : ", min_duration_met ? "Yes" : "NO");
  summary("  Min queries satisfied : ", min_queries_met ? "Yes" : "NO");

  if (!all_constraints_met) {
    summary("Recommendations:");
    if (!perf_constraints_met) {
      summary(" * " + perf_constraints_recommendation);
    }
    if (!min_duration_met) {
      summary(" * " + min_duration_recommendation);
    }
    if (!min_queries_met) {
      summary(
          " * The test exited early, before enough queries were issued.\n"
          "   See the detailed log for why this may have occurred.");
    }
  }

  summary(
      "\n"
      "================================================\n"
      "Additional Stats\n"
      "================================================");

  if (settings.scenario == TestScenario::SingleStream) {
    double qps_w_lg = (sample_count - 1) / pr.final_query_issued_time;
    double qps_wo_lg = 1 / QuerySampleLatencyToSeconds(sample_latency_mean);
    summary("QPS w/ loadgen overhead         : " + DoubleToString(qps_w_lg));
    summary("QPS w/o loadgen overhead        : " + DoubleToString(qps_wo_lg));
    summary("");
  } else if (settings.scenario == TestScenario::Server) {
    double qps_as_completed =
        (sample_count - 1) / pr.final_query_all_samples_done_time;
    summary("Completed samples per second    : ",
            DoubleToString(qps_as_completed));
    summary("");
  } else if (settings.scenario == TestScenario::MultiStream ||
             settings.scenario == TestScenario::MultiStreamFree) {
    double ms_per_interval = std::milli::den / settings.target_qps;
    summary("Intervals between each IssueQuery:  ", "qps", settings.target_qps,
            "ms", ms_per_interval);
    for (auto& lp : latency_percentiles) {
      summary(DoubleToString(lp.percentile * 100) + " percentile : ",
              lp.query_intervals);
    }

    summary("");
    double target_ns = settings.target_latency.count();
    double target_ms = target_ns * std::milli::den / std::nano::den;
    summary("Per-query latency:  ", "target_ns",
            settings.target_latency.count(), "target_ms", target_ms);
    for (auto& lp : latency_percentiles) {
      summary(
          DoubleToString(lp.percentile * 100) + " percentile latency (ns)   : ",
          lp.query_latency);
    }

    summary("");
    summary("Per-sample latency:");
  }

  summary("Min latency (ns)                : ", sample_latency_min);
  summary("Max latency (ns)                : ", sample_latency_max);
  summary("Mean latency (ns)               : ", sample_latency_mean);
  for (auto& lp : latency_percentiles) {
    summary(
        DoubleToString(lp.percentile * 100) + " percentile latency (ns)   : ",
        lp.sample_latency);
  }

  summary(
      "\n"
      "================================================\n"
      "Test Parameters Used\n"
      "================================================");
  settings.LogSummary(summary);
}

void LoadSamplesToRam(QuerySampleLibrary* qsl,
                      const std::vector<QuerySampleIndex>& samples) {
  LogDetail([&samples](AsyncDetail& detail) {
    std::string set("\"[");
    for (auto i : samples) {
      set += std::to_string(i) + ",";
    }
    set.resize(set.size() - 1);
    set += "]\"";
    detail("Loading QSL : ", "set", set);
  });
  qsl->LoadSamplesToRam(samples);
}

/// \brief Generates random sets of samples in the QSL that we can load into
/// RAM at the same time.
std::vector<LoadableSampleSet> GenerateLoadableSets(
    QuerySampleLibrary* qsl, const TestSettingsInternal& settings) {
  auto tracer = MakeScopedTracer(
      [](AsyncTrace& trace) { trace("GenerateLoadableSets"); });

  std::vector<LoadableSampleSet> result;
  std::mt19937 qsl_rng(settings.qsl_rng_seed);

  // Generate indices for all available samples in the QSL.
  const size_t qsl_total_count = qsl->TotalSampleCount();
  std::vector<QuerySampleIndex> samples(qsl_total_count);
  for (size_t i = 0; i < qsl_total_count; i++) {
    samples[i] = static_cast<QuerySampleIndex>(i);
  }

  // Randomize the order of the samples.
  std::shuffle(samples.begin(), samples.end(), qsl_rng);

  // Partition the samples into loadable sets.
  const size_t set_size = settings.performance_sample_count;
  const size_t set_padding =
      (settings.scenario == TestScenario::MultiStream ||
       settings.scenario == TestScenario::MultiStreamFree)
          ? settings.samples_per_query - 1
          : 0;
  std::vector<QuerySampleIndex> loadable_set;
  loadable_set.reserve(set_size + set_padding);

  for (auto s : samples) {
    loadable_set.push_back(s);
    if (loadable_set.size() == set_size) {
      result.push_back({std::move(loadable_set), set_size});
      loadable_set.clear();
      loadable_set.reserve(set_size + set_padding);
    }
  }

  if (!loadable_set.empty()) {
    // Copy the size since it will become invalid after the move.
    size_t loadable_set_size = loadable_set.size();
    result.push_back({std::move(loadable_set), loadable_set_size});
  }

  // Add padding for the multi stream scenario. Padding allows the
  // startings sample to be the same for all SUTs, independent of the value
  // of samples_per_query, while enabling samples in a query to be contiguous.
  for (auto& loadable_set : result) {
    auto& set = loadable_set.set;
    for (size_t i = 0; i < set_padding; i++) {
      // It's not clear in the spec if the STL deallocates the old container
      // before assigning, which would invalidate the source before the
      // assignment happens. Even though we should have reserved enough
      // elements above, copy the source first anyway since we are just moving
      // integers around.
      QuerySampleIndex p = set[i];
      set.push_back(p);
    }
  }

  return result;
}

/// \brief Opens and owns handles to all of the log files.
struct LogOutputs {
  LogOutputs(const LogOutputSettings& output_settings,
             const std::string& test_date_time) {
    std::string prefix = output_settings.outdir;
    prefix += "/" + output_settings.prefix;
    if (output_settings.prefix_with_datetime) {
      prefix += test_date_time + "_";
    }
    const std::string& suffix = output_settings.suffix;

    summary_out.open(prefix + "summary" + suffix + ".txt");
    detail_out.open(prefix + "detail" + suffix + ".txt");
    accuracy_out.open(prefix + "accuracy" + suffix + ".json");
    trace_out.open(prefix + "trace" + suffix + ".json");
  }

  bool CheckOutputs() {
    bool all_ofstreams_good = true;
    if (!summary_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open summary file.";
    }
    if (!detail_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open detailed log file.";
    }
    if (!accuracy_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open accuracy log file.";
    }
    if (!trace_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open trace file.";
    }
    return all_ofstreams_good;
  }

  std::ofstream summary_out;
  std::ofstream detail_out;
  std::ofstream accuracy_out;
  std::ofstream trace_out;
};

/// \brief Find boundaries of performance settings by widening bounds
/// exponentially.
/// \details To find an upper bound of performance, widen an
/// upper bound exponentially until finding a bound that can't satisfy
/// performance constraints. i.e. [1, 2) -> [2, 4) -> [4, 8) -> ...
template <TestScenario scenario>
std::pair<PerformanceSummary, PerformanceSummary> FindBoundaries(
    SystemUnderTest* sut, QuerySampleLibrary* qsl, SequenceGen* sequence_gen,
    PerformanceSummary l_perf_summary) {
  // Get upper bound
  TestSettingsInternal u_settings = l_perf_summary.settings;
  find_peak_performance::WidenPerformanceField<scenario>(&u_settings);

  LogDetail(
      [l_field = find_peak_performance::ToStringPerformanceField<scenario>(
           l_perf_summary.settings),
       u_field = find_peak_performance::ToStringPerformanceField<scenario>(
           u_settings)](AsyncDetail& detail) {
        detail("FindBoundaries: Checking fields [" + l_field + ", " + u_field +
               ")");
      });

  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, u_settings));
  const LoadableSampleSet& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set.set);

  PerformanceResult u_pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, u_settings, performance_set, sequence_gen));
  PerformanceSummary u_perf_summary{sut->Name(), u_settings, std::move(u_pr)};

  qsl->UnloadSamplesFromRam(performance_set.set);

  std::string tmp;
  if (!u_perf_summary.PerfConstraintsMet(&tmp)) {
    return std::make_pair(l_perf_summary, u_perf_summary);
  } else {
    return FindBoundaries<scenario>(sut, qsl, sequence_gen, u_perf_summary);
  }
}

/// \brief Find peak performance by binary search.
/// \details The found lower & upper bounds by the function 'FindBoundaries' are
/// used as initial bounds of binary search
template <TestScenario scenario>
PerformanceSummary FindPeakPerformanceBinarySearch(
    SystemUnderTest* sut, QuerySampleLibrary* qsl, SequenceGen* sequence_gen,
    const LoadableSampleSet& performance_set, PerformanceSummary l_perf_summary,
    PerformanceSummary u_perf_summary) {
  if (find_peak_performance::IsFinished<scenario>(l_perf_summary.settings,
                                                  u_perf_summary.settings)) {
    return l_perf_summary;
  }

  const TestSettingsInternal m_settings =
      find_peak_performance::MidOfBoundaries<scenario>(l_perf_summary.settings,
                                                       u_perf_summary.settings);

  LogDetail([l_field =
                 find_peak_performance::ToStringPerformanceField<scenario>(
                     l_perf_summary.settings),
             u_field =
                 find_peak_performance::ToStringPerformanceField<scenario>(
                     u_perf_summary.settings),
             m_field =
                 find_peak_performance::ToStringPerformanceField<scenario>(
                     m_settings)](AsyncDetail& detail) {
    detail(
        "FindPeakPerformanceBinarySearch: Testing the mid value of bounds [" +
        l_field + ", " + u_field + "): " + m_field);
  });

  PerformanceResult m_pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, m_settings, performance_set, sequence_gen));
  PerformanceSummary m_perf_summary{sut->Name(), m_settings, std::move(m_pr)};

  std::string tmp;
  if (m_perf_summary.PerfConstraintsMet(&tmp)) {
    return FindPeakPerformanceBinarySearch<scenario>(
        sut, qsl, sequence_gen, performance_set, m_perf_summary,
        u_perf_summary);
  } else {
    return FindPeakPerformanceBinarySearch<scenario>(
        sut, qsl, sequence_gen, performance_set, l_perf_summary,
        m_perf_summary);
  }
}

/// \brief Runs the performance mode, templated by scenario.
template <TestScenario scenario>
void RunPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                        const TestSettingsInternal& settings,
                        SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) { detail("Starting performance mode:"); });

  // Use first loadable set as the performance set.
  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, settings));
  const LoadableSampleSet& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set.set);

  // Start PerfClock/system_clock timers for measuring performance interval
  // for comparison vs external timer.
  auto pc_start_ts = PerfClock::now();
  auto sc_start_ts = std::chrono::system_clock::now();
  if (settings.print_timestamps) {
    std::cout << "Loadgen :: Perf mode start. system_clock Timestamp = "
              << std::chrono::system_clock::to_time_t(sc_start_ts) << "\n"
              << std::flush;
  }

  PerformanceResult pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, settings, performance_set, sequence_gen));

  // Measure PerfClock/system_clock timer durations for comparison vs
  // external timer.
  auto pc_stop_ts = PerfClock::now();
  auto sc_stop_ts = std::chrono::system_clock::now();
  auto pc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                         pc_stop_ts - pc_start_ts)
                         .count();
  auto sc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                         sc_stop_ts - sc_start_ts)
                         .count();
  float pc_sc_ratio = static_cast<float>(pc_duration) / sc_duration;
  if (settings.print_timestamps) {
    std::cout << "Loadgen :: Perf mode stop. systme_clock Timestamp = "
              << std::chrono::system_clock::to_time_t(sc_stop_ts) << "\n"
              << std::flush;
    std::cout << "Loadgen :: PerfClock Perf duration = " << pc_duration
              << "ms\n"
              << std::flush;
    std::cout << "Loadgen :: system_clock Perf duration = " << sc_duration
              << "ms\n"
              << std::flush;
    std::cout << "Loadgen :: PerfClock/system_clock ratio = " << std::fixed
              << std::setprecision(4) << pc_sc_ratio << "\n"
              << std::flush;
  }

  if (pc_sc_ratio > 1.01 || pc_sc_ratio < 0.99) {
    LogDetail([pc_sc_ratio](AsyncDetail& detail) {
      detail.Error("PerfClock and system_clock differ by more than 1%! ",
                   "pc_sc_ratio", pc_sc_ratio);
    });
  } else if (pc_sc_ratio > 1.001 || pc_sc_ratio < 0.999) {
    LogDetail([pc_sc_ratio](AsyncDetail& detail) {
      detail.Warning("PerfClock and system_clock differ by more than 0.1%. ",
                     "pc_sc_ratio", pc_sc_ratio);
    });
  }

  sut->ReportLatencyResults(pr.sample_latencies);

  LogSummary(
      [perf_summary = PerformanceSummary{sut->Name(), settings, std::move(pr)}](
          AsyncSummary& summary) mutable { perf_summary.Log(summary); });

  qsl->UnloadSamplesFromRam(performance_set.set);
}

/// \brief Runs the binary search mode, templated by scenario.
/// \details 1. Check whether lower bound from user satisfies the performance
/// constraints, 2. Find an upper bound using the function 'FindBoundaries'
/// based on the lower bound, 3. Find peak performance settings using the
/// function 'FindPeakPerformanceBinarySearch'. note: Since we can't find a
/// lower bound programmatically because of the monotonicity issue of Server
/// scenario, rely on user's settings. After resolving this issue, we can
/// make the function 'FindBoundaries' find a lower bound as well from some
/// random initial settings.
template <TestScenario scenario>
void FindPeakPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                             const TestSettingsInternal& base_settings,
                             SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) {
    detail("Starting FindPeakPerformance mode:");
  });

  if (scenario != TestScenario::MultiStream &&
      scenario != TestScenario::MultiStreamFree &&
      scenario != TestScenario::Server) {
    LogDetail([unsupported_scenario = ToString(scenario)](AsyncDetail& detail) {
      detail.Error(find_peak_performance::kNotSupportedMsg);
    });
    return;
  }

  LogDetail([base_field =
                 find_peak_performance::ToStringPerformanceField<scenario>(
                     base_settings)](AsyncDetail& detail) {
    detail("FindPeakPerformance: Check validity of the base settings field: " +
           base_field);
  });

  // 1. Check whether the lower bound came from user satisfy performance
  // constraints or not.
  std::vector<loadgen::LoadableSampleSet> base_loadable_sets(
      loadgen::GenerateLoadableSets(qsl, base_settings));
  const LoadableSampleSet& base_performance_set = base_loadable_sets.front();
  LoadSamplesToRam(qsl, base_performance_set.set);

  PerformanceResult base_pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, base_settings, base_performance_set, sequence_gen));
  PerformanceSummary base_perf_summary{sut->Name(), base_settings,
                                       std::move(base_pr)};

  // We can also use all_constraints_met to check performance constraints,
  // but to reduce searching time, leave it up to whether the settings satisfy
  // min duration & min queries or not to users.
  std::string msg;
  if (!base_perf_summary.PerfConstraintsMet(&msg)) {
    LogDetail([msg](AsyncDetail& detail) {
      detail.Error(
          "FindPeakPerformance: Initial lower bound does not satisfy "
          "performance constraints, msg: " +
          msg);
    });

    sut->ReportLatencyResults(base_perf_summary.pr.sample_latencies);

    LogSummary(
        [perf_summary = PerformanceSummary{sut->Name(), base_settings,
                                           std::move(base_perf_summary.pr)}](
            AsyncSummary& summary) mutable { perf_summary.Log(summary); });

    qsl->UnloadSamplesFromRam(base_performance_set.set);

    return;
  }

  // Clear loaded samples.
  qsl->UnloadSamplesFromRam(base_performance_set.set);

  // 2. Find an upper bound based on the lower bound.
  std::pair<PerformanceSummary, PerformanceSummary> boundaries =
      FindBoundaries<scenario>(sut, qsl, sequence_gen, base_perf_summary);
  PerformanceSummary l_perf_summary = boundaries.first;
  PerformanceSummary u_perf_summary = boundaries.second;

  LogDetail(
      [l_field = find_peak_performance::ToStringPerformanceField<scenario>(
           l_perf_summary.settings),
       u_field = find_peak_performance::ToStringPerformanceField<scenario>(
           u_perf_summary.settings)](AsyncDetail& detail) {
        detail("FindPeakPerformance: Found boundaries: [" + l_field + ", " +
               u_field + ")");
      });

  // Reuse performance_set, u_perf_summary has the largest 'samples_per_query'.
  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, u_perf_summary.settings));
  const LoadableSampleSet& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set.set);

  // 3. Find peak performance settings using the found boundaries
  PerformanceSummary perf_summary = FindPeakPerformanceBinarySearch<scenario>(
      sut, qsl, sequence_gen, performance_set, l_perf_summary, u_perf_summary);

  // Print-out the peak performance test setting.
  LogDetail([field = find_peak_performance::ToStringPerformanceField<scenario>(
                 perf_summary.settings)](AsyncDetail& detail) {
    detail("FindPeakPerformance: Found peak performance field: " + field);
  });

  sut->ReportLatencyResults(perf_summary.pr.sample_latencies);

  LogSummary(
      [perf_summary = PerformanceSummary{sut->Name(), perf_summary.settings,
                                         std::move(perf_summary.pr)}](
          AsyncSummary& summary) mutable { perf_summary.Log(summary); });

  qsl->UnloadSamplesFromRam(performance_set.set);
}

/// \brief Runs the accuracy mode, templated by scenario.
template <TestScenario scenario>
void RunAccuracyMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                     const TestSettingsInternal& settings,
                     SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) { detail("Starting accuracy mode:"); });

  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, settings));

  for (auto& loadable_set : loadable_sets) {
    {
      auto tracer = MakeScopedTracer(
          [count = loadable_set.set.size()](AsyncTrace& trace) {
            trace("LoadSamples", "count", count);
          });
      LoadSamplesToRam(qsl, loadable_set.set);
    }

    PerformanceResult pr(IssueQueries<scenario, TestMode::AccuracyOnly>(
        sut, settings, loadable_set, sequence_gen));

    {
      auto tracer = MakeScopedTracer(
          [count = loadable_set.set.size()](AsyncTrace& trace) {
            trace("UnloadSampes", "count", count);
          });
      qsl->UnloadSamplesFromRam(loadable_set.set);
    }
  }
}

/// \brief Routes runtime scenario requests to the corresponding instances
/// of its templated mode functions.
struct RunFunctions {
  using Signature = void(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                         const TestSettingsInternal& settings,
                         SequenceGen* sequence_gen);

  template <TestScenario compile_time_scenario>
  static RunFunctions GetCompileTime() {
    return {(RunAccuracyMode<compile_time_scenario>),
            (RunPerformanceMode<compile_time_scenario>),
            (FindPeakPerformanceMode<compile_time_scenario>)};
  }

  static RunFunctions Get(TestScenario run_time_scenario) {
    switch (run_time_scenario) {
      case TestScenario::SingleStream:
        return GetCompileTime<TestScenario::SingleStream>();
      case TestScenario::MultiStream:
        return GetCompileTime<TestScenario::MultiStream>();
      case TestScenario::MultiStreamFree:
        return GetCompileTime<TestScenario::MultiStreamFree>();
      case TestScenario::Server:
        return GetCompileTime<TestScenario::Server>();
      case TestScenario::Offline:
        return GetCompileTime<TestScenario::Offline>();
    }
    // We should not reach this point.
    assert(false);
    return GetCompileTime<TestScenario::SingleStream>();
  }

  Signature& accuracy;
  Signature& performance;
  Signature& find_peak_performance;
};

}  // namespace loadgen

void StartTest(SystemUnderTest* sut, QuerySampleLibrary* qsl,
               const TestSettings& requested_settings,
               const LogSettings& log_settings) {
  GlobalLogger().StartIOThread();

  const std::string test_date_time = CurrentDateTimeISO8601();

  loadgen::LogOutputs log_outputs(log_settings.log_output, test_date_time);
  if (!log_outputs.CheckOutputs()) {
    return;
  }

  GlobalLogger().StartLogging(&log_outputs.summary_out, &log_outputs.detail_out,
                              &log_outputs.accuracy_out,
                              log_settings.log_output.copy_detail_to_stdout,
                              log_settings.log_output.copy_summary_to_stdout);

  if (log_settings.enable_trace) {
    GlobalLogger().StartNewTrace(&log_outputs.trace_out, PerfClock::now());
  }

  LogDetail([sut, qsl, test_date_time](AsyncDetail& detail) {
    detail("Date + time of test: ", test_date_time);
    detail("System Under Test (SUT) name: ", sut->Name());
    detail("Query Sample Library (QSL) name: ", qsl->Name());
    detail("QSL total size: ", qsl->TotalSampleCount());
    detail("QSL performance size*: ", qsl->PerformanceSampleCount());
    detail("*TestSettings (performance_sample_count_override) can override");
    detail("*Refer to Effective Settings for actual value");
  });

  TestSettings test_settings = requested_settings;
  // Look for Audit Config file to override TestSettings during audit
  const std::string audit_config_filename = "audit.config";
  if (FileExists(audit_config_filename)) {
    LogDetail([](AsyncDetail& detail) {
      detail(
          "Found Audit Config file (audit.config)."
          " Overriding TestSettings from audit.config file.");
    });
    std::string audit_scenario = loadgen::ToString(test_settings.scenario);
    // Remove Spaces from the string
    RemoveValue(&audit_scenario, ' ');
    const std::string generic_model = "*";
    test_settings.FromConfig(audit_config_filename, generic_model,
                             audit_scenario);
  }

  loadgen::TestSettingsInternal sanitized_settings(
      test_settings, qsl->PerformanceSampleCount());
  sanitized_settings.LogAllSettings();

  auto run_funcs = loadgen::RunFunctions::Get(sanitized_settings.scenario);

  loadgen::SequenceGen sequence_gen;
  switch (sanitized_settings.mode) {
    case TestMode::SubmissionRun:
      run_funcs.accuracy(sut, qsl, sanitized_settings, &sequence_gen);
      run_funcs.performance(sut, qsl, sanitized_settings, &sequence_gen);
      break;
    case TestMode::AccuracyOnly:
      run_funcs.accuracy(sut, qsl, sanitized_settings, &sequence_gen);
      break;
    case TestMode::PerformanceOnly:
      run_funcs.performance(sut, qsl, sanitized_settings, &sequence_gen);
      break;
    case TestMode::FindPeakPerformance:
      run_funcs.find_peak_performance(sut, qsl, sanitized_settings,
                                      &sequence_gen);
      break;
  }

  // Stop tracing after logging so all logs are captured in the trace.
  GlobalLogger().StopLogging();
  GlobalLogger().StopTracing();
  GlobalLogger().StopIOThread();
}

void QuerySamplesComplete(QuerySampleResponse* responses,
                          size_t response_count) {
  PerfClock::time_point timestamp = PerfClock::now();

  auto tracer = MakeScopedTracer(
      [](AsyncTrace& trace) { trace("QuerySamplesComplete"); });

  const QuerySampleResponse* end = responses + response_count;

  // Notify first to unblock loadgen production ASAP.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    loadgen::SampleMetadata* sample =
        reinterpret_cast<loadgen::SampleMetadata*>(response->id);
    loadgen::QueryMetadata* query = sample->query_metadata;
    query->NotifyOneSampleCompleted(timestamp);
  }

  // Log samples.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    loadgen::SampleMetadata* sample =
        reinterpret_cast<loadgen::SampleMetadata*>(response->id);
    loadgen::QueryMetadata* query = sample->query_metadata;
    query->response_delegate->SampleComplete(sample, response, timestamp);
  }
}

}  // namespace mlperf
