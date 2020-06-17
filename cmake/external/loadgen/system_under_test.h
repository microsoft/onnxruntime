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
/// \brief Defines the SystemUnderTest interface.

#ifndef MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H
#define MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H

#include <string>
#include <vector>

#include "query_sample.h"

namespace mlperf {

/// \addtogroup LoadgenAPI
/// @{

/// \brief The interface a client implements for the loadgen to test.
/// \todo Add hook for an untimed warm up period for the SUT.
/// \todo Add hook for an untimed warm up period for the loadgen logic.
/// \todo Support power hooks for cool-down period before runing performance
/// traffic.
/// \todo Support power hooks for correlating test timeline with power
/// measurment timeline.
class SystemUnderTest {
 public:
  virtual ~SystemUnderTest() {}

  /// \brief A human-readable string for logging purposes.
  virtual const std::string& Name() const = 0;

  /// \brief Lets the loadgen issue N samples to the SUT.
  /// \details The SUT may either a) return immediately and signal completion
  /// at a later time on another thread or b) it may block and signal
  /// completion on the current stack. The load generator will handle both
  /// cases properly.
  /// Note: The data for neighboring samples may or may not be contiguous
  /// depending on the scenario.
  virtual void IssueQuery(const std::vector<QuerySample>& samples) = 0;

  /// \brief Called immediately after the last call to IssueQuery
  /// in a series is made.
  /// \details This doesn't necessarily signify the end of the
  /// test since there may be multiple series involved during a test; for
  /// example in accuracy mode.
  /// Clients can use this to flush any deferred queries immediately, rather
  /// than waiting for some timeout.
  /// This is especially useful in the server scenario.
  virtual void FlushQueries() = 0;

  /// \brief Reports the raw latency results to the SUT of each sample issued as
  /// recorded by the load generator. Units are nanoseconds.
  virtual void ReportLatencyResults(
      const std::vector<QuerySampleLatency>& latencies_ns) = 0;
};

/// @}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_SYSTEM_UNDER_TEST_H
