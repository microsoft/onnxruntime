// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_c_api.h>
#include "core/platform/env.h"
#include <core/platform/EigenNonBlockingThreadPool.h>
#include <random>
#include "test_configuration.h"
#include "heap_buffer.h"

#include "loadgen.h"
#include "test_settings.h"
#include "query_sample.h"
#include "system_under_test.h"
#include "query_sample_library.h"
#include "performance_result.h"

class ITestCase;
namespace onnxruntime {


namespace perftest {

//Load ONNX style protobuf data files into Ort::Value
class SampleLoader : public mlperf::QuerySampleLibrary {
 private:
  ITestCase* test_case_;
  std::vector<OrtValue*> inputs_;
  onnxruntime::test::HeapBuffer b_;
  std::vector<std::string> input_names_;
  size_t input_length_;
 public:
  SampleLoader(OrtSession* sess, ITestCase* test_case);

  OrtValue* const * GetInput(size_t test_data_id) const{
    return  inputs_.data() + test_data_id * input_length_;
  }

  const std::string& Name() const override;

  size_t TotalSampleCount() override;

  size_t PerformanceSampleCount() override;

  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;
};

class OnnxRuntimeTestSession : public mlperf::SystemUnderTest {
 public:
  OnnxRuntimeTestSession(OrtSession* sess, SampleLoader* sample_loader, std::random_device& rd, size_t concurrent_session_runs);

  ~OnnxRuntimeTestSession() override {
    for (char* p : input_names_) {
      free(p);
    }
  }

  /// \brief A human-readable string for logging purposes.
  const std::string& Name() const override {
    return name_;
  }

  /// \brief Lets the loadgen issue N samples to the SUT.
  /// \details The SUT may either a) return immediately and signal completion
  /// at a later time on another thread or b) it may block and signal
  /// completion on the current stack. The load generator will handle both
  /// cases properly.
  /// Note: The data for neighboring samples may or may not be contiguous
  /// depending on the scenario.
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;


  /// \brief Called immediately after the last call to IssueQuery
  /// in a series is made.
  /// \details This doesn't necessarily signify the end of the
  /// test since there may be multiple series involved during a test; for
  /// example in accuracy mode.
  /// Clients can use this to flush any deferred queries immediately, rather
  /// than waiting for some timeout.
  /// This is especially useful in the server scenario.
  void FlushQueries() override {

  }

  /// \brief Reports the raw latency results to the SUT of each sample issued as
  /// recorded by the load generator. Units are nanoseconds.
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {
    for(const mlperf::QuerySampleLatency& l:latencies_ns){
      performance_result_.time_costs.emplace_back(l);
      performance_result_.total_time_cost += l;
    }
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeTestSession);
  const perftest::PerformanceResult& GetPerformanceResult() const{
    return performance_result_;
  }
 private:
  const std::string name_  = "onnxruntime";
  OrtSession* sess_;
  SampleLoader* sample_loader_;
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<int> dist_;
  std::vector<std::string> output_names_;
  // The same size with output_names_.
  // TODO: implement a customized allocator, then we can remove output_names_ to simplify this code
  std::vector<const char*> output_names_raw_ptr;
  std::vector<char*> input_names_;
  size_t input_length_;
  perftest::PerformanceResult performance_result_;
  onnxruntime::ThreadOptions thread_options_;
  std::unique_ptr<onnxruntime::ThreadPoolTempl<onnxruntime::Env> > eigen_threadpool_;
};

OrtSession* CreateOrtSession(OrtEnv* env,
                             const PerformanceTestConfig& performance_test_config);

}  // namespace perftest
}  // namespace onnxruntime
