// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <iomanip>

#include <core/session/onnxruntime_cxx_api.h>
#include "test_configuration.h"
#include "test_session.h"
#include "providers.h"
#include "TestCase.h"
#include "test/openenclave/session_enclave/host/session_enclave.h"

class TestModelInfo;
namespace onnxruntime {
namespace perftest {

using namespace onnxruntime::openenclave;

class OnnxRuntimeOpenEnclaveTestSession : public TestSession {
 public:
  OnnxRuntimeOpenEnclaveTestSession(std::random_device& rd,
                                    const PerformanceTestConfig& performance_test_config, const TestModelInfo& m)
      : rand_engine_(rd()), input_length_(m.GetInputCount()), session_enclave_(performance_test_config.machine_config.enclave_file_path, true /* debug */, performance_test_config.run_config.enable_openenclave_simulation) {
    inference_timestamps_path_ = performance_test_config.model_info.result_file_path + ".inference_timestamps";

    const std::string& provider_name = performance_test_config.machine_config.provider_type_name;

    if (!provider_name.empty() && provider_name != onnxruntime::kCpuExecutionProvider) {
      ORT_THROW("This backend only supports the CPU provider.\n");
    }

    if (!performance_test_config.run_config.profile_file.empty()) {
      ORT_THROW("This backend does not support profile files.\n");
    }

    session_enclave_.CreateSession(performance_test_config.model_info.model_file_path,
                                   performance_test_config.run_config.f_verbose ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING,
                                   performance_test_config.run_config.execution_mode == ExecutionMode::ORT_SEQUENTIAL,
                                   performance_test_config.run_config.intra_op_num_threads,
                                   performance_test_config.run_config.inter_op_num_threads,
                                   performance_test_config.run_config.optimization_level);
  }

  ~OnnxRuntimeOpenEnclaveTestSession() override {
    session_enclave_.DestroySession();

    std::ofstream outfile(inference_timestamps_path_);
    if (!outfile.good()) {
      std::cerr << "failed to open inference timestamps file" << std::endl;
      return;
    }
    outfile << "start,end" << std::endl;
    for (auto& ts : inference_timestamps_) {
      auto start_unix_ts = std::chrono::duration_cast<std::chrono::microseconds>(ts.start.time_since_epoch()).count() * 1.0e-6;
      auto end_unix_ts = std::chrono::duration_cast<std::chrono::microseconds>(ts.end.time_since_epoch()).count() * 1.0e-6;
      outfile << std::fixed << start_unix_ts << "," << end_unix_ts << std::endl;
    }
  }

  std::chrono::duration<double> Run() override {
    //Randomly pick one input set from test_inputs_. (NOT ThreadSafe)
    const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(test_inputs_.size() - 1));
    const size_t id = static_cast<size_t>(dist_(rand_engine_, p));
    auto& input = test_inputs_.at(id);

    OrtInferenceTimestamps timestamps;
    session_enclave_.Run(input, false /* return_outputs */, &timestamps);

    std::chrono::duration<double> duration_seconds = timestamps.end - timestamps.start;

    if (inference_timestamps_.size() < 10) {
      inference_timestamps_.emplace_back(std::move(timestamps));
    }

    return duration_seconds;
  }

  void PreLoadTestData(size_t test_data_id, size_t input_id, Ort::Value&& value) override {
    if (test_inputs_.size() < test_data_id + 1) {
      test_inputs_.resize(test_data_id + 1);
    }
    if (test_inputs_.at(test_data_id).size() == 0) {
      for (size_t i = 0; i < input_length_; i++)
        test_inputs_[test_data_id].emplace_back(nullptr);
    }
    test_inputs_[test_data_id][input_id] = std::move(value);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeOpenEnclaveTestSession);

 private:
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<int> dist_;
  std::vector<std::vector<Ort::Value>> test_inputs_;
  const size_t input_length_;

  SessionEnclave session_enclave_;
  std::vector<OrtInferenceTimestamps> inference_timestamps_;
  std::string inference_timestamps_path_;
};

}  // namespace perftest
}  // namespace onnxruntime