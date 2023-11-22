// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_cxx_api.h>
#include <random>
#include "test_configuration.h"
#include "test_session.h"
class TestModelInfo;
namespace onnxruntime {
namespace perftest {
class OnnxRuntimeTestSession : public TestSession {
 public:
  OnnxRuntimeTestSession(Ort::Env& env, std::random_device& rd, const PerformanceTestConfig& performance_test_config,
                         const TestModelInfo& m);

  void PreLoadTestData(size_t test_data_id, size_t input_id, Ort::Value&& value) override {
    if (test_inputs_.size() < test_data_id + 1) {
      test_inputs_.resize(test_data_id + 1);
    }
    if (test_inputs_.at(test_data_id).size() == 0) {
      for (int i = 0; i < input_length_; i++)
        test_inputs_[test_data_id].emplace_back(nullptr);
    }
    test_inputs_[test_data_id][input_id] = std::move(value);
  }

  bool PopulateGeneratedInputTestData(int32_t seed);

  ~OnnxRuntimeTestSession() = default;

  std::chrono::duration<double> Run() override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeTestSession);

 private:
  Ort::Session session_{nullptr};
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<int> dist_;
  std::vector<std::vector<Ort::Value>> test_inputs_;
  std::vector<std::string> output_names_;
  // The same size with output_names_.
  // TODO: implement a customized allocator, then we can remove output_names_ to simplify this code
  std::vector<const char*> output_names_raw_ptr;
  std::vector<const char*> input_names_;
  std::vector<std::string> input_names_str_;
  const int input_length_;
  std::string provider_name_;
};

}  // namespace perftest
}  // namespace onnxruntime
