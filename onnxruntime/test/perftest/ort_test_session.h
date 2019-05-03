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
  OnnxRuntimeTestSession(OrtEnv* env, std::random_device& rd, const PerformanceTestConfig& performance_test_config,
                         const TestModelInfo* m);

  void PreLoadTestData(size_t test_data_id, size_t input_id, OrtValue* value) override {
    if (test_inputs.size() < test_data_id + 1) {
      test_inputs.resize(test_data_id + 1);
    }
    if (test_inputs.at(test_data_id) == nullptr) {
      test_inputs[test_data_id] = new OrtValueArray(input_length);
    }
    test_inputs[test_data_id]->Set(input_id, value);
  }

  ~OnnxRuntimeTestSession() override {
    if (session_object_ != nullptr) OrtReleaseSession(session_object_);
    for (char* p : input_names_) {
      free(p);
    }
  }
  std::chrono::duration<double> Run() override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeTestSession);

 private:
  OrtSession* session_object_ = nullptr;
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<int> dist_;
  std::vector<OrtValueArray*> test_inputs;
  std::vector<std::string> output_names_;
  // The same size with output_names_.
  // TODO: implement a customized allocator, then we can remove output_names_ to simplify this code
  std::vector<const char*> output_names_raw_ptr;
  std::vector<OrtValue*> output_values_;
  std::vector<char*> input_names_;
  int input_length;
};

}  // namespace perftest
}  // namespace onnxruntime