// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_cxx_api.h>
#include "test_configuration.h"
#include "test_session.h"
class TestModelInfo;
namespace onnxruntime {
namespace perftest {
class OnnxRuntimeTestSession : public TestSession {
 public:
  OnnxRuntimeTestSession(OrtEnv* env, PerformanceTestConfig& performance_test_config, const TestModelInfo* m);

  ~OnnxRuntimeTestSession() override {
    if (session_object_ != nullptr) OrtReleaseSession(session_object_);
    for (char* p : input_names_) {
      free(p);
    }
  }
  std::chrono::duration<double> Run(const OrtValue* const* input) override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeTestSession);

 private:
  OrtSession* session_object_ = nullptr;
  std::vector<std::string> output_names_;
  // The same size with output_names_.
  // TODO: implement a customized allocator, then we can remove output_names_ to simplify this code
  std::vector<const char*> output_names_raw_ptr;
  std::vector<OrtValue*> output_values_;
  std::vector<char*> input_names_;
};

}  // namespace perftest
}  // namespace onnxruntime