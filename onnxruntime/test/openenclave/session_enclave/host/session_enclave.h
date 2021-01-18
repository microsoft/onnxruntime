// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <chrono>
#include <openenclave/host.h>
#include <core/common/common.h>
#include <core/session/onnxruntime_cxx_api.h>

namespace onnxruntime {
namespace openenclave {

typedef struct OrtInferenceTimestamps {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
} OrtInferenceTimestamps;

class SessionEnclave {
 public:
  SessionEnclave(const std::string& enclave_path, bool debug = true, bool simulate = false);
  ~SessionEnclave();

  void CreateSession(const std::string& model_path, OrtLoggingLevel log_level,
                     bool enable_sequential_execution,
                     int intra_op_num_threads,
                     int inter_op_num_threads,
                     uint32_t optimization_level);

  void DestroySession();

  std::vector<Ort::Value> Run(const std::vector<Ort::Value>& inputs,
                              bool return_outputs = true, OrtInferenceTimestamps* timestamps = nullptr) const;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SessionEnclave);

 private:
  size_t output_count_;
  oe_enclave_t* enclave_;
};

}  // namespace openenclave
}  // namespace onnxruntime
