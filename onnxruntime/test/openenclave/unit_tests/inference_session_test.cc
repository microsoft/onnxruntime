// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"

#include "test/onnx/OrtValueList.h"
#include "test/openenclave/session_enclave/host/session_enclave.h"
#include "test_config.h"

#include "gtest/gtest.h"

using namespace onnxruntime::openenclave;

namespace onnxruntime {
namespace test {

static const std::string MODEL_URI = "testdata/mul_1.onnx";

void VerifyOutputs(std::vector<Ort::Value>& outputs, const std::vector<int64_t>& expected_dims,
                   const std::vector<float>& expected_values) {
  ASSERT_EQ(1, outputs.size());
  auto& output_tensor = outputs.at(0);

  auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), expected_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(expected_values.size(), total_len);

  float* f = output_tensor.GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(expected_values[i], f[i]);
  }
}

void RunModel(SessionEnclave& session_enclave) {
  Ort::AllocatorWithDefaultOptions default_allocator{};
  const OrtMemoryInfo* allocator_info = default_allocator.GetInfo();

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  Ort::Value input = Ort::Value::CreateTensor(allocator_info, values_mul_x.data(), values_mul_x.size(),
                                              dims_mul_x.data(), dims_mul_x.size());
  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(input));

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  std::vector<Ort::Value> outputs = session_enclave.Run(inputs);
  VerifyOutputs(outputs, expected_dims_mul_y, expected_values_mul_y);
}

// Test whether the following succeeds:
// - Initialize the enclave
// - Send model to enclave and create inference session inside enclave
// - Send inference input to the enclave
// - Run inference inside the enclave
// - Receive inference outputs from the enclave
// - Verify outputs against expected values
// - Destroy enclave

TEST(OpenEnclaveInferenceSessionTests, Sequential) {
  SessionEnclave session_enclave{ENCLAVE_PATH};
  bool enable_sequential_execution = true;
  int intra_op_num_threads = 1;
  int inter_op_num_threads = 1;
  uint32_t optimization_level = 2;
  session_enclave.CreateSession(MODEL_URI, ORT_LOGGING_LEVEL_VERBOSE,
                                enable_sequential_execution, 
                                intra_op_num_threads,
                                inter_op_num_threads,
                                optimization_level);

  RunModel(session_enclave);
}

TEST(OpenEnclaveInferenceSessionTests, Parallel) {
  SessionEnclave session_enclave{ENCLAVE_PATH};
  bool enable_sequential_execution = false;
  // 1 main thread + 5 intra threads + 2 inter threads = 8 threads.
  // Maximum allowed threads are defined in session_enclave.cc with OE_SET_ENCLAVE_SGX.
  // Note that setting threads to 0 would normally default to number of cores,
  // but Open Enclave returns 0 for std::thread::hardware_concurrency(),
  // so the threads have to be defined manually.
  int intra_op_num_threads = 5;
  int inter_op_num_threads = 2;
  uint32_t optimization_level = 2;
  session_enclave.CreateSession(MODEL_URI, ORT_LOGGING_LEVEL_VERBOSE,
                                enable_sequential_execution, 
                                intra_op_num_threads,
                                inter_op_num_threads,
                                optimization_level);

  RunModel(session_enclave);
}

}  // namespace test
}  // namespace onnxruntime
