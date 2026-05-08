// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

constexpr const ORTCHAR_T* kSqueezeNetModelUri =
    ORT_TSTR("testdata/squeezenet/model.onnx");

class OVEPWorkloadTypeTests : public ::testing::Test {
 protected:
  // Check whether the NPU device can be registered at all.
  static bool IsNPUAvailable() {
    try {
      Ort::SessionOptions opts;
      std::unordered_map<std::string, std::string> ov;
      ov["device_type"] = "NPU";
      opts.AppendExecutionProvider_OpenVINO_V2(ov);
      return true;
    } catch (...) {
      return false;
    }
  }

  // Allow NPU resources to be fully released between tests.
  // Without this delay the NPU driver may fail to re-initialise
  void TearDown() override {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  static Ort::Session CreateSqueezeNetSession(
      Ort::SessionOptions& session_options,
      std::unordered_map<std::string, std::string>& ov_options) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.AppendExecutionProvider_OpenVINO_V2(ov_options);

    return Ort::Session(*ort_env, kSqueezeNetModelUri, session_options);
  }

  // Run a single inference on the SqueezeNet session and return output
  static std::vector<float> RunSqueezeNet(Ort::Session& session,
                                          const std::string& phase_label) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name =
        session.GetInputNameAllocated(0, allocator).get();
    std::string output_name =
        session.GetOutputNameAllocated(0, allocator).get();
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};

    // SqueezeNet input: 1 × 3 × 224 × 224 = 150 528 floats
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    constexpr size_t kInputSize = 1 * 3 * 224 * 224;
    std::vector<float> input_values(kInputSize, 1.0f);

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_values.data(), input_values.size(),
        input_shape.data(), input_shape.size());

    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                               &input_tensor, 1, output_names, 1);

    EXPECT_EQ(outputs.size(), 1u) << phase_label;
    if (outputs.empty()) return {};

    auto type_shape = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_shape = type_shape.GetShape();

    // Expected: [1, 1000, 1, 1]
    EXPECT_EQ(out_shape.size(), 4u) << phase_label;
    if (out_shape.size() == 4u) {
      EXPECT_EQ(out_shape[0], 1) << phase_label;
      EXPECT_EQ(out_shape[1], 1000) << phase_label;
      EXPECT_EQ(out_shape[2], 1) << phase_label;
      EXPECT_EQ(out_shape[3], 1) << phase_label;
    }

    size_t num_elements = type_shape.GetElementCount();
    EXPECT_EQ(num_elements, 1000u) << phase_label;

    const float* out_data = outputs[0].GetTensorData<float>();
    std::vector<float> result(out_data, out_data + num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
      EXPECT_TRUE(std::isfinite(result[i]))
          << phase_label << " index " << i << " is not finite";
    }

    return result;
  }

  // Compare two output vectors element-wise within a tolerance.
  static void CompareOutputs(const std::vector<float>& expected,
                             const std::vector<float>& actual,
                             const std::string& label,
                             float tolerance = 1e-4f) {
    ASSERT_EQ(expected.size(), actual.size()) << label << " size mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << label << " mismatch at index " << i;
    }
  }
};

namespace onnxruntime {
namespace test {

// Test 1: Dynamic workload-type switching with consistency check
//   Baseline (no workload type) → Efficient → Default
TEST_F(OVEPWorkloadTypeTests, OVEPWorkloadTypeDynamicSwitch) {
  if (!IsNPUAvailable()) {
    GTEST_SKIP() << "NPU device not available, skipping workload type test";
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"] = "NPU";

  Ort::Session session = CreateSqueezeNetSession(session_options, ov_options);

  const char* const keys[] = {"ep.dynamic.workload_type"};

  // Phase 1: Baseline (no workload type set)
  auto baseline_output = RunSqueezeNet(session, "Baseline");

  // Phase 2: Switch to Efficient
  const char* const eff_val[] = {"Efficient"};
  session.SetEpDynamicOptions(keys, eff_val, 1);
  auto efficient_output = RunSqueezeNet(session, "Efficient");

  // Phase 3: Switch to Default
  const char* const def_val[] = {"Default"};
  session.SetEpDynamicOptions(keys, def_val, 1);
  auto default_output = RunSqueezeNet(session, "Default");

  // All modes should produce the same results
  CompareOutputs(baseline_output, efficient_output,
                 "Baseline vs Efficient");
  CompareOutputs(baseline_output, default_output,
                 "Baseline vs Default");
}

// Test 2: Multiple inferences per workload mode
// Runs 10 inferences in each mode:
// Baseline × 10 → Efficient × 10 → Default × 10
TEST_F(OVEPWorkloadTypeTests, OVEPWorkloadTypeMultipleInferencesPerMode) {
  if (!IsNPUAvailable()) {
    GTEST_SKIP() << "NPU device not available, skipping workload type test";
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"] = "NPU";

  Ort::Session session = CreateSqueezeNetSession(session_options, ov_options);

  const char* const keys[] = {"ep.dynamic.workload_type"};
  const char* const eff_val[] = {"Efficient"};
  const char* const def_val[] = {"Default"};

  constexpr int kIterationsPerMode = 10;

  // Phase 1: Baseline – 10 runs without workload type
  // Save the first run as the reference output.
  auto reference_output = RunSqueezeNet(session, "Baseline iter 0");
  for (int i = 1; i < kIterationsPerMode; ++i) {
    auto output = RunSqueezeNet(session, "Baseline iter " + std::to_string(i));
    CompareOutputs(reference_output, output,
                   "Baseline iter " + std::to_string(i) + " vs reference");
  }

  // Phase 2: Efficient – 10 runs
  session.SetEpDynamicOptions(keys, eff_val, 1);
  for (int i = 0; i < kIterationsPerMode; ++i) {
    auto output = RunSqueezeNet(session, "Efficient iter " + std::to_string(i));
    CompareOutputs(reference_output, output,
                   "Efficient iter " + std::to_string(i) + " vs reference");
  }

  // Phase 3: Default – 10 runs
  session.SetEpDynamicOptions(keys, def_val, 1);
  for (int i = 0; i < kIterationsPerMode; ++i) {
    auto output = RunSqueezeNet(session, "Default iter " + std::to_string(i));
    CompareOutputs(reference_output, output,
                   "Default iter " + std::to_string(i) + " vs reference");
  }
}

}  // namespace test
}  // namespace onnxruntime
