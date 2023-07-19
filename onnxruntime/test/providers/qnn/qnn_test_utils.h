// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <string>
#include <unordered_map>
#include "core/framework/provider_options.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/util/include/test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

using GetTestModelFn = std::function<void(ModelTestBuilder& builder)>;

// Class that defines an input that can be created with ModelTestBuilder.
// Defines whether the input is an initializer and if the data should be randomized or if
// set to an explicit value.
template <typename T>
struct TestInputDef {
  struct RawData {
    std::vector<T> data;
  };

  struct RandomData {
    T min;
    T max;
  };

  TestInputDef() : is_initializer_(false) {}

  // Creates a random input definition. Specify its shape, whether it's an initializer, and
  // the min/max range.
  TestInputDef(std::vector<int64_t> shape, bool is_initializer, T rand_min, T rand_max)
      : shape_(std::move(shape)),
        data_info_(RandomData{rand_min, rand_max}),
        is_initializer_(is_initializer) {}

  // Create an input definition with explicit data. Specify its shape, whether it's an initializer,
  // and the raw data.
  TestInputDef(std::vector<int64_t> shape, bool is_initializer, std::vector<T> data)
      : shape_(std::move(shape)),
        data_info_(RawData{std::move(data)}),
        is_initializer_(is_initializer) {}

  TestInputDef(TestInputDef&& other) = default;
  TestInputDef(const TestInputDef& other) = default;

  TestInputDef& operator=(const TestInputDef& other) = default;
  TestInputDef& operator=(TestInputDef&& other) = default;

  const std::vector<int64_t>& GetShape() const {
    return shape_;
  }

  bool IsInitializer() const {
    return is_initializer_;
  }

  bool IsRandomData() const {
    return data_info_.index() == 1;
  }

  const RandomData& GetRandomDataInfo() const {
    return std::get<RandomData>(data_info_);
  }

  bool IsRawData() const {
    return data_info_.index() == 0;
  }

  const std::vector<T>& GetRawData() const {
    return std::get<RawData>(data_info_).data;
  }

  std::pair<T, T> GetRange() const {
    auto which_type = data_info_.index();
    std::pair<T, T> range;

    if (which_type == 0) {
      // Get min/max of raw data.
      range.first = std::numeric_limits<T>::max();
      range.second = std::numeric_limits<T>::min();

      for (auto val : std::get<RawData>(data_info_).data) {
        range.first = std::min(range.first, val);
        range.second = std::max(range.second, val);
      }
    } else {
      assert(which_type == 1);
      RandomData rand_info = std::get<RandomData>(data_info_);
      range.first = rand_info.min;
      range.second = rand_info.max;
    }

    return range;
  }

 private:
  std::vector<int64_t> shape_;
  std::variant<RawData, RandomData> data_info_;
  bool is_initializer_;
};

template <typename QType = uint8_t>
struct QuantParams {
  float scale;
  QType zero_point;

  static QuantParams<QType> Compute(float rmin, float rmax) {
    constexpr float qmin = static_cast<float>(std::numeric_limits<QType>::min());
    constexpr float qmax = static_cast<float>(std::numeric_limits<QType>::max());

    const float scale = (rmax - rmin) / (qmax - qmin);
    const QType zero_point = static_cast<QType>((qmin - rmin) / scale);

    return QuantParams<QType>{scale, zero_point};
  }
};

template <typename QType = uint8_t>
inline QuantParams<QType> GetTestInputQuantParams(const TestInputDef<float>& input_def) {
  const std::pair<float, float> frange = input_def.GetRange();
  return QuantParams<QType>::Compute(frange.first, frange.second);
}

/**
 * Creates and returns an input in a test model graph. The input's characteristics are defined
 * by the provided input definition.
 *
 * \param builder Model builder object used to build the model's inputs, outputs, and nodes.
 * \param input_def Input definition that describes what kind of input to create.
 * \return A pointer to the new input.
 */
template <typename T>
inline NodeArg* MakeTestInput(ModelTestBuilder& builder, const TestInputDef<T>& input_def) {
  NodeArg* input = nullptr;
  const auto& shape = input_def.GetShape();
  const bool is_initializer = input_def.IsInitializer();

  if (input_def.IsRawData()) {  // Raw data.
    const std::vector<T>& raw_data = input_def.GetRawData();

    if (is_initializer) {
      input = builder.MakeInitializer<T>(shape, raw_data);
    } else {
      input = builder.MakeInput<T>(shape, raw_data);
    }
  } else {  // Random data
    const auto& rand_info = input_def.GetRandomDataInfo();

    if (is_initializer) {
      input = builder.MakeInitializer<T>(shape, rand_info.min, rand_info.max);
    } else {
      input = builder.MakeInput<T>(shape, rand_info.min, rand_info.max);
    }
  }

  return input;
}

/**
 * Runs a test model on the QNN EP. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param build_test_case Function that builds a test model. See test/optimizer/qdq_test_utils.h
 * \param provider_options Provider options for QNN EP.
 * \param opset_version The opset version.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param fp32_abs_err The acceptable error between CPU EP and QNN EP.
 * \param log_severity The logger's minimum severity level.
 */
void RunQnnModelTest(const GetTestModelFn& build_test_case, const ProviderOptions& provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                     float fp32_abs_err = 1e-5f, logging::Severity log_severity = logging::Severity::kERROR);

enum class BackendSupport {
  SUPPORT_UNKNOWN,
  UNSUPPORTED,
  SUPPORTED,
  SUPPORT_ERROR,
};

// Testing fixture class for tests that require the QNN HTP backend. Checks if HTP is available before the test begins.
// The test is skipped if HTP is unavailable (may occur on Windows ARM64).
// TODO: Remove once HTP can be emulated on Windows ARM64.
class QnnHTPBackendTests : public ::testing::Test {
 protected:
  void SetUp() override;

  static BackendSupport cached_htp_support_;  // Set by the first test using this fixture.
};

// Testing fixture class for tests that require the QNN CPU backend. Checks if QNN CPU is available before the test
// begins. The test is skipped if the CPU backend is unavailable (may occur on Windows ARM64 VM).
// TODO: Remove once QNN CPU backend works on Windows ARM64 pipeline VM.
class QnnCPUBackendTests : public ::testing::Test {
 protected:
  void SetUp() override;

  static BackendSupport cached_cpu_support_;  // Set by the first test using this fixture.
};

/**
 * Returns true if the given reduce operator type (e.g., "ReduceSum") and opset version (e.g., 13)
 * supports "axes" as an input (instead of an attribute).
 *
 * \param op_type The string denoting the reduce operator's type (e.g., "ReduceSum").
 * \param opset_version The opset of the operator.
 *
 * \return True if "axes" is an input, or false if "axes" is an attribute.
 */
bool ReduceOpHasAxesInput(const std::string& op_type, int opset_version);

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)