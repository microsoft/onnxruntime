// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <string>
#include <cmath>
#include <unordered_map>
#include "core/framework/provider_options.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/util/include/test_utils.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/default_providers.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

using GetTestModelFn = std::function<void(ModelTestBuilder& builder)>;

template <typename QType = uint8_t>
struct QuantParams {
  float scale;
  QType zero_point;

  static QuantParams<QType> Compute(float rmin, float rmax) {
    constexpr float qmin = static_cast<float>(std::numeric_limits<QType>::min());
    constexpr float qmax = static_cast<float>(std::numeric_limits<QType>::max());

    const float scale = (rmax - rmin) / (qmax - qmin);
    const QType zero_point = static_cast<QType>(std::roundf((qmin - rmin) / scale));

    return QuantParams<QType>{scale, zero_point};
  }
};

template <typename QType = uint8_t>
inline QuantParams<QType> GetDataQuantParams(const std::vector<float>& data) {
  // Get min/max of raw data.
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::min();

  for (auto val : data) {
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
  }

  return QuantParams<QType>::Compute(min_val, max_val);
}

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
inline QuantParams<QType> GetTestInputQuantParams(const TestInputDef<float>& input_def) {
  const std::pair<float, float> frange = input_def.GetRange();
  return QuantParams<QType>::Compute(frange.first, frange.second);
}

template <typename QuantType>
using GetTestQDQModelFn = std::function<void(ModelTestBuilder& builder, const std::vector<QuantParams<QuantType>>& output_qparams)>;

/**
 * Inferences a given serialized model. Returns output values via an out-param.
 *
 * \param model_data The serialized ONNX model to inference.
 * \param log_id The logger ID.
 * \param execution_provider The EP on which to run the model. Set to nullptr for CPU EP.
 * \param expected_ep_assignment Describes "which nodes" should be assigned to the EP.
 * \param feeds The input feeds.
 * \param output_names If empty, the function will write the output names.
 * \param output_vals Initialized to the inference results.
 */
void InferenceModel(const std::string& model_data, const char* log_id,
                    std::unique_ptr<IExecutionProvider> execution_provider,
                    ExpectedEPNodeAssignment expected_ep_assignment, const NameMLValMap& feeds,
                    std::vector<std::string>& output_names, std::vector<OrtValue>& output_vals);

/**
 * Tests the accuracy of a QDQ model on QNN EP by runnning 3 inferences:
 *
 * 1. float model on CPU EP (baseline)
 * 2. QDQ model on CPU EP
 * 3. QDQ model on QNN EP
 *
 * This function checks that running the QDQ model on QNN EP (#3) is at least as accurate (+- small tolerance)
 * as running the QDQ model on CPU EP (#2). We primarily measure accuracy by comparing to the baseline (#1).
 *
 * \param f32_model_fn Function that builds the float model (baseline for comparison).
 * \param qdq_model_fn Function that builds the QDQ model (run by CPU EP and QNN EP).
 * \param qnn_options QNN EP provider options.
 * \param opset_version The opset version.
 * \param expected_ep_assignment Describes "which nodes" should be assigned to the EP.
 * \param fp32_abs_err Small tolerance used for floating-point comparisons.
 * \param log_severity The logger's severity setting.
 */
template <typename QuantType = uint8_t>
inline void TestQDQModelAccuracy(const GetTestModelFn& f32_model_fn, const GetTestQDQModelFn<QuantType>& qdq_model_fn,
                                 const ProviderOptions& qnn_options, int opset_version,
                                 ExpectedEPNodeAssignment expected_ep_assignment, float fp32_abs_err,
                                 logging::Severity log_severity = logging::Severity::kERROR) {
  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(log_severity);

  // Create float model and serialize it to a string.
  onnxruntime::Model f32_model("f32_model", false, ModelMetaData(), PathString(),
                               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                               logging_manager.DefaultLogger());
  ModelTestBuilder f32_helper(f32_model.MainGraph());
  std::string f32_model_data;
  f32_model_fn(f32_helper);
  f32_helper.SetGraphOutputs();
  ASSERT_STATUS_OK(f32_model.MainGraph().Resolve());
  f32_model.ToProto().SerializeToString(&f32_model_data);

  // Run f32 model on CPU EP and collect outputs.
  std::vector<OrtValue> cpu_f32_outputs;
  std::vector<std::string> output_names;
  InferenceModel(f32_model_data, "f32_model_logger", nullptr, ExpectedEPNodeAssignment::All,
                 f32_helper.feeds_, output_names, cpu_f32_outputs);
  const size_t num_outputs = cpu_f32_outputs.size();

  // Compute output range(s) and quantization params.
  std::vector<QuantParams<QuantType>> output_qparams;
  std::vector<gsl::span<const float>> output_vals;
  std::vector<int32_t> output_types;
  output_qparams.resize(num_outputs);
  output_vals.resize(num_outputs);
  output_types.resize(num_outputs);

  for (size_t i = 0; i < num_outputs; i++) {
    auto& tensor = cpu_f32_outputs[i].Get<Tensor>();
    int32_t elem_type = tensor.GetElementType();

    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      output_vals[i] = tensor.DataAsSpan<float>();
      float min_val = std::numeric_limits<float>::max();
      float max_val = std::numeric_limits<float>::min();

      for (auto val : output_vals[i]) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
      }

      output_qparams[i] = QuantParams<QuantType>::Compute(min_val, max_val);
    }

    output_types[i] = elem_type;
  }

  // Create QDQ model and serialize it to a string.
  onnxruntime::Model qdq_model("qdq_model", false, ModelMetaData(), PathString(),
                               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                               logging_manager.DefaultLogger());
  ModelTestBuilder qdq_helper(qdq_model.MainGraph());
  std::string qdq_model_data;
  qdq_model_fn(qdq_helper, output_qparams);
  qdq_helper.SetGraphOutputs();
  ASSERT_STATUS_OK(qdq_model.MainGraph().Resolve());
  qdq_model.ToProto().SerializeToString(&qdq_model_data);

  // Run QDQ model on QNN EP and collect outputs.
  std::vector<OrtValue> qnn_qdq_outputs;
  InferenceModel(qdq_model_data, "qdq_model_logger", QnnExecutionProviderWithOptions(qnn_options),
                 expected_ep_assignment, qdq_helper.feeds_, output_names, qnn_qdq_outputs);

  if (expected_ep_assignment != ExpectedEPNodeAssignment::None) {
    // Run QDQ model on CPU EP and collect outputs.
    std::vector<OrtValue> cpu_qdq_outputs;
    InferenceModel(qdq_model_data, "qdq_model_logger", nullptr, ExpectedEPNodeAssignment::All,
                   qdq_helper.feeds_, output_names, cpu_qdq_outputs);
    ASSERT_EQ(cpu_qdq_outputs.size(), num_outputs);
    ASSERT_EQ(qnn_qdq_outputs.size(), num_outputs);

    // Compare accuracy of QDQ results with float model.
    // QNN EP must be at least as accurate as CPU EP when running the QDQ model.
    for (size_t i = 0; i < num_outputs; i++) {
      auto& cpu_qdq_tensor = cpu_qdq_outputs[i].Get<Tensor>();
      auto& qnn_qdq_tensor = qnn_qdq_outputs[i].Get<Tensor>();

      ASSERT_EQ(cpu_qdq_tensor.GetElementType(), output_types[i]);
      ASSERT_EQ(qnn_qdq_tensor.GetElementType(), output_types[i]);

      if (output_types[i] == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        const size_t num_vals = output_vals[i].size();
        gsl::span<const float> cpu_f32_vals = output_vals[i];
        gsl::span<const float> cpu_qdq_vals = cpu_qdq_tensor.DataAsSpan<float>();
        gsl::span<const float> qnn_qdq_vals = qnn_qdq_tensor.DataAsSpan<float>();

        ASSERT_EQ(num_vals, cpu_qdq_vals.size());
        ASSERT_EQ(num_vals, qnn_qdq_vals.size());

        for (size_t j = 0; j < num_vals; j++) {
          const float expected_val = cpu_f32_vals[j];  // "ground-truth"
          const float qnn_val = qnn_qdq_vals[j];
          const float cpu_val = cpu_qdq_vals[j];
          const float cpu_err = std::fabsf(expected_val - cpu_val);
          const float qnn_err = std::fabsf(expected_val - qnn_val);
          const float qnn_cpu_qdq_err = std::fabsf(cpu_val - qnn_val);
          const float cpu_qdq_quant_dist = std::ceilf(cpu_err / output_qparams[i].scale);

          // The QDQ results from CPU EP and QNN EP are "some" quantized distance away from the ground-truth.
          // Are those quantized distances at most the same?
          const bool is_same_quant_dist_from_actual = qnn_err <= cpu_qdq_quant_dist * output_qparams[i].scale + fp32_abs_err;

          // Is the result from QNN EP 1 quantized unit away from the QDQ model on CPU EP?
          const bool is_1_quant_unit_from_cpu_qdq = qnn_cpu_qdq_err <= output_qparams[i].scale + fp32_abs_err;

          // Is the raw result from QNN EP closer (or same dist) to the ground-truth?
          const bool is_as_accurate_as_cpu_qdq = qnn_err <= cpu_err + fp32_abs_err;

          EXPECT_TRUE(is_same_quant_dist_from_actual || is_1_quant_unit_from_cpu_qdq || is_as_accurate_as_cpu_qdq)
              << "Inaccuracy detected for output '"
              << output_names[i]
              << "', element " << j
              << ".\nOutput quant params: scale=" << output_qparams[i].scale
              << ", zero_point=" << static_cast<int32_t>(output_qparams[i].zero_point)
              << ".\nExpected val: " << expected_val << "\n"
              << "QNN QDQ val: " << qnn_val << " (err " << qnn_err << ")\n"
              << "CPU QDQ val: " << cpu_val << " (err " << cpu_err << ")";
        }
      } else {
        VerifyOutput(output_names[i], cpu_f32_outputs[i].Get<Tensor>(), qnn_qdq_tensor, fp32_abs_err);
      }
    }
  }
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

template <>
inline NodeArg* MakeTestInput(ModelTestBuilder& builder, const TestInputDef<bool>& input_def) {
  NodeArg* input = nullptr;
  const auto& shape = input_def.GetShape();
  const bool is_initializer = input_def.IsInitializer();

  if (input_def.IsRawData()) {  // Raw data.
    const std::vector<bool>& raw_data = input_def.GetRawData();

    if (is_initializer) {
      input = builder.MakeInitializerBool(shape, raw_data);
    } else {
      input = builder.MakeInput<bool>(shape, raw_data);
    }
  } else {  // Random data
    if (is_initializer) {
      input = builder.MakeRandInitializerBool(shape);
    } else {
      input = builder.MakeInputBool(shape);
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