// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <string>
#include <cmath>
#include <unordered_map>
#include "core/framework/provider_options.h"
#include "core/util/qmath.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/util/include/test_utils.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/default_providers.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Signature for function that builds a float32 model.
using GetTestModelFn = std::function<void(ModelTestBuilder& builder)>;

// Class that stores quantization params (scale, zero point).
// Has a static function that computes quantization parameters from a floating-point range.
template <typename QType = uint8_t>
struct QuantParams {
  float scale;
  QType zero_point;

  static QuantParams<QType> Compute(float rmin, float rmax) {
    // Ensure a minimum range of 0.0001 (required by QNN)
    rmax = std::max(rmax, rmin + 0.0001f);

    // Both QNN and ORT require the range to include 0.0f
    rmin = std::min(rmin, 0.0f);
    rmax = std::max(rmax, 0.0f);

    constexpr float qmin = static_cast<float>(std::numeric_limits<QType>::min());
    constexpr float qmax = static_cast<float>(std::numeric_limits<QType>::max());

    const float scale = rmax == rmin ? 1.0f : (rmax - rmin) / (qmax - qmin);
    const float initial_zero_point = qmin - (rmin / scale);
    const QType zero_point = static_cast<QType>(RoundHalfToEven(std::max(qmin, std::min(qmax, initial_zero_point))));

    return QuantParams<QType>{scale, zero_point};
  }
};

// Signature for function that builds a QDQ model.
// The parameter `output_qparams` contains quantization parameters that *can* be used for the QDQ model output.
// These output quantization parameters are computed by first running the float32 model and determining the
// range of output values. Note that the function is able to overwrite the output_qparams parameter if necessary
// (Example: MaxPool must have identical input and output quantization params).
template <typename QuantType>
using GetTestQDQModelFn = std::function<void(ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams)>;

// Computes quantization parameters for an array of floating-point values.
template <typename QType = uint8_t>
inline QuantParams<QType> GetDataQuantParams(gsl::span<const float> data) {
  // Get min/max of raw data.
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::min();

  for (auto val : data) {
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
  }

  return QuantParams<QType>::Compute(min_val, max_val);
}

/**
 * Returns a float vector with data in the specified range. Uses linear interpolation to fill the elements in the array
 * and ensures that min_val, 0.0f, and max_val are all included.
 * TODO(adrianlizarraga): Should use this instead of random *float* test inputs for test repeatability/stability!
 *
 * \param min_val The minimum value.
 * \param max_val The maximum value.
 * \param num_elems The number of elements in the result. Should be at least 3 to include min, 0, and max.
 * \return A vector of floats with elements set to values in the specified range.
 */
std::vector<float> GetFloatDataInRange(float min_val, float max_val, size_t num_elems);

/**
 * Returns a float vector with sequential data.
 *
 * \param shape The tensor shape used to determine the number of values.
 * \param start The starting value.
 * \param step The step size.
 * \return A vector of sequential floats.
 */
std::vector<float> GetSequentialFloatData(const std::vector<int64_t>& shape, float start = 0.0f, float step = 1.0f);

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

  TestInputDef() = default;

  // Creates a random input definition. Specify its shape, whether it's an initializer, and
  // the min/max range.
  TestInputDef(std::vector<int64_t> shape, bool is_initializer, T rand_min, T rand_max)
      : shape_(std::move(shape)),
        data_info_(RandomData{rand_min, rand_max}),
        is_initializer_(is_initializer),
        has_range_override_(false),
        range_override_() {}

  // Create an input definition with explicit data. Specify its shape, whether it's an initializer,
  // and the raw data.
  TestInputDef(std::vector<int64_t> shape, bool is_initializer, std::vector<T> data)
      : shape_(std::move(shape)),
        data_info_(RawData{std::move(data)}),
        is_initializer_(is_initializer),
        has_range_override_(false),
        range_override_() {}

  TestInputDef(TestInputDef&& other) = default;
  TestInputDef(const TestInputDef& other) = default;

  TestInputDef& operator=(const TestInputDef& other) = default;
  TestInputDef& operator=(TestInputDef&& other) = default;

  // Overrides the range of input values reported by TestInputDef::GetRange().
  // This is useful when you want to quantize over a range that is larger or smaller
  // than the actual range of the data.
  //
  // Returns a reference to this object to allow chaining.
  TestInputDef& OverrideValueRange(T range_min, T range_max) {
    range_override_.first = range_min;
    range_override_.second = range_max;
    has_range_override_ = true;
    return *this;
  }

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

  // Get the range of values represented by this input, which is necessary for computing quantization parameters.
  // For raw data, we return [min, max] of the elements.
  // For random data, we return [rand_min, rand_max].
  // Optionally, the user can override this range by using OverrideValueRange().
  std::pair<T, T> GetRange() const {
    if (has_range_override_) {
      return range_override_;
    }

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
  bool is_initializer_{false};
  bool has_range_override_{false};
  std::pair<T, T> range_override_;
};

template <typename QType>
inline QuantParams<QType> GetTestInputQuantParams(const TestInputDef<float>& input_def) {
  const std::pair<float, float> frange = input_def.GetRange();
  return QuantParams<QType>::Compute(frange.first, frange.second);
}

/**
 * Inferences a given serialized model. Returns output values via an out-param.
 *
 * \param model_data The serialized ONNX model to inference.
 * \param log_id The logger ID.
 * \param provider_options provider options key value pair.
 * \param expected_ep_assignment Describes "which nodes" should be assigned to the EP.
 * \param feeds The input feeds.
 * \param output_vals Initialized to the inference results.
 * \param is_qnn_ep Ture: QNN EP is used. False: CPU EP is used (default).
 * \param session_option_pairs extra session options.
 */
void InferenceModel(const std::string& model_data, const char* log_id,
                    const ProviderOptions& provider_options,
                    ExpectedEPNodeAssignment expected_ep_assignment, const NameMLValMap& feeds,
                    std::vector<OrtValue>& output_vals,
                    bool is_qnn_ep = false,
                    const std::unordered_map<std::string, std::string>& session_option_pairs = {});

/**
 * If the ORT_UNIT_TEST_ENABLE_QNN_SAVER environment variable is enabled (set to 1), this function modifies
 * the QNN EP provider options to enable the QNN Saver backend, which dumps QNN API calls (and weights) to disk.
 *
 * - saver_output/saver_output.c: C file containing all QNN API calls.
 * - saver_output/params.bin: binary file containing all input/output/parameter tensor data provided during tensor
 *                            creation, op config validation, and graph execution.
 *
 * Enabling the QNN Saver backend has 2 note-worthy effects:
 * 1. All QNN API calls will succeed.
 * 2. Inference output returns dummy data.
 *
 * Because output files from QNN Saver are always overwritten, it is recommended to run individual unit tests via the
 * --gtest_filter command-line option. Ex: --gtest_filter=QnnHTPBackendTests.Resize_DownSample_Linear_AlignCorners
 *
 * \param qnn_options QNN EP provider options that may be modified to enable QNN Saver.
 */
void TryEnableQNNSaver(ProviderOptions& qnn_options);

struct QDQTolerance {
  // When comparing output activations between QNN EP and CPU EP (both running the QDQ model),
  // this value defines the maximum tolerable difference as a percentage of the output range.
  // Ex: (qdq@QNN_EP - qdq@CPU_EP) / (rmax_output - rmin_output) <= DEFAULT_QDQ_TOLERANCE.
  static constexpr float DEFAULT_QDQ_TOLERANCE = 0.004f;  // 0.4% is equivalent to 1 int8 quantization unit
                                                          // or 262 int16 quantization units.

  QDQTolerance() : value(DEFAULT_QDQ_TOLERANCE) {}
  explicit QDQTolerance(float tolerance) : value(tolerance) {}

  float value;
};

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
 * \param tolerance The percent tolerance (as fraction) QNN EP results are allowed to differ from the QDQ model on CPU EP.
 *                  This tolerance is a percentage of the output range.
 * \param log_severity The logger's severity setting.
 */
template <typename QuantType>
inline void TestQDQModelAccuracy(const GetTestModelFn& f32_model_fn, const GetTestQDQModelFn<QuantType>& qdq_model_fn,
                                 ProviderOptions qnn_options, int opset_version,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 QDQTolerance tolerance = QDQTolerance(),
                                 logging::Severity log_severity = logging::Severity::kERROR,
                                 const std::string& qnn_ctx_model_path = "",
                                 const std::unordered_map<std::string, std::string>& session_option_pairs = {}) {
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
  InferenceModel(f32_model_data, "f32_model_logger", {}, ExpectedEPNodeAssignment::All,
                 f32_helper.feeds_, cpu_f32_outputs);
  ASSERT_FALSE(cpu_f32_outputs.empty());

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
      output_qparams[i] = GetDataQuantParams<QuantType>(output_vals[i]);
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

  bool is_qnn_ep = true;
  TryEnableQNNSaver(qnn_options);
  std::vector<OrtValue> qnn_qdq_outputs;
  if (!qnn_ctx_model_path.empty()) {
    onnx::ModelProto model_proto;
    onnxruntime::Model qnn_ctx_model;
    // Load the QNN context cache model from path specified
    ASSERT_STATUS_OK(qnn_ctx_model.Load(ToPathString(qnn_ctx_model_path), model_proto));
    std::string qnn_ctx_model_data;
    model_proto.SerializeToString(&qnn_ctx_model_data);
    // Run QNN context cache model on QNN EP and collect outputs.
    InferenceModel(qnn_ctx_model_data, "qnn_ctx_model_logger", qnn_options,
                   expected_ep_assignment, qdq_helper.feeds_, qnn_qdq_outputs, is_qnn_ep);
  } else {
    // Run QDQ model on QNN EP and collect outputs.
    // Only need to apply the extra session options to this QDQ model inference on QNN EP
    InferenceModel(qdq_model_data, "qdq_model_logger", qnn_options, expected_ep_assignment,
                   qdq_helper.feeds_, qnn_qdq_outputs, is_qnn_ep, session_option_pairs);
  }

  if (expected_ep_assignment != ExpectedEPNodeAssignment::None) {
    // Run QDQ model on CPU EP and collect outputs.
    std::vector<OrtValue> cpu_qdq_outputs;
    InferenceModel(qdq_model_data, "qdq_model_logger", {}, ExpectedEPNodeAssignment::All,
                   qdq_helper.feeds_, cpu_qdq_outputs);
    ASSERT_EQ(cpu_qdq_outputs.size(), num_outputs);
    ASSERT_EQ(qnn_qdq_outputs.size(), num_outputs);

    // limit the error message count in case test with large data failed
    size_t max_error_count = 10;
    size_t error_count = 0;

    // Compare accuracy of QDQ results with float model.
    // QNN EP must be at least as accurate as CPU EP when running the QDQ model.
    const std::string base_output_name = "output_";
    for (size_t i = 0; i < num_outputs; i++) {
      std::string debug_output_name = base_output_name + std::to_string(i);
      auto& cpu_qdq_tensor = cpu_qdq_outputs[i].Get<Tensor>();
      auto& qnn_qdq_tensor = qnn_qdq_outputs[i].Get<Tensor>();

      ASSERT_EQ(cpu_qdq_tensor.GetElementType(), output_types[i]);
      ASSERT_EQ(qnn_qdq_tensor.GetElementType(), output_types[i]);

      if (output_types[i] == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        const size_t num_vals = output_vals[i].size();
        gsl::span<const float> cpu_f32_vals = output_vals[i];
        gsl::span<const float> cpu_qdq_vals = cpu_qdq_tensor.DataAsSpan<float>();
        gsl::span<const float> qnn_qdq_vals = qnn_qdq_tensor.DataAsSpan<float>();
        constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
        constexpr QuantType qmax = std::numeric_limits<QuantType>::max();
        const float output_range = output_qparams[i].scale * static_cast<float>(qmax - qmin);

        ASSERT_EQ(num_vals, cpu_qdq_vals.size());
        ASSERT_EQ(num_vals, qnn_qdq_vals.size());

        float max_f32_err = 0.0f;
        float max_qdq_err = 0.0f;
        bool print_accuracy_warning = false;

        for (size_t j = 0; j < num_vals && error_count < max_error_count; j++) {
          const float expected_val = cpu_f32_vals[j];  // f32@CPU_EP val ("ground-truth")
          const float qnn_qdq_val = qnn_qdq_vals[j];   // qdq@QNN_EP val
          const float cpu_qdq_val = cpu_qdq_vals[j];   // qdq@CPU_EP val

          // Get errors of qdq@CPU_EP and qdq@QNN_EP against f32@CPU_EP.
          const float cpu_err = std::fabs(expected_val - cpu_qdq_val);
          const float cpu_err_norm = cpu_err / output_range;
          const float qnn_err = std::fabs(expected_val - qnn_qdq_val);
          const float qnn_err_norm = qnn_err / output_range;

          // Also compare the QDQ values against each other.
          // This is equivalent to abs(qdq@QNN_EP - qdq@CPU_EP) / output_range
          const float qdq_vals_err_norm = std::fabs(qnn_err_norm - cpu_err_norm);

          // True if qdq@QNN_EP is at least as accurate as qdq@CPU_EP when compared to expected f32@CPU_EP value.
          const bool is_as_accurate_as_cpu_ep = qnn_err_norm <= cpu_err_norm;

          // True if the normalized difference between qdq@QNN_EP and qdq@CPU_EP is within tolerance.
          const bool qdq_vals_diff_within_tolerance = qdq_vals_err_norm <= tolerance.value;

          const bool passed_test = is_as_accurate_as_cpu_ep || qdq_vals_diff_within_tolerance;
          if (!passed_test) {
            ++error_count;
          }
          EXPECT_TRUE(passed_test)
              << "Inaccuracy detected for output '" << debug_output_name
              << "', element " << j
              << "\noutput_range=" << output_range << ", tolerance=" << (tolerance.value * 100) << "%"
              << ".\nExpected val (f32@CPU_EP): " << expected_val << "\n"
              << "qdq@QNN_EP val: " << qnn_qdq_val << " (err: " << qnn_err << ", err/output_range: "
              << qnn_err_norm * 100.0f << "%)\n"
              << "qdq@CPU_EP val: " << cpu_qdq_val << " (err: " << cpu_err << ", err/output_range: "
              << cpu_err_norm * 100.0f << "%)\n"
              << "abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = " << qdq_vals_err_norm * 100.0f << "%";

          max_f32_err = std::max(max_f32_err, qnn_err_norm);
          max_qdq_err = std::max(max_qdq_err, qdq_vals_err_norm);
          if (passed_test && !is_as_accurate_as_cpu_ep && (qdq_vals_err_norm > QDQTolerance::DEFAULT_QDQ_TOLERANCE)) {
            print_accuracy_warning = true;
          }
        }

        if (print_accuracy_warning) {
          std::cerr << std::endl
                    << "[WARNING]: Output " << i
                    << " required larger tolerance to pass accuracy checks" << std::endl
                    << "Max normalized error against f32@CPU_EP = " << max_f32_err * 100.0f << "%" << std::endl
                    << "Max normalized error against qdq@CPU_EP = " << max_qdq_err * 100.0f << "%" << std::endl
                    << "Default tolerance = " << QDQTolerance::DEFAULT_QDQ_TOLERANCE * 100.0f << "%" << std::endl
                    << "Tolerance used = " << tolerance.value * 100.0f << "%" << std::endl;
        }
      } else {
        VerifyOutput(debug_output_name, cpu_f32_outputs[i].Get<Tensor>(), qnn_qdq_tensor, 1e-4f);
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

// ONNX spec does not allow quantizing float to int32. However, this function will create an int32 input (divide by scale)
// and then return the output of DequantizeLinear. Note that bias_scale should be generally be equal
// to input_scale * weights_scale. See quantization tool: onnx_quantizer.py::quantize_bias_static()
//
// i.e., initial bias => manual quantization (int32) => DQ => final float bias
NodeArg* MakeTestQDQBiasInput(ModelTestBuilder& builder, const TestInputDef<float>& bias_def, float bias_scale,
                              bool use_contrib_qdq = false);

/**
 * Returns a function that builds a model with a single operator with N inputs type InputType1 and M inputs
 * of type InputType2.
 *
 * \param op_type The operator to instantiate.
 * \param input_defs_1 List of input definitions of type InputType1.
 * \param input_defs_2 List of input definitions of type InputType2.
 * \param attrs List of operator attributes.
 * \param op_domain The operator's domain. Defaults to the ONNX domain (i.e., "").
 * \returns A model building function.
 */
template <typename InputType1, typename InputType2 = int64_t>
inline GetTestModelFn BuildOpTestCase(const std::string& op_type,
                                      const std::vector<TestInputDef<InputType1>>& input_defs_1,
                                      const std::vector<TestInputDef<InputType2>>& input_defs_2,
                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                      const std::string& op_domain = kOnnxDomain) {
  return [op_type, input_defs_1, input_defs_2, attrs, op_domain](ModelTestBuilder& builder) {
    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(input_defs_1.size() + input_defs_2.size());

    for (const auto& input_def : input_defs_1) {
      NodeArg* input = MakeTestInput<InputType1>(builder, input_def);
      op_inputs.push_back(input);
    }

    for (const auto& input_def : input_defs_2) {
      NodeArg* input = MakeTestInput<InputType2>(builder, input_def);
      op_inputs.push_back(input);
    }

    auto* output = builder.MakeOutput();
    Node& onnx_node = builder.AddNode(op_type, op_inputs, {output}, op_domain);

    for (const auto& attr : attrs) {
      onnx_node.AddAttributeProto(attr);
    }
  };
}

/**
 * Returns a function that builds a model with a single QDQ operator with N float (quantizeable) inputs
 * and M inputs of a potentially different type.
 *
 * \param op_type The operator to instantiate.
 * \param input_defs List of input definitions.
 * \param attrs List of operator attributes.
 * \param op_domain The operator's domain. Defaults to the ONNX domain (i.e., "").
 * \returns A model building function.
 */
template <typename QuantType, typename OtherInputType = int64_t>
inline GetTestQDQModelFn<QuantType> BuildQDQOpTestCase(const std::string& op_type,
                                                       const std::vector<TestInputDef<float>>& quant_input_defs,
                                                       const std::vector<TestInputDef<OtherInputType>>& non_quant_input_defs,
                                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                       const std::string& op_domain = kOnnxDomain,
                                                       bool use_contrib_qdq = false) {
  return [op_type, quant_input_defs, non_quant_input_defs, attrs, op_domain,
          use_contrib_qdq](ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(quant_input_defs.size() + non_quant_input_defs.size());

    // Create QDQ inputs
    for (const auto& input_def : quant_input_defs) {
      NodeArg* input = MakeTestInput<float>(builder, input_def);
      QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      NodeArg* input_after_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale,
                                                           input_qparams.zero_point, use_contrib_qdq);
      op_inputs.push_back(input_after_qdq);
    }

    // Create non-QDQ inputs
    for (const auto& input_def : non_quant_input_defs) {
      NodeArg* input = MakeTestInput<OtherInputType>(builder, input_def);
      op_inputs.push_back(input);
    }

    // Op -> op_output
    auto* op_output = builder.MakeIntermediate();
    Node& onnx_node = builder.AddNode(op_type, op_inputs, {op_output}, op_domain);

    for (const auto& attr : attrs) {
      onnx_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, op_output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point, use_contrib_qdq);
  };
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
void RunQnnModelTest(const GetTestModelFn& build_test_case, ProviderOptions provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                     float fp32_abs_err = 1e-5f,
                     logging::Severity log_severity = logging::Severity::kERROR,
                     bool verify_outputs = true);

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
