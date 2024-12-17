// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <cmath>
#include <string>
#include <type_traits>
#include <unordered_map>
#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/float16.h"
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

  inline std::pair<float, float> CalcRminRmax() const {
    constexpr float qmin = static_cast<float>(std::numeric_limits<QType>::min());
    constexpr float qmax = static_cast<float>(std::numeric_limits<QType>::max());
    const float qrange = (qmax - qmin);
    const float rrange = this->scale * qrange;
    const float rmin = -(static_cast<float>(this->zero_point) - qmin) * this->scale;
    const float rmax = rrange + rmin;

    return {rmin, rmax};
  }

  inline bool IsSymmetric() const {
    constexpr float qmin = static_cast<float>(std::numeric_limits<QType>::min());
    constexpr float qmax = static_cast<float>(std::numeric_limits<QType>::max());
    float init_zero_point = (qmin + qmax) / 2.0;
    const QType symm_zero_point = static_cast<QType>(RoundHalfToEven(
        std::max(qmin, std::min(qmax, init_zero_point))));

    return this->zero_point == symm_zero_point;
  }

  static QuantParams<QType> Compute(float rmin, float rmax, bool symmetric = false) {
    return Compute(
        rmin,
        rmax,
        std::numeric_limits<QType>::min(),
        std::numeric_limits<QType>::max(),
        symmetric);
  }

  static QuantParams<QType> Compute(float rmin, float rmax, QType qmin, QType qmax, bool symmetric = false) {
    // Ensure a minimum range of 0.0001 (required by QNN)
    rmax = std::max(rmax, rmin + 0.0001f);

    // Both QNN and ORT require the range to include 0.0f
    rmin = std::min(rmin, 0.0f);
    rmax = std::max(rmax, 0.0f);

    if (symmetric) {
      const float abs_max = std::max(std::abs(rmin), std::abs(rmax));
      rmax = abs_max;
      rmin = -abs_max;
    }

    const float qmin_flt = qmin;
    const float qmax_flt = qmax;
    const float scale = (rmax - rmin) / (qmax_flt - qmin_flt);
    float initial_zero_point = 0.0f;

    if (symmetric) {
      // Symmetric uses same formula for zero-point as asymmetric, but we can cancel out terms for
      // increased numerical accuracy.
      initial_zero_point = (qmin_flt + qmax_flt) / 2.0f;
    } else {
      initial_zero_point = qmin_flt - (rmin / scale);
    }

    const QType zero_point = static_cast<QType>(RoundHalfToEven(std::max(qmin_flt,
                                                                         std::min(qmax_flt, initial_zero_point))));

    return QuantParams<QType>{scale, zero_point};
  }
};

// Utitity that converts quantization parameters from one type to another (e.g., uint8 to uint16).
template <typename SrcQType, typename DstQType>
inline QuantParams<DstQType> ConvertQuantParams(QuantParams<SrcQType> src_qparams) {
  std::pair<float, float> src_rmin_rmax = src_qparams.CalcRminRmax();
  return QuantParams<DstQType>::Compute(src_rmin_rmax.first, src_rmin_rmax.second, src_qparams.IsSymmetric());
}

// Signature for function that builds a QDQ model.
// The parameter `output_qparams` contains quantization parameters that *can* be used for the QDQ model output.
// These output quantization parameters are computed by first running the float32 model and determining the
// range of output values. Note that the function is able to overwrite the output_qparams parameter if necessary
// (Example: MaxPool must have identical input and output quantization params).
template <typename QuantType>
using GetTestQDQModelFn = std::function<void(ModelTestBuilder& builder,
                                             std::vector<QuantParams<QuantType>>& output_qparams)>;

// Computes quantization parameters for an array of floating-point values.
template <typename QType = uint8_t>
inline QuantParams<QType> GetDataQuantParams(gsl::span<const float> data, bool symmetric = false) {
  // Get min/max of raw data.
  float min_val = std::numeric_limits<float>::max();
  float max_val = std::numeric_limits<float>::min();

  for (auto val : data) {
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
  }

  return QuantParams<QType>::Compute(min_val, max_val, symmetric);
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

  const TensorShape GetTensorShape() const {
    return TensorShape(shape_);
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

  std::vector<std::pair<T, T>> GetRangePerChannel(size_t axis) const {
    auto which_type = data_info_.index();
    const size_t num_ranges = static_cast<size_t>(shape_.at(axis));

    // Random. All axis dims get the same ranges (rand_min -> rand_max)
    if (which_type == 1) {
      RandomData rand_info = std::get<RandomData>(data_info_);
      return std::vector<std::pair<T, T>>(num_ranges, std::pair<T, T>(rand_info.min, rand_info.max));
    }

    // Raw data. Get min/max per axis dim val
    assert(which_type == 0);

    const std::vector<T>& raw_data = std::get<RawData>(data_info_).data;
    std::pair<T, T> init_range(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest());
    std::vector<std::pair<T, T>> per_axis_ranges(num_ranges, init_range);
    TensorShape shape(shape_);
    size_t num_blocks = shape.SizeToDimension(axis);
    size_t block_size = shape.SizeFromDimension(axis + 1);

    size_t i = 0;
    for (size_t n = 0; n < num_blocks; n++) {
      for (size_t r = 0; r < num_ranges; r++) {
        for (size_t j = 0; j < block_size; j++) {
          std::pair<T, T>& range = per_axis_ranges[r];
          range.first = std::min(range.first, raw_data[i]);
          range.second = std::max(range.second, raw_data[i]);
          i++;
        }
      }
    }
    assert(i == raw_data.size());

    return per_axis_ranges;
  }

 private:
  std::vector<int64_t> shape_;
  std::variant<RawData, RandomData> data_info_;
  bool is_initializer_{false};
  bool has_range_override_{false};
  std::pair<T, T> range_override_;
};

// Convert a float input definition to a float16 input definition.
TestInputDef<MLFloat16> ConvertToFP16InputDef(const TestInputDef<float>& input_def);

template <typename QType>
inline QuantParams<QType> GetTestInputQuantParams(const TestInputDef<float>& input_def, bool symmetric = false) {
  const std::pair<float, float> frange = input_def.GetRange();
  return QuantParams<QType>::Compute(frange.first, frange.second, symmetric);
}

template <typename QType>
static void GetTestInputQuantParamsPerChannel(const TestInputDef<float>& input_def, std::vector<float>& scales,
                                              std::vector<QType>& zero_points, size_t axis, bool symmetric = false) {
  const auto f32_ranges = input_def.GetRangePerChannel(axis);

  scales.reserve(f32_ranges.size());
  zero_points.reserve(f32_ranges.size());

  for (const auto& range : f32_ranges) {
    QuantParams<QType> params = QuantParams<QType>::Compute(range.first, range.second, symmetric);
    scales.push_back(params.scale);
    zero_points.push_back(params.zero_point);
  }
}

// Define functions to get the quantization parameters (i.e., scale/zp) for input data that will be quantized
// as int4 per-channel.
#define DEF_GET_INPUT_QPARAMS_PER_CHAN_INT4_FUNC(INT4x2_TYPE)                                                 \
  template <>                                                                                                 \
  inline void GetTestInputQuantParamsPerChannel<INT4x2_TYPE>(const TestInputDef<float>& input_def,            \
                                                             std::vector<float>& scales,                      \
                                                             std::vector<INT4x2_TYPE>& zero_points,           \
                                                             size_t axis, bool symmetric) {                   \
    using UnpackedType = typename INT4x2_TYPE::UnpackedType;                                                  \
    const auto f32_ranges = input_def.GetRangePerChannel(axis);                                               \
    const size_t num_ranges = f32_ranges.size();                                                              \
                                                                                                              \
    scales.resize(num_ranges);                                                                                \
    zero_points.resize(INT4x2_TYPE::CalcNumInt4Pairs(num_ranges));                                            \
                                                                                                              \
    for (size_t i = 0; i < num_ranges; i++) {                                                                 \
      const auto& range = f32_ranges[i];                                                                      \
      QuantParams<UnpackedType> params = QuantParams<UnpackedType>::Compute(range.first, range.second,        \
                                                                            INT4x2_TYPE::min_val,             \
                                                                            INT4x2_TYPE::max_val, symmetric); \
      scales[i] = params.scale;                                                                               \
                                                                                                              \
      size_t r = i >> 1;                                                                                      \
      size_t c = i & 0x1;                                                                                     \
      zero_points[r].SetElem(c, params.zero_point);                                                           \
    }                                                                                                         \
  }

DEF_GET_INPUT_QPARAMS_PER_CHAN_INT4_FUNC(Int4x2)
DEF_GET_INPUT_QPARAMS_PER_CHAN_INT4_FUNC(UInt4x2)

template <typename FloatType, typename QuantType>
static void QuantizeValues(gsl::span<const FloatType> input, gsl::span<QuantType> output, const TensorShape& shape,
                           gsl::span<const FloatType> scales, gsl::span<const QuantType> zero_points,
                           std::optional<int64_t> axis) {
  const size_t input_rank = shape.NumDimensions();
  const size_t num_elems = static_cast<size_t>(shape.Size());
  ORT_ENFORCE(input.size() == num_elems);
  ORT_ENFORCE(output.size() == num_elems);

  size_t block_count = 1;
  size_t broadcast_dim = 1;
  size_t block_size = num_elems;

  if (axis.has_value()) {
    size_t axis_no_neg = *axis < 0 ? static_cast<size_t>(*axis) + input_rank : static_cast<size_t>(*axis);
    block_count = shape.SizeToDimension(axis_no_neg);
    broadcast_dim = shape[axis_no_neg];
    block_size = shape.SizeFromDimension(axis_no_neg + 1);
  }

  ORT_ENFORCE(scales.size() == broadcast_dim);
  ORT_ENFORCE(zero_points.empty() || zero_points.size() == broadcast_dim);

  size_t i = 0;

  for (size_t n = 0; n < block_count; n++) {
    for (size_t bd = 0; bd < broadcast_dim; bd++) {
      QuantType zp = zero_points.empty() ? static_cast<QuantType>(0) : zero_points[bd];
      if constexpr (std::is_same_v<QuantType, int32_t>) {
        for (size_t e = 0; e < block_size; e++) {
          output[i + e] = static_cast<QuantType>(input[i + e] / scales[bd]) + zp;
        }
      } else {
        ParQuantizeLinearStd(&input[i], &output[i], block_size, scales[bd], zp, nullptr);
      }
      i += block_size;
    }
  }
}

// Define functions to quantize input data to 4-bits. Quantization can be done per-tensor or per-channel.
#define DEF_QUANTIZE_VALUES_INT4_FUNC(INT4x2_TYPE, QUANT_FUNC)                                               \
  template <>                                                                                                \
  inline void QuantizeValues<float, INT4x2_TYPE>(gsl::span<const float> input,                               \
                                                 gsl::span<INT4x2_TYPE> output,                              \
                                                 const TensorShape& shape,                                   \
                                                 gsl::span<const float> scales,                              \
                                                 gsl::span<const INT4x2_TYPE> zero_points,                   \
                                                 std::optional<int64_t> axis) {                              \
    using UnpackedType = typename INT4x2_TYPE::UnpackedType;                                                 \
    const size_t input_rank = shape.NumDimensions();                                                         \
    const size_t num_int4_elems = static_cast<size_t>(shape.Size());                                         \
    ORT_ENFORCE(input.size() == num_int4_elems);                                                             \
    ORT_ENFORCE(output.size() == INT4x2_TYPE::CalcNumInt4Pairs(num_int4_elems));                             \
                                                                                                             \
    size_t block_count = 1;                                                                                  \
    size_t broadcast_dim = 1;                                                                                \
    size_t block_size = num_int4_elems;                                                                      \
                                                                                                             \
    if (axis.has_value()) {                                                                                  \
      size_t axis_no_neg = *axis < 0 ? static_cast<size_t>(*axis) + input_rank : static_cast<size_t>(*axis); \
      block_count = shape.SizeToDimension(axis_no_neg);                                                      \
      broadcast_dim = shape[axis_no_neg];                                                                    \
      block_size = shape.SizeFromDimension(axis_no_neg + 1);                                                 \
    }                                                                                                        \
                                                                                                             \
    ORT_ENFORCE(scales.size() == broadcast_dim);                                                             \
    ORT_ENFORCE(zero_points.empty() || zero_points.size() == INT4x2_TYPE::CalcNumInt4Pairs(broadcast_dim));  \
                                                                                                             \
    size_t i = 0;                                                                                            \
                                                                                                             \
    for (size_t n = 0; n < block_count; n++) {                                                               \
      for (size_t bd = 0; bd < broadcast_dim; bd++) {                                                        \
        size_t bd_i = bd >> 1;  /* bd / 2 */                                                                 \
        size_t bd_j = bd & 0x1; /* bd % 2 */                                                                 \
        UnpackedType zp = !zero_points.empty() ? zero_points[bd_i].GetElem(bd_j) : 0;                        \
        QUANT_FUNC(&input[i], output.data(), i, i + block_size, scales[bd], INT4x2_TYPE(zp, 0), nullptr);    \
        i += block_size;                                                                                     \
      }                                                                                                      \
    }                                                                                                        \
    assert(i == (block_count * broadcast_dim * block_size));                                                 \
  }

DEF_QUANTIZE_VALUES_INT4_FUNC(Int4x2, ParQuantizeLinearStdS4)
DEF_QUANTIZE_VALUES_INT4_FUNC(UInt4x2, ParQuantizeLinearStdU4)

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
 * \param graph_checker Function called on the Graph.
 */
void InferenceModel(const std::string& model_data, const char* log_id,
                    const ProviderOptions& provider_options,
                    ExpectedEPNodeAssignment expected_ep_assignment, const NameMLValMap& feeds,
                    std::vector<OrtValue>& output_vals,
                    bool is_qnn_ep = false,
                    const std::unordered_map<std::string, std::string>& session_option_pairs = {},
                    std::function<void(const Graph&)>* graph_checker = nullptr);

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
 * \param tolerance The percent tolerance (as fraction) QNN EP results are allowed to differ from the QDQ model
 *                  on CPU EP. This tolerance is a percentage of the output range.
 * \param log_severity The logger's severity setting.
 * \param ep_graph_checker Function called on the Graph generated for the QNN EP's session. Used to check node
 *                         EP assignment.
 */
template <typename QuantType>
inline void TestQDQModelAccuracy(const GetTestModelFn& f32_model_fn, const GetTestQDQModelFn<QuantType>& qdq_model_fn,
                                 ProviderOptions qnn_options, int opset_version,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 QDQTolerance tolerance = QDQTolerance(),
                                 logging::Severity log_severity = logging::Severity::kERROR,
                                 const std::string& qnn_ctx_model_path = "",
                                 const std::unordered_map<std::string, std::string>& session_option_pairs = {},
                                 std::function<void(const Graph&)>* qnn_ep_graph_checker = nullptr) {
  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();

  // Uncomment to dump LOGGER() output to stdout.
  // logging_manager.RemoveSink(logging::SinkType::EtwSink);

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

  // Uncomment to save f32 model to disk for debugging.
  // ASSERT_STATUS_OK(onnxruntime::Model::Save(f32_model, ToPathString("cmp_accuracy.f32.onnx")));

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

  // Uncomment to save QDQ model to disk for debugging.
  // ASSERT_STATUS_OK(onnxruntime::Model::Save(qdq_model, ToPathString("cmp_accuracy.qdq.onnx")));

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
                   expected_ep_assignment, qdq_helper.feeds_, qnn_qdq_outputs, is_qnn_ep, session_option_pairs);
  } else {
    // Run QDQ model on QNN EP and collect outputs.
    // Only need to apply the extra session options to this QDQ model inference on QNN EP
    InferenceModel(qdq_model_data, "qdq_model_logger", qnn_options, expected_ep_assignment,
                   qdq_helper.feeds_, qnn_qdq_outputs, is_qnn_ep, session_option_pairs, qnn_ep_graph_checker);
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
 * Tests the accuracy of a FP16 model on QNN EP by runnning 3 inferences:
 *
 * 1. float32 model on CPU EP (baseline)
 * 2. FP16 model on CPU EP
 * 3. FP16 model on QNN EP
 *
 * This function checks that running the FP16 model on QNN EP (#3) is at least as accurate (+- small tolerance)
 * as running the FP16 model on CPU EP (#2). We primarily measure accuracy by comparing to the baseline (#1).
 *
 * \param f32_model_fn Function that builds the float model (baseline for comparison).
 * \param f16_model_fn Function that builds the FP16 model (run by CPU EP and QNN EP).
 * \param qnn_options QNN EP provider options.
 * \param opset_version The opset version.
 * \param expected_ep_assignment Describes "which nodes" should be assigned to the EP.
 * \param tolerance The percent tolerance (as fraction) QNN EP results are allowed to differ from the FP16 model
 *                  on CPU EP. This tolerance is a percentage of the output range.
 * \param log_severity The logger's severity setting.
 */
inline void TestFp16ModelAccuracy(const GetTestModelFn& f32_model_fn,
                                  const GetTestModelFn& f16_model_fn,
                                  ProviderOptions qnn_options,
                                  int opset_version,
                                  ExpectedEPNodeAssignment expected_ep_assignment,
                                  float tolerance = 0.004,
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
  std::vector<gsl::span<const float>> output_vals;
  std::vector<int32_t> output_types;
  output_vals.resize(num_outputs);
  output_types.resize(num_outputs);

  for (size_t i = 0; i < num_outputs; i++) {
    auto& tensor = cpu_f32_outputs[i].Get<Tensor>();
    int32_t elem_type = tensor.GetElementType();

    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      output_vals[i] = tensor.DataAsSpan<float>();
    }

    output_types[i] = elem_type;
  }

  // Create FP16 model and serialize it to a string.
  onnxruntime::Model f16_model("fp16_model", false, ModelMetaData(), PathString(),
                               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                               logging_manager.DefaultLogger());
  ModelTestBuilder f16_helper(f16_model.MainGraph());
  std::string f16_model_data;
  f16_model_fn(f16_helper);
  f16_helper.SetGraphOutputs();
  ASSERT_STATUS_OK(f16_model.MainGraph().Resolve());
  f16_model.ToProto().SerializeToString(&f16_model_data);

  bool is_qnn_ep = true;
  TryEnableQNNSaver(qnn_options);
  std::vector<OrtValue> qnn_f16_outputs;
  if (!qnn_ctx_model_path.empty()) {
    onnx::ModelProto model_proto;
    onnxruntime::Model qnn_ctx_model;
    // Load the QNN context cache model from path specified
    ASSERT_STATUS_OK(qnn_ctx_model.Load(ToPathString(qnn_ctx_model_path), model_proto));
    std::string qnn_ctx_model_data;
    model_proto.SerializeToString(&qnn_ctx_model_data);
    // Run QNN context cache model on QNN EP and collect outputs.
    InferenceModel(qnn_ctx_model_data, "qnn_ctx_model_logger", qnn_options,
                   expected_ep_assignment, f16_helper.feeds_, qnn_f16_outputs, is_qnn_ep, session_option_pairs);
  } else {
    // Run QDQ model on QNN EP and collect outputs.
    // Only need to apply the extra session options to this QDQ model inference on QNN EP
    InferenceModel(f16_model_data, "fp16_model_logger", qnn_options, expected_ep_assignment,
                   f16_helper.feeds_, qnn_f16_outputs, is_qnn_ep, session_option_pairs);
  }

  if (expected_ep_assignment != ExpectedEPNodeAssignment::None) {
    // Run QDQ model on CPU EP and collect outputs.
    std::vector<OrtValue> cpu_f16_outputs;
    InferenceModel(f16_model_data, "fp16_model_logger", {}, ExpectedEPNodeAssignment::All,
                   f16_helper.feeds_, cpu_f16_outputs);
    ASSERT_EQ(cpu_f16_outputs.size(), num_outputs);
    ASSERT_EQ(qnn_f16_outputs.size(), num_outputs);

    // limit the error message count in case test with large data failed
    size_t max_error_count = 10;
    size_t error_count = 0;

    // Compare accuracy of QDQ results with float model.
    // QNN EP must be at least as accurate as CPU EP when running the QDQ model.
    const std::string base_output_name = "output_";
    for (size_t i = 0; i < num_outputs; i++) {
      std::string debug_output_name = base_output_name + std::to_string(i);
      auto& cpu_f16_tensor = cpu_f16_outputs[i].Get<Tensor>();
      auto& qnn_f16_tensor = qnn_f16_outputs[i].Get<Tensor>();

      ASSERT_EQ(cpu_f16_tensor.GetElementType(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
      ASSERT_EQ(qnn_f16_tensor.GetElementType(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
      ASSERT_EQ(output_types[i], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      const size_t num_vals = output_vals[i].size();
      gsl::span<const float> cpu_f32_vals = output_vals[i];
      gsl::span<const MLFloat16> cpu_f16_vals = cpu_f16_tensor.DataAsSpan<MLFloat16>();
      gsl::span<const MLFloat16> qnn_f16_vals = qnn_f16_tensor.DataAsSpan<MLFloat16>();

      ASSERT_EQ(num_vals, cpu_f16_vals.size());
      ASSERT_EQ(num_vals, qnn_f16_vals.size());

      float max_f16_cpu_err = 0.0f;
      float max_f16_qnn_err = 0.0f;

      for (size_t j = 0; j < num_vals && error_count < max_error_count; j++) {
        const float expected_val = cpu_f32_vals[j];           // f32@CPU_EP val ("ground-truth")
        const float qnn_f16_val = qnn_f16_vals[j].ToFloat();  // f16@QNN_EP val
        const float cpu_f16_val = cpu_f16_vals[j].ToFloat();  // f16@CPU_EP val

        // Get errors of f16@CPU_EP and f16@QNN_EP against f32@CPU_EP.
        constexpr float epsilon = 1e-16f;
        const float cpu_relative_err = std::fabs(expected_val - cpu_f16_val) / (expected_val + epsilon);
        const float qnn_relative_err = std::fabs(expected_val - qnn_f16_val) / (expected_val + epsilon);

        // Also compare the FP16 values against each other.
        // This is equivalent to abs(f16@QNN_EP - f16@CPU_EP) / output_range
        const float f16_vals_err = std::fabs(qnn_relative_err - cpu_relative_err);

        // True if f16@QNN_EP is at least as accurate as f16@CPU_EP when compared to expected f32@CPU_EP value.
        const bool is_as_accurate_as_cpu_ep = qnn_relative_err <= qnn_relative_err;

        // True if the normalized difference between f16@QNN_EP and f16@CPU_EP is within tolerance.
        const bool f16_vals_diff_within_tolerance = f16_vals_err <= tolerance;

        const bool passed_test = is_as_accurate_as_cpu_ep || f16_vals_diff_within_tolerance;
        if (!passed_test) {
          ++error_count;
        }
        EXPECT_TRUE(passed_test)
            << "Inaccuracy detected for output '" << debug_output_name
            << "', element " << j << ", tolerance=" << (tolerance * 100) << "%"
            << ".\nExpected val (f32@CPU_EP): " << expected_val << "\n"
            << "f16@QNN_EP val: " << qnn_f16_val << " (err: " << qnn_relative_err << ")\n"
            << "f16@CPU_EP val: " << cpu_f16_val << " (err: " << cpu_relative_err << ")\n";

        max_f16_cpu_err = std::max(max_f16_cpu_err, cpu_relative_err);
        max_f16_qnn_err = std::max(max_f16_qnn_err, qnn_relative_err);
      }

      if (error_count > 0) {
        std::cerr << std::endl
                  << "[WARNING]: Output " << i
                  << " required larger tolerance to pass accuracy checks" << std::endl
                  << "Max relative error against f32@CPU_EP = " << max_f16_cpu_err << std::endl
                  << "Max relative error against f16@CPU_EP = " << max_f16_qnn_err << std::endl;
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
 * \param allocator Optional allocator to use to allocate the input ORT value.
 * \return A pointer to the new input.
 */
template <typename T>
inline NodeArg* MakeTestInput(ModelTestBuilder& builder, const TestInputDef<T>& input_def,
                              AllocatorPtr allocator = nullptr) {
  NodeArg* input = nullptr;
  const auto& shape = input_def.GetShape();
  const bool is_initializer = input_def.IsInitializer();

  if (input_def.IsRawData()) {  // Raw data.
    const std::vector<T>& raw_data = input_def.GetRawData();

    if (is_initializer) {
      input = builder.MakeInitializer<T>(shape, raw_data);
    } else {
      input = builder.MakeInput<T>(shape, raw_data, allocator);
    }
  } else {  // Random data
    const auto& rand_info = input_def.GetRandomDataInfo();

    if (is_initializer) {
      input = builder.MakeInitializer<T>(shape, rand_info.min, rand_info.max);
    } else {
      input = builder.MakeInput<T>(shape, rand_info.min, rand_info.max, allocator);
    }
  }

  return input;
}

template <>
inline NodeArg* MakeTestInput(ModelTestBuilder& builder, const TestInputDef<bool>& input_def,
                              AllocatorPtr allocator) {
  NodeArg* input = nullptr;
  const auto& shape = input_def.GetShape();
  const bool is_initializer = input_def.IsInitializer();

  if (input_def.IsRawData()) {  // Raw data.
    const std::vector<bool>& raw_data = input_def.GetRawData();

    if (is_initializer) {
      input = builder.MakeInitializerBool(shape, raw_data);
    } else {
      input = builder.MakeInput<bool>(shape, raw_data, allocator);
    }
  } else {  // Random data
    if (is_initializer) {
      input = builder.MakeRandInitializerBool(shape);
    } else {
      input = builder.MakeInputBool(shape, allocator);
    }
  }

  return input;
}

// ONNX spec does not allow quantizing float to int32. However, this function will create an int32
// input (divide by scale) and then return the output of DequantizeLinear. Note that bias_scale should
// be generally be equal to input_scale * weights_scale.
// See quantization tool: onnx_quantizer.py::quantize_bias_static()
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
 * \param input_allocator Optional allocator to use to allocate input ORT values.
 * \returns A model building function.
 */
template <typename InputType1, typename InputType2 = int64_t>
inline GetTestModelFn BuildOpTestCase(const std::string& op_type,
                                      const std::vector<TestInputDef<InputType1>>& input_defs_1,
                                      const std::vector<TestInputDef<InputType2>>& input_defs_2,
                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                      const std::string& op_domain = kOnnxDomain,
                                      AllocatorPtr input_allocator = nullptr) {
  return [op_type, input_defs_1, input_defs_2, attrs, op_domain, input_allocator](ModelTestBuilder& builder) {
    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(input_defs_1.size() + input_defs_2.size());

    for (const auto& input_def : input_defs_1) {
      NodeArg* input = MakeTestInput<InputType1>(builder, input_def, input_allocator);
      op_inputs.push_back(input);
    }

    for (const auto& input_def : input_defs_2) {
      NodeArg* input = MakeTestInput<InputType2>(builder, input_def, input_allocator);
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
 * \param use_contrib_qdq Whether to use Q/DQ ops from the MS domain instead of the ONNX domain.
 * \param input_allocator Optional allocator to use to allocate input ORT values.
 * \returns A model building function.
 */
template <typename QuantType, typename OtherInputType = int64_t>
inline GetTestQDQModelFn<QuantType> BuildQDQOpTestCase(
    const std::string& op_type,
    const std::vector<TestInputDef<float>>& quant_input_defs,
    const std::vector<TestInputDef<OtherInputType>>& non_quant_input_defs,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    const std::string& op_domain = kOnnxDomain,
    bool use_contrib_qdq = false,
    AllocatorPtr input_allocator = nullptr) {
  return [op_type, quant_input_defs, non_quant_input_defs, attrs, op_domain,
          use_contrib_qdq, input_allocator](
             ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(quant_input_defs.size() + non_quant_input_defs.size());

    // Create QDQ inputs
    for (const auto& input_def : quant_input_defs) {
      NodeArg* input = MakeTestInput<float>(builder, input_def, input_allocator);
      QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
      NodeArg* input_after_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale,
                                                           input_qparams.zero_point, use_contrib_qdq);
      op_inputs.push_back(input_after_qdq);
    }

    // Create non-QDQ inputs
    for (const auto& input_def : non_quant_input_defs) {
      NodeArg* input = MakeTestInput<OtherInputType>(builder, input_def, input_allocator);
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
 * \param verify_outputs True to verify that the outputs match (within tolerance).
 * \param ep_graph_checker Function called on the Graph generated for the EP's session. Used to check node
 *                         EP assignment.
 */
void RunQnnModelTest(const GetTestModelFn& build_test_case, ProviderOptions provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                     float fp32_abs_err = 1e-5f,
                     logging::Severity log_severity = logging::Severity::kERROR,
                     bool verify_outputs = true,
                     std::function<void(const Graph&)>* ep_graph_checker = nullptr);

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
