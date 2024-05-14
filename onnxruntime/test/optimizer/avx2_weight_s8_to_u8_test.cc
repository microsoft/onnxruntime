// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/avx2_weight_s8_to_u8.h"

#include <functional>
#include <string>
#include <vector>

#include "core/common/span_utils.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/common/quantization_test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

#include "core/session/onnxruntime_session_options_config_keys.h"

#ifndef DISABLE_CONTRIB_OPS

namespace onnxruntime {
namespace test {

static Status RunModel(ModelTestBuilder& helper, const std::string& model_data,
                       std::vector<OrtValue>& outputs, bool enable_s8u8_convertor = true) {
  SessionOptions session_options;
  if (enable_s8u8_convertor) {
    ORT_RETURN_IF_ERROR(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsAvx2PrecisionMode, "1"));
    session_options.graph_optimization_level = TransformerLevel::Level2;
  }

#if 0  // enable to dump model for debugging
    session_options.optimized_model_filepath =
        ToPathString("model" + std::to_string(static_cast<int>(level)) + ".onnx");
#endif
  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ORT_RETURN_IF_ERROR(session.Load(model_data.data(), static_cast<int>(model_data.size())));

  ORT_RETURN_IF_ERROR(session.Initialize());

  RunOptions run_options;
  ORT_RETURN_IF_ERROR(session.Run(run_options,
                                  helper.feeds_,
                                  helper.output_names_,
                                  &outputs));
  return Status::OK();
}

template <typename WeightType>
void BuildMatMulIntegerToFloatGraph(ModelTestBuilder& helper,
                                    const std::vector<int64_t> A_dims, const std::vector<int64_t> B_dims,
                                    const std::vector<uint8_t>& A_data, const std::vector<WeightType>& B_data,
                                    const std::vector<float>& A_scale, const std::vector<float>& B_scale,
                                    const std::vector<uint8_t>& A_zero_point,
                                    const std::vector<WeightType>& B_zero_point,
                                    const std::vector<float>& Bias) {
  auto* input_A = helper.MakeInput<uint8_t>(A_dims, A_data);
  auto* input_B = helper.MakeInitializer<WeightType>(B_dims, B_data);
  auto* input_a_scale = helper.MakeInitializer<float>({1}, A_scale);
  auto* input_b_scale = helper.MakeInitializer<float>({B_dims.back()}, B_scale);
  auto* input_a_zero_point = helper.MakeInitializer<uint8_t>({1}, A_zero_point);
  auto* input_b_zero_point = helper.MakeInitializer<WeightType>({B_dims.back()},
                                                                B_zero_point);
  auto* input_bias = helper.MakeInitializer<float>({B_dims.back()}, Bias);

  auto* output_arg = helper.MakeOutput();

  helper.AddNode("MatMulIntegerToFloat",
                 {input_A, input_B, input_a_scale, input_b_scale,
                  input_a_zero_point, input_b_zero_point, input_bias},
                 {output_arg}, kMSDomain);
  helper.SetGraphOutputs();
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6262)
#endif
TEST(CPU_U8S8_Precision_Tests, MatMulIntegerToFloat) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  domain_to_version[kMSDomain] = 1;

  // create rand inputs
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  RandomValueGenerator random{};

  std::vector<uint8_t> A_data = random.Uniform<uint8_t>(A_dims, 240, 255);
  std::vector<uint8_t> B_data = random.Uniform<uint8_t>(B_dims, 240, 255);

  std::vector<float> A_scale = random.Uniform<float>(
      AsSpan<int64_t>({1}), -0.1f, 0.1f);
  std::vector<uint8_t> A_zero_point{245};

  std::vector<float> B_scale = random.Uniform<float>(AsSpan({B_dims.back()}), -0.1f, 0.1f);

  std::vector<uint8_t> B_zero_point = random.Uniform<uint8_t>(AsSpan({B_dims.back()}), 240, 250);

  std::vector<float> Bias = random.Uniform<float>(AsSpan({B_dims.back()}), -0.1f, 0.1f);

  std::vector<OrtValue> baseline_fetches;

  {
    Model model("AVX2S8U8TransformerBase", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);

    BuildMatMulIntegerToFloatGraph<uint8_t>(helper, A_dims, B_dims, A_data, B_data,
                                            A_scale, B_scale, A_zero_point,
                                            B_zero_point, Bias);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(helper, model_data, baseline_fetches, false));
  }

  std::vector<int8_t> s8_B_data;
  std::transform(B_data.begin(), B_data.end(),
                 std::back_inserter(s8_B_data), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });
  std::vector<int8_t> s8_b_zero_point;
  std::transform(B_zero_point.begin(), B_zero_point.end(),
                 std::back_inserter(s8_b_zero_point), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });

  std::vector<OrtValue> outputs;
  {
    Model model("AVX2S8U8TransformerTests", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    std::unique_ptr<ModelTestBuilder> helper = std::make_unique<ModelTestBuilder>(graph);

    BuildMatMulIntegerToFloatGraph<int8_t>(*helper, A_dims, B_dims, A_data,
                                           s8_B_data, A_scale, B_scale,
                                           A_zero_point, s8_b_zero_point, Bias);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(*helper, model_data, outputs));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(outputs[i],
                        baseline_fetches[i],
                        0.0,
                        0.0,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <typename WeightType>
void BuildDynamicQuantizeMatMulGraph(
    ModelTestBuilder& helper,
    const std::vector<int64_t> A_dims, const std::vector<int64_t> B_dims,
    const std::vector<float>& A_data, const std::vector<WeightType>& B_data,
    const std::vector<float>& B_scale, const std::vector<WeightType>& B_zero_point,
    const std::vector<float>& Bias) {
  auto* input_A = helper.MakeInput<float>(A_dims, A_data);
  auto* input_B = helper.MakeInitializer<WeightType>(B_dims, B_data);
  auto* input_b_scale = helper.MakeInitializer<float>({B_dims.back()}, B_scale);
  auto* input_b_zero_point = helper.MakeInitializer<WeightType>(
      {B_dims.back()}, B_zero_point);
  auto* input_bias = helper.MakeInitializer<float>({B_dims.back()}, Bias);

  auto* output_arg = helper.MakeOutput();

  helper.AddNode("DynamicQuantizeMatMul",
                 {input_A, input_B, input_b_scale, input_b_zero_point, input_bias},
                 {output_arg}, kMSDomain);
  helper.SetGraphOutputs();
}

TEST(CPU_U8S8_Precision_Tests, DynamicQuantizeMatMul) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  domain_to_version[kMSDomain] = 1;

  // create rand inputs
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  RandomValueGenerator random{};

  std::vector<float> A_data = random.Uniform<float>(A_dims, -1.0f, 1.0f);

  std::vector<uint8_t> B_data = random.Uniform<uint8_t>(B_dims, 240, 255);

  std::vector<float> B_scale = random.Uniform<float>(AsSpan({B_dims.back()}), -0.1f, 0.1f);

  std::vector<uint8_t> B_zero_point = random.Uniform<uint8_t>(AsSpan({B_dims.back()}), 240, 250);

  std::vector<float> Bias = random.Uniform<float>(AsSpan({B_dims.back()}), -0.1f, 0.1f);

  std::vector<OrtValue> baseline_fetches;

  {
    Model model("AVX2S8U8TransformerBase", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    std::unique_ptr<ModelTestBuilder> helper = std::make_unique<ModelTestBuilder>(graph);

    BuildDynamicQuantizeMatMulGraph<uint8_t>(*helper, A_dims, B_dims, A_data, B_data,
                                             B_scale, B_zero_point, Bias);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(*helper, model_data, baseline_fetches, false));
  }

  std::vector<int8_t> s8_B_data;
  std::transform(B_data.begin(), B_data.end(),
                 std::back_inserter(s8_B_data), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });
  std::vector<int8_t> s8_b_zero_point;
  std::transform(B_zero_point.begin(), B_zero_point.end(),
                 std::back_inserter(s8_b_zero_point), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });

  std::vector<OrtValue> outputs;
  {
    Model model("AVX2S8U8TransformerTests", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);

    BuildDynamicQuantizeMatMulGraph<int8_t>(helper, A_dims, B_dims, A_data,
                                            s8_B_data, B_scale, s8_b_zero_point, Bias);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(helper, model_data, outputs));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(outputs[i],
                        baseline_fetches[i],
                        0.0,
                        0.0,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

TEST(CPU_U8S8_Precision_Tests, QGemm) {
  // Essentially copied QGemm unit test, where the original unit test
  // use reduced range int8_t to avoid possible overflow in SSE4.1, AVX2
  // or AVX512, we intentionally use big numbers to trigger it.

  constexpr int M = 4;
  constexpr int N = 8;
  constexpr int K = 68;

  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> random_A(235, 255);
  static std::uniform_int_distribution<int> random_B(107, 127);
  static std::uniform_real_distribution<float> n_apha(1.0f, 2.0f);
  static std::uniform_real_distribution<float> n_scale(0.003f, 0.004f);

  Eigen::MatrixXi matrix_a = Eigen::MatrixXi::Random(K, M)
                                 .unaryExpr([](int) { return random_A(e); });
  std::vector<uint8_t> matrix_a_data;
  matrix_a_data = ToVector<uint8_t>(matrix_a.data(), M * K);

  uint8_t a_zero_point = GetMiddle(matrix_a_data);
  Eigen::MatrixXi matrix_a_offset =
      matrix_a - a_zero_point * Eigen::MatrixXi::Ones(K, M);
  float a_scale = n_scale(e);

  Eigen::MatrixXi matrix_b = Eigen::MatrixXi::Random(N, K)
                                 .unaryExpr([](int) { return random_B(e); });
  std::vector<int8_t> matrix_b_data;
  matrix_b_data = ToVector<int8_t>(matrix_b.data(), N * K);

  int8_t b_zero_point = GetMiddle(matrix_b_data);
  std::vector<float> b_scale({n_scale(e)});
  std::vector<int8_t> b_zp_per_column({b_zero_point});
  Eigen::MatrixXi b_zp_matrix = b_zero_point * Eigen::MatrixXi::Ones(N, K);
  Eigen::MatrixXf b_scale_matrix = b_scale[0] * Eigen::MatrixXf::Ones(N, M);
  b_zp_per_column.resize(N);
  b_scale.resize(N);
  for (int i = 0; i < N; i++) {
    b_zp_per_column[i] = b_zero_point + i % 2 == 0 ? 1 : -1;
    b_zp_matrix.row(i).setConstant(b_zp_per_column[i]);
    b_scale[i] = n_scale(e);
    b_scale_matrix.row(i).setConstant(b_scale[i]);
  }

  float alpha = n_apha(e);

  Eigen::MatrixXi matrix_c = Eigen::MatrixXi::Random(N, M)
                                 .unaryExpr([](int) { return random_A(e); });

  Eigen::MatrixXi matrix_int32 = (matrix_b - b_zp_matrix) * matrix_a_offset;
  matrix_int32 = matrix_int32 + matrix_c;

  Eigen::MatrixXf matrix_output = alpha * a_scale * (b_scale_matrix.cwiseProduct((matrix_int32.eval().cast<float>())));

  OpTester test("QGemm", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("transA", 0);
  test.AddAttribute<int64_t>("transB", 0);
  test.AddAttribute<float>("alpha", alpha);
  test.AddInput<uint8_t>("A", std::vector<int64_t>({M, K}), std::move(matrix_a_data));
  test.AddInput<float>("a_scale", {}, {a_scale});
  test.AddInput<uint8_t>("a_zero_point", {}, {a_zero_point});
  test.AddInput<int8_t>("B", std::vector<int64_t>({K, N}), std::move(matrix_b_data), true);
  test.AddInput<float>("b_scale", {SafeInt<int64_t>(b_scale.size())}, b_scale);
  test.AddInput<int8_t>("b_zero_point", {SafeInt<int64_t>(b_zp_per_column.size())}, b_zp_per_column, true);
  test.AddInput<int32_t>("C", {M, N}, ToVector<int32_t>(matrix_c.data(), M * N));

  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOutput<float>("Y", {M, N}, std::vector<float>(matrix_output.data(), matrix_output.data() + M * N));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());

  SessionOptions so;
  so.use_per_session_threads = false;
  so.session_logid = "QGemmAvx2PrecTransformer";
  so.session_log_verbosity_level = 1;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsAvx2PrecisionMode, "1"));
  so.graph_optimization_level = TransformerLevel::Level2;
  test.Run(so, OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

TEST(CPU_U8S8_Precision_Tests, MatMulInteger) {
  // Essentially copied unit test, where the original unit test
  // use reduced range int8_t to avoid possible overflow in SSE4.1, AVX2
  // or AVX512, we intentionally use big numbers to trigger it.

  constexpr int M = 4;
  constexpr int N = 8;
  constexpr int K = 68;

  OpTester test("MatMulInteger", 10);
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(235, 254);
  static std::uniform_int_distribution<int> n_xint8(107, 126);

  Eigen::MatrixXi matrix_a = Eigen::MatrixXi::Random(K, M)
                                 .unaryExpr([](int) { return n_unsigned(e); });
  std::vector<uint8_t> matrix_a_data = ToVector<uint8_t>(matrix_a.data(), M * K);
  uint8_t a_zero_point = GetMiddle(matrix_a_data);
  Eigen::MatrixXi matrix_a_offset = matrix_a - a_zero_point * Eigen::MatrixXi::Ones(K, M);

  Eigen::MatrixXi matrix_b = Eigen::MatrixXi::Random(N, K)
                                 .unaryExpr([](int) { return n_xint8(e); });
  std::vector<int8_t> matrix_b_data = ToVector<int8_t>(matrix_b.data(), N * K);
  int8_t b_zero_point = 25;
  std::vector<int8_t> b_zp_per_column(N, b_zero_point);
  Eigen::MatrixXi b_zp_matrix = b_zero_point * Eigen::MatrixXi::Ones(N, K);
  for (int i = 0; i < N; i++) {
    b_zp_per_column[i] += i % 2 == 0 ? 1 : -1;
    b_zp_matrix.row(i).setConstant(b_zp_per_column[i]);
  }

  Eigen::MatrixXi matrix_c = ((matrix_b - b_zp_matrix) * matrix_a_offset).eval();

  test.AddInput<uint8_t>("T1", {M, K}, std::move(matrix_a_data));
  test.AddInput<int8_t>("T2", {K, N}, std::move(matrix_b_data), true);
  test.AddInput<uint8_t>("a_zero_point", {}, {a_zero_point});
  test.AddInput<int8_t>("b_zero_point", {N}, b_zp_per_column, true);

  test.AddOutput<int32_t>("T3", {M, N}, ToVector<int32_t>(matrix_c.data(), M * N));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());

  SessionOptions so;
  so.use_per_session_threads = false;
  so.session_logid = "QGemmAvx2PrecTransformer";
  so.session_log_verbosity_level = 1;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsAvx2PrecisionMode, "1"));
  so.graph_optimization_level = TransformerLevel::Level2;
  test.Run(so, OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

template <typename WeightType>
void BuildQLinearMatMulGraph(ModelTestBuilder& helper,
                             const std::vector<int64_t> A_dims, const std::vector<int64_t> B_dims,
                             const std::vector<uint8_t>& A_data, const std::vector<WeightType>& B_data,
                             float A_scale, float B_scale,
                             uint8_t A_zero_point, WeightType B_zero_point,
                             float Y_scale, uint8_t Y_zero_point) {
  auto* input_A = helper.MakeInput<uint8_t>(A_dims, A_data);
  auto* input_B = helper.MakeInitializer<WeightType>(B_dims, B_data);
  auto* input_a_scale = helper.MakeInitializer<float>({1}, {A_scale});
  auto* input_b_scale = helper.MakeInitializer<float>({1}, {B_scale});
  auto* input_a_zero_point = helper.MakeInitializer<uint8_t>({1}, {A_zero_point});
  auto* input_b_zero_point = helper.MakeInitializer<WeightType>(
      {1}, {B_zero_point});
  auto* input_y_scale = helper.MakeInitializer<float>({1}, {Y_scale});
  auto* input_y_zero_point = helper.MakeInitializer<uint8_t>({1}, {Y_zero_point});

  auto* output_arg = helper.MakeOutput();

  helper.AddNode("QLinearMatMul",
                 {input_A, input_a_scale, input_a_zero_point,
                  input_B, input_b_scale, input_b_zero_point,
                  input_y_scale, input_y_zero_point},
                 {output_arg}, kOnnxDomain);
  helper.SetGraphOutputs();
}

TEST(CPU_U8S8_Precision_Tests, QLinearMatMul) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  domain_to_version[kMSDomain] = 1;

  // create rand inputs
  std::vector<int64_t> A_dims{2, 2, 4};
  std::vector<int64_t> B_dims{2, 4, 3};
  std::vector<int64_t> Y_dims{2, 2, 3};
  RandomValueGenerator random{};

  std::vector<uint8_t> A_data = random.Uniform<uint8_t>(A_dims, 240, 255);
  float a_scale = 0.12f;
  uint8_t a_zero_point = 102;

  std::vector<uint8_t> B_data = random.Uniform<uint8_t>(B_dims, 240, 255);
  float b_scale = 0.08f;
  uint8_t b_zero_point = 96;

  float y_scale = 0.17f;
  uint8_t y_zero_point = 105;

  std::vector<OrtValue> baseline_fetches;
  {
    Model model("AVX2S8U8TransformerBase", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    std::unique_ptr<ModelTestBuilder> helper = std::make_unique<ModelTestBuilder>(graph);

    BuildQLinearMatMulGraph<uint8_t>(*helper, A_dims, B_dims, A_data, B_data,
                                     a_scale, b_scale, a_zero_point,
                                     b_zero_point, y_scale, y_zero_point);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(*helper, model_data, baseline_fetches, false));
  }

  std::vector<int8_t> s8_B_data;
  std::transform(B_data.begin(), B_data.end(),
                 std::back_inserter(s8_B_data), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });
  int8_t s8_b_zero_point = b_zero_point - 128;

  std::vector<OrtValue> outputs;
  {
    Model model("AVX2S8U8TransformerTests", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);

    BuildQLinearMatMulGraph<int8_t>(helper, A_dims, B_dims, A_data, s8_B_data,
                                    a_scale, b_scale, a_zero_point,
                                    s8_b_zero_point, y_scale, y_zero_point);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(helper, model_data, outputs));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(outputs[i],
                        baseline_fetches[i],
                        0.0,
                        0.0,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

template <typename FilterType>
void BuildQLinearConvGraph(
    ModelTestBuilder& helper,
    const std::vector<int64_t> x_dims, const std::vector<int64_t> w_dims,
    const std::vector<uint8_t>& x_data, const std::vector<FilterType>& w_data,
    float x_scale, float w_scale, float y_scale,
    uint8_t x_zero_point, FilterType w_zero_point, uint8_t y_zero_point) {
  auto* input_x = helper.MakeInput<uint8_t>(x_dims, x_data);
  auto* input_w = helper.MakeInitializer<FilterType>(w_dims, w_data);
  auto* input_x_scale = helper.MakeInitializer<float>({1}, {x_scale});
  auto* input_w_scale = helper.MakeInitializer<float>({1}, {w_scale});
  auto* input_x_zero_point = helper.MakeInitializer<uint8_t>({1}, {x_zero_point});
  auto* input_w_zero_point = helper.MakeInitializer<FilterType>(
      {1}, {w_zero_point});
  auto* input_y_scale = helper.MakeInitializer<float>({1}, {y_scale});
  auto* input_y_zero_point = helper.MakeInitializer<uint8_t>({1}, {y_zero_point});

  auto* output_arg = helper.MakeOutput();

  helper.AddNode("QLinearConv",
                 {input_x, input_x_scale, input_x_zero_point,
                  input_w, input_w_scale, input_w_zero_point,
                  input_y_scale, input_y_zero_point},
                 {output_arg}, kOnnxDomain);
  helper.SetGraphOutputs();
}

TEST(CPU_U8S8_Precision_Tests, QLinearConv) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  domain_to_version[kMSDomain] = 1;

  // create rand inputs
  std::vector<int64_t> x_dims{3, 24, 15, 11};
  std::vector<int64_t> w_dims{32, 24, 3, 3};
  RandomValueGenerator random{};

  std::vector<uint8_t> x_data = random.Uniform<uint8_t>(x_dims, 1, 255);
  float x_scale = 0.022f;
  uint8_t x_zero_point = 102;

  std::vector<uint8_t> w_data = random.Uniform<uint8_t>(w_dims, 1, 255);
  float w_scale = 0.009f;
  uint8_t w_zero_point = 96;

  float y_scale = 0.17f;
  uint8_t y_zero_point = 105;

  std::vector<OrtValue> baseline_fetches;
  {
    Model model("AVX2S8U8TransformerBase", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    std::unique_ptr<ModelTestBuilder> helper = std::make_unique<ModelTestBuilder>(graph);

    BuildQLinearConvGraph<uint8_t>(*helper, x_dims, w_dims, x_data, w_data,
                                   x_scale, w_scale, y_scale, x_zero_point,
                                   w_zero_point, y_zero_point);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(*helper, model_data, baseline_fetches, false));
  }

  std::vector<int8_t> s8_w_data;
  std::transform(w_data.begin(), w_data.end(),
                 std::back_inserter(s8_w_data), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });
  int8_t s8_w_zero_point = w_zero_point - 128;

  std::vector<OrtValue> outputs;
  {
    Model model("AVX2S8U8TransformerTests", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);

    BuildQLinearConvGraph<int8_t>(helper, x_dims, w_dims, x_data, s8_w_data,
                                  x_scale, w_scale, y_scale, x_zero_point,
                                  s8_w_zero_point, y_zero_point);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(helper, model_data, outputs));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(outputs[i],
                        baseline_fetches[i],
                        0.0,
                        0.0,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

template <typename QType>
void BuildDynamicQuantizeLSTMGraph(
    ModelTestBuilder& helper, int64_t seq_len, int64_t hidden_size,
    const std::vector<int64_t>& x_dims, const std::vector<float>& x_data,
    const std::vector<int64_t>& w_dims, const std::vector<QType>& w_data,
    float w_scale, QType w_zero_point,
    const std::vector<int64_t>& r_dims, const std::vector<QType>& r_data,
    float r_scale, QType r_zero_point,
    const std::vector<int64_t>& b_dims, const std::vector<float>& b_data,
    const std::vector<int64_t>& initial_h_dims, const std::vector<float>& initial_h_data,
    const std::vector<int64_t>& i_c_dims, const std::vector<float>& i_c_data,
    const std::vector<int64_t>& p_dims, const std::vector<float>& p_data) {
  auto* input_x = helper.MakeInput<float>(x_dims, x_data);
  auto* input_w = helper.MakeInitializer<QType>(w_dims, w_data);
  auto* input_r = helper.MakeInitializer<QType>(r_dims, r_data);
  auto* input_b = helper.MakeInput<float>(b_dims, b_data);

  std::vector<int> seqs(x_dims[1]);
  for (auto& v : seqs) {
    v = (int)seq_len;
  }
  auto* input_sq = helper.MakeInput<int>({x_dims[1]}, seqs);

  auto* input_init_h = helper.MakeInput<float>(initial_h_dims, initial_h_data);
  auto* input_init_c = helper.MakeInput<float>(i_c_dims, i_c_data);
  auto* input_p = helper.MakeInput<float>(p_dims, p_data);

  auto* input_w_scale = helper.MakeInitializer<float>({1}, {w_scale});
  auto* input_w_zero_point = helper.MakeInitializer<QType>(
      {1}, {w_zero_point});
  auto* input_r_scale = helper.MakeInitializer<float>({1}, {r_scale});
  auto* input_r_zero_point = helper.MakeInitializer<QType>(
      {1}, {r_zero_point});

  auto* output_arg = helper.MakeOutput();

  auto& node = helper.AddNode("DynamicQuantizeLSTM",
                              {input_x, input_w, input_r, input_b, input_sq,
                               input_init_h, input_init_c, input_p, input_w_scale,
                               input_w_zero_point, input_r_scale, input_r_zero_point},
                              {output_arg}, kMSDomain);

  node.AddAttribute("activations", std::vector<std::string>{"sigmoid", "tanh", "tanh"});

  node.AddAttribute("direction", "forward");
  node.AddAttribute("hidden_size", hidden_size);
  node.AddAttribute("input_forget", int64_t(0));

  helper.SetGraphOutputs();
}

TEST(CPU_U8S8_Precision_Tests, DynamicQuantizeLSTM) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  domain_to_version[kMSDomain] = 1;

  constexpr int64_t num_directions = 1;
  constexpr int64_t seq_len = 1;
  constexpr int64_t batch_size = 12;
  constexpr int64_t input_size = 3;
  constexpr int64_t hidden_size = 278;

  // create rand inputs
  RandomValueGenerator random{};

  std::vector<int64_t> x_dims = {seq_len, batch_size, input_size};
  std::vector<float> x_data = random.Gaussian<float>(x_dims, 0.0f, 0.25f);

  std::vector<int64_t> w_dims{num_directions, input_size, 4 * hidden_size};
  std::vector<uint8_t> w_data = random.Uniform<uint8_t>(w_dims, 1, 255);

  std::vector<int64_t> r_dims = {num_directions, hidden_size, 4 * hidden_size};
  std::vector<uint8_t> r_data = random.Uniform<uint8_t>(r_dims, 1, 255);

  std::vector<int64_t> b_dims = {num_directions, 8 * hidden_size};
  std::vector<float> b_data = random.Gaussian<float>(b_dims, 0.0f, 0.25f);

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data = random.Gaussian<float>(initial_h_dims, 0.0f, 0.25f);

  std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_c_data = random.Gaussian<float>(initial_c_dims, 0.0f, 0.25f);

  std::vector<int64_t> p_dims = {num_directions, 3 * hidden_size};
  std::vector<float> p_data = random.Gaussian<float>(p_dims, 0.0f, 0.25f);

  float w_scale = 0.022f;
  uint8_t w_zero_point = 102;

  float r_scale = 0.009f;
  uint8_t r_zero_point = 96;

  std::vector<OrtValue> baseline_fetches;
  {
    Model model("AVX2S8U8TransformerBase", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    std::unique_ptr<ModelTestBuilder> helper = std::make_unique<ModelTestBuilder>(graph);

    BuildDynamicQuantizeLSTMGraph<uint8_t>(
        *helper, seq_len, hidden_size, x_dims, x_data, w_dims, w_data,
        w_scale, w_zero_point,
        r_dims, r_data, r_scale, r_zero_point, b_dims, b_data,
        initial_h_dims, initial_h_data,
        initial_c_dims, initial_c_data, p_dims, p_data);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(*helper, model_data, baseline_fetches, false));
  }

  std::vector<int8_t> s8_w_data;
  std::transform(w_data.begin(), w_data.end(),
                 std::back_inserter(s8_w_data), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });
  int8_t s8_w_zero_point = w_zero_point - 128;

  std::vector<int8_t> s8_r_data;
  std::transform(r_data.begin(), r_data.end(),
                 std::back_inserter(s8_r_data), [](uint8_t v) -> int8_t {
                   return static_cast<int8_t>(v ^ 0x80);
                 });
  int8_t s8_r_zero_point = r_zero_point - 128;

  std::vector<OrtValue> outputs;
  {
    Model model("AVX2S8U8TransformerTests", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);

    BuildDynamicQuantizeLSTMGraph<int8_t>(
        helper, seq_len, hidden_size, x_dims, x_data, w_dims, s8_w_data,
        w_scale, s8_w_zero_point,
        r_dims, s8_r_data, r_scale, s8_r_zero_point, b_dims, b_data,
        initial_h_dims, initial_h_data,
        initial_c_dims, initial_c_data, p_dims, p_data);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

    ASSERT_STATUS_OK(RunModel(helper, model_data, outputs));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(outputs[i],
                        baseline_fetches[i],
                        0.0,
                        0.0,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

// input:      [batch_size, sequence_length, hidden_size]
// weights:    [hidden_size, 3 * hidden_size]
// bias:       [3 * hidden_size]
// mask_index: [batch_size]
// output:     [batch_size, sequence_length, hidden_size]
void RunQAttention(const std::vector<float>& input_data,
                   const std::vector<float>& weights_data,
                   const std::vector<float>& bias_data,
                   const std::vector<int32_t>& mask_index_data,
                   const std::vector<float>& output_data,
                   quantization::Params<uint8_t>& input_quant_params,
                   quantization::Params<int8_t>& weight_quant_params,
                   int batch_size,
                   int sequence_length,
                   int hidden_size,
                   int number_of_heads) {
  OpTester tester("QAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weights_dims = {hidden_size, static_cast<int64_t>(3 * hidden_size)};
  std::vector<int64_t> bias_dims = {static_cast<int64_t>(3 * hidden_size)};
  std::vector<int64_t> mask_index_dims = {batch_size};

  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  tester.AddInput<uint8_t>(
      "input", input_dims,
      QuantizeTestVector<uint8_t>(input_data, input_quant_params));
  tester.AddInput<int8_t>(
      "weight", weights_dims,
      QuantizeTestVector<int8_t>(weights_data, weight_quant_params), true);
  tester.AddInput<float>("bias", bias_dims, bias_data);
  tester.AddInput<float>("input_scale", {1}, {input_quant_params.scale});
  tester.AddInput<float>("weight_scale", {1}, {weight_quant_params.scale}, true);
  tester.AddOutput<float>("output", output_dims, output_data);

  tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);

  tester.AddInput<uint8_t>("input_zero_point", {1}, {input_quant_params.zero_point});
  tester.AddInput<int8_t>("weight_zero_point", {1}, {weight_quant_params.zero_point}, true);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());

  SessionOptions so;
  so.use_per_session_threads = false;
  so.session_logid = "QAttention_U8S8_Precision_Test";
  so.session_log_verbosity_level = 1;
  so.execution_mode = ORT_SEQUENTIAL;
  so.graph_optimization_level = TransformerLevel::Level2;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsAvx2PrecisionMode, "1"));
  tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "",
             {kTensorrtExecutionProvider}, nullptr, &execution_providers);
}

// AVX2/AVX512 CPU has overflow problem with U8S8 matrix multiplication
// This test is ensure when precision mode is turned on, the graph
// transformer convert U8S8 to U8U8 if necessary to produce correct
// result.
TEST(CPU_U8S8_Precision_Tests, QAttention) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;
  constexpr int hidden_size = 4;
  constexpr int number_of_heads = 2;

  std::vector<float> input_data = {
      1.39f, 1.37f, 1.42f, 1.47f,
      1.32f, 1.44f, 1.33f, 1.4f};

  std::vector<float> weight_data = {
      9.1f, -0.2f, 0.3f, 9.0f, 1.1f, 9.3f, 9.5f, 0.2f, 9.3f, -0.6f, 9.5f, 8.0f,
      9.5f, 9.1f, 0.4f, 9.6f, 9.0f, 9.0f, 0.4f, 8.8f, 7.9f, 7.1f, -1.3f, 9.7f,
      9.3f, 9.2f, 4.0f, 9.2f, 9.6f, 9.8f, 9.7f, 0.2f, 8.4f, 8.0f, 1.2f, 8.5f,
      9.2f, 9.1f, 0.4f, 9.6f, 9.4f, 9.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 9.9f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      48.5260010f, 20.6529999f, 15.7064486f, 51.1611328f, 48.5260010f, 20.6529999f, 15.6836147f, 51.1308899f};

  // Need to create resulting integer numbers large enough to trigger u8s8 overflow
  // in AVX2 and AVX512 CPUs. To make sure it's working, turn off precision mode
  // (below) and make sure this test fail.
  quantization::Params<uint8_t> input_quant_params(/*scale=*/0.01f, /*zero_point=*/108);
  quantization::Params<int8_t> weights_quant_params(/*scale=*/0.05f, /*zero_point=*/-73);

  RunQAttention(
      input_data, weight_data, bias_data, mask_index_data, output_data, input_quant_params, weights_quant_params,
      batch_size, sequence_length, hidden_size, number_of_heads);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // DISABLE_CONTRIB_OPS
