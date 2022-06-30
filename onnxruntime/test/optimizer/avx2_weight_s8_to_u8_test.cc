#include "core/optimizer/avx2_weight_s8_to_u8.h"

#include <functional>
#include <string>
#include <vector>

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
#include "test/util/include/inference_session_wrapper.h"

#include "core/session/onnxruntime_session_options_config_keys.h"


namespace onnxruntime {
namespace test {

static void RunModel(ModelTestBuilder& helper, const std::string& model_data,
    std::vector<OrtValue>& outputs, bool enable_s8u8_convertor = true) {

  SessionOptions session_options;
  if (enable_s8u8_convertor) {
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsAvx2PrecisionMode, "1"));
    session_options.graph_optimization_level = TransformerLevel::Level2;
  }

#if 0  // enable to dump model for debugging
    session_options.optimized_model_filepath =
        ToPathString("model" + std::to_string(static_cast<int>(level)) + ".onnx");
#endif
  InferenceSessionWrapper session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));

  ASSERT_STATUS_OK(session.Initialize());

  RunOptions run_options;
  ASSERT_STATUS_OK(session.Run(run_options,
                               helper.feeds_,
                               helper.output_names_,
                               &outputs));
}

template <typename WeightType>
void BuildMatMulIntegerToFloatGraph(ModelTestBuilder& helper, 
    const std::vector<int64_t> A_dims, const std::vector<int64_t> B_dims,
    const std::vector<uint8_t>& A_data, const std::vector<WeightType>& B_data,
    const std::vector<float>& A_scale, const std::vector<float>& B_scale,
    const std::vector<uint8_t>& A_zero_point,
    const std::vector<WeightType>& B_zero_point,
    const std::vector<float>& Bias
) {
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

TEST(AVX2S8U8TransformerTests, MatMulIntegerToFloat) {
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
      std::array<int64_t, 1>{1}, -0.1f, 0.1f);
  std::vector<uint8_t> A_zero_point{245};

  std::vector<float> B_scale = random.Uniform<float>({B_dims.back()}, -0.1f, 0.1f);

  std::vector<uint8_t> B_zero_point = random.Uniform<uint8_t>(B_dims.back(), 240, 250);

  std::vector<float> Bias = random.Uniform<float>({B_dims.back()}, -0.1f, 0.1f);

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
    model.ToProto().SerializeToString(&model_data);

    RunModel(helper, model_data, baseline_fetches, false);
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

    BuildMatMulIntegerToFloatGraph<int8_t>(helper, A_dims, B_dims, A_data,
                                           s8_B_data, A_scale, B_scale,
                                           A_zero_point, s8_b_zero_point, Bias);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    model.ToProto().SerializeToString(&model_data);

    RunModel(helper, model_data, outputs);
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


TEST(AVX2S8U8TransformerTests, DynamicQuantizeMatMul) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 12;
  domain_to_version[kMSDomain] = 1;

  // create rand inputs
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  RandomValueGenerator random{};


  std::vector<float> A_data = random.Uniform<float>(A_dims, -1.0f, 1.0f);

  std::vector<uint8_t> B_data = random.Uniform<uint8_t>(B_dims, 240, 255);

  std::vector<float> B_scale = random.Uniform<float>({B_dims.back()}, -0.1f, 0.1f);

  std::vector<uint8_t> B_zero_point = random.Uniform<uint8_t>({B_dims.back()}, 240, 250);

  std::vector<float> Bias = random.Uniform<float>({B_dims.back()}, -0.1f, 0.1f);

  std::vector<OrtValue> baseline_fetches;

  {
    Model model("AVX2S8U8TransformerBase", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);

    BuildDynamicQuantizeMatMulGraph<uint8_t>(helper, A_dims, B_dims, A_data, B_data,
                                            B_scale, B_zero_point, Bias);

    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    // Serialize the model to a string.
    std::string model_data;
    model.ToProto().SerializeToString(&model_data);

    RunModel(helper, model_data, baseline_fetches, false);
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
    model.ToProto().SerializeToString(&model_data);

    RunModel(helper, model_data, outputs);
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


TEST(AVX2S8U8TransformerTests, QGemm) {

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

  SessionOptions so;
  so.use_per_session_threads = false;
  so.session_logid = "QGemmAvx2PrecTransformer";
  so.session_log_verbosity_level = 1;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsAvx2PrecisionMode, "1"));
  so.graph_optimization_level = TransformerLevel::Level2;
  test.Run(so);

}


TEST(AVX2S8U8TransformerTests, MatMulInteger) {
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

  SessionOptions so;
  so.use_per_session_threads = false;
  so.session_logid = "QGemmAvx2PrecTransformer";
  so.session_log_verbosity_level = 1;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsAvx2PrecisionMode, "1"));
  so.graph_optimization_level = TransformerLevel::Level2;
  test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {kNupharExecutionProvider});
}


}  // namespace test
}  // namespace onnxruntime
