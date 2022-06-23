#include "core/optimizer/avx2_weight_s8_to_u8.h"

#include <functional>
#include <string>
#include <vector>

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/optimizer/graph_transform_test_builder.h"

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
    const std::vector<uint8_t>& A_zero_point, const std::vector<WeightType>& B_zero_point,
    const std::vector<float>& Bias
) {
  auto* input_A = helper.MakeInput<uint8_t>(A_dims, A_data);
  auto* input_B = helper.MakeInitializer<WeightType>(B_dims, B_data);
  auto* input_a_scale = helper.MakeInitializer<float>({1}, A_scale);
  auto* input_b_scale = helper.MakeInitializer<float>({B_dims.back()}, B_scale);
  auto* input_a_zero_point = helper.MakeInitializer<uint8_t>({1}, A_zero_point);
  auto* input_b_zero_point = helper.MakeInitializer<WeightType>({B_dims.back()}, B_zero_point);
  auto* input_bias = helper.MakeInitializer<float>({B_dims.back()}, Bias);

  auto* output_arg = helper.MakeOutput();

  helper.AddNode("MatMulIntegerToFloat", {input_A, input_B, input_a_scale, input_b_scale,
      input_a_zero_point, input_b_zero_point, input_bias}, {output_arg}, kMSDomain);
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

  std::vector<float> A_scale = random.Uniform<float>(std::array<int64_t, 1>{1}, -0.1f, 0.1f);
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
void BuildDynamicQuantizeMatMulGraph(ModelTestBuilder& helper,
                                    const std::vector<int64_t> A_dims, const std::vector<int64_t> B_dims,
                                    const std::vector<float>& A_data, const std::vector<WeightType>& B_data,
                                    const std::vector<float>& B_scale, const std::vector<WeightType>& B_zero_point,
                                    const std::vector<float>& Bias) {
  auto* input_A = helper.MakeInput<float>(A_dims, A_data);
  auto* input_B = helper.MakeInitializer<WeightType>(B_dims, B_data);
  auto* input_b_scale = helper.MakeInitializer<float>({B_dims.back()}, B_scale);
  auto* input_b_zero_point = helper.MakeInitializer<WeightType>({B_dims.back()}, B_zero_point);
  auto* input_bias = helper.MakeInitializer<float>({B_dims.back()}, Bias);

  auto* output_arg = helper.MakeOutput();

  helper.AddNode("DynamicQuantizeMatMul", {input_A, input_B, input_b_scale, input_b_zero_point, input_bias}, {output_arg}, kMSDomain);
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

    BuildDynamicQuantizeMatMulGraph<int8_t>(helper, A_dims, B_dims, A_data, s8_B_data,
                                             B_scale, s8_b_zero_point, Bias);

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


}  // namespace test
}  // namespace onnxruntime
