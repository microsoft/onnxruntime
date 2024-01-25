// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <string>
#include <thread>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/cpu/cpu_provider_factory.h"  // For OrtSessionOptionsAppendExecutionProvider_CPU
#include "core/session/inference_session.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Create a model with Case + Add (quantized)
// cast_input -> Cast -> Q -> DQ \
//                                Add -> Q -> DQ -> output
//             input2 -> Q -> DQ /
static GetTestModelFn BuildCastAddTestCase() {
  return [](ModelTestBuilder& builder) {
    // Creat Cast node int32 -> float32
    NodeArg* cast_input = MakeTestInput(builder, TestInputDef<int32_t>({2, 3}, false, {0, 1, 0, 1, 0, 1}));

    auto* cast_output = builder.MakeIntermediate();
    Node& cast_node = builder.AddNode("Cast", {cast_input}, {cast_output});
    cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));

    // Create Add node
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    auto* add_input1_qdq = AddQDQNodePair<uint8_t>(builder, cast_output, q_parameter.scale, q_parameter.zero_point);

    NodeArg* add_input2 = MakeTestInput(builder, TestInputDef<float>({2, 3}, false, data));
    auto* add_input2_qdq = AddQDQNodePair<uint8_t>(builder, add_input2, q_parameter.scale, q_parameter.zero_point);

    auto* add_output = builder.MakeIntermediate();

    builder.AddNode("Add", {add_input1_qdq, add_input2_qdq}, {add_output});

    // add_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add_output, q_parameter.scale, q_parameter.zero_point);
  };
}

// Test that models with 2 inputs which has different data type can still generate the context binary
TEST_F(QnnHTPBackendTests, QnnContextBinaryGeneration2InputTypes) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildCastAddTestCase()(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_binary_file = "./qnn_context_binary_int32_fp32_inputs_test.onnx";
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_binary_file.c_str());

  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  // clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

// Generate context cache model from the ONNX models with 2 inputs.
// The generated model should have same input order.
// The input ONNX model is created in the way that the model inputs order
// is different with the order in the graph (topological order).
// It cause issue if the generated model doesn't set the inputs/outputs explicitly.
TEST_F(QnnHTPBackendTests, QnnContextGeneration2InputsOrderIssue) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  const std::string context_binary_file = "./qnn_ctx_2_inputs_order_test_gen.onnx";
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_binary_file.c_str());

  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx_2_inputs_order_test.onnx"), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(context_binary_file), model, nullptr, DefaultLoggingManager().DefaultLogger()));
  auto inputs = model->MainGraph().GetInputs();
  EXPECT_TRUE(inputs.size() == 2);
  EXPECT_TRUE(inputs[0]->Name() == "attention_mask");
  EXPECT_TRUE(inputs[1]->Name() == "Add_input_0");

  // clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

// Run QDQ model on HTP 3 times
// 1st run will generate the Qnn context cache onnx file
// 2nd run directly loads and run from Qnn context cache model
TEST_F(QnnHTPBackendTests, QnnContextBinaryCacheEmbedModeTest) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  const std::string context_binary_file = "./qnn_context_binary_test.onnx";
  std::remove(context_binary_file.c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  // 2nd run directly loads and run from Qnn context cache model
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       context_binary_file);
  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

// Run QDQ model on HTP 3 times
// 1st run will generate the Onnx skeleton file + Qnn context cache binary file
// 2nd run directly loads and run from Onnx skeleton file + Qnn context cache binary file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCacheNonEmbedModeTest) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  const std::string context_binary_file = "./qnn_context_cache_non_embed.onnx";
  std::string qnn_ctx_bin = "qnn_context_cache_non_embed.onnx_QNNExecutionProvider_QNN_8283143575221199085_1_0.bin";

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  std::remove(context_binary_file.c_str());
  std::remove(qnn_ctx_bin.c_str());

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(qnn_ctx_bin));

  // 2nd run directly loads and run from Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       context_binary_file);

  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
  ASSERT_EQ(std::remove(qnn_ctx_bin.c_str()), 0);
}

// Run QDQ model on HTP 2 times
// 1st run will generate the Onnx skeleton file + Qnn context cache binary file
// Then delete the context bin file to make the 2nd sesssion.Initialize() return the status with code INVALID_GRAPH
TEST_F(QnnHTPBackendTests, QnnContextBinaryCache_InvalidGraph) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  const std::string context_binary_file = "./qnn_context_cache_non_embed.onnx";
  std::filesystem::path context_bin = "qnn_context_cache_non_embed.onnx_QNNExecutionProvider_QNN_8283143575221199085_1_0.bin";
  std::remove(context_binary_file.c_str());
  std::remove(context_bin.string().c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_bin));
  // Delete the Qnn context cache binary file
  EXPECT_TRUE(std::filesystem::remove(context_bin));

  // loads and run from Onnx skeleton file + Qnn context cache binary file
  onnx::ModelProto model_proto;
  onnxruntime::Model qnn_ctx_model;
  // Load the QNN context cache model from path specified
  ASSERT_STATUS_OK(qnn_ctx_model.Load(ToPathString(context_binary_file), model_proto));
  std::string qnn_ctx_model_data;
  model_proto.SerializeToString(&qnn_ctx_model_data);

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  std::string provider_type = kCpuExecutionProvider;
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(qnn_ctx_model_data.data(), static_cast<int>(qnn_ctx_model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);

  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

std::string CreateQnnCtxModelWithNonEmbedMode(std::string external_bin_path) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 11}, {kMSDomain, 1}};
  auto& logging_manager = DefaultLoggingManager();
  onnxruntime::Model model("QNN_ctx_model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  std::vector<int64_t> shape = {2, 3};
  NodeArg* graph_input = MakeTestInput(helper, TestInputDef<float>(shape, true, {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}));
  auto* graph_output = helper.MakeOutput<float>(shape);
  Node& ep_context_node = helper.AddNode("EPContext", {graph_input}, {graph_output}, kMSDomain);
  ep_context_node.AddAttribute("embed_mode", static_cast<int64_t>(0));
  // The .. in the path will cause INVALID_GRAPH
  ep_context_node.AddAttribute("ep_cache_context", external_bin_path);
  ep_context_node.AddAttribute("partition_name", "QNNExecutionProvider_QNN_1110111000111000111_1_0");
  ep_context_node.AddAttribute("source", "QNN");
  helper.SetGraphOutputs();
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  return model_data;
}

// Create a model with EPContext node. Set the node property ep_cache_context has ".."
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryRelativePathTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("../qnn_context.bin");

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);
}

// Create a model with EPContext node. Set the node property ep_cache_context has absolute path
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryAbsolutePathTest) {
#if defined(_WIN32)
  std::string external_ctx_bin_path = "D:/qnn_context.bin";
#else
  std::string external_ctx_bin_path = "/data/qnn_context.bin";
#endif
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode(external_ctx_bin_path);

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);
}

// Create a model with EPContext node. Set the node property ep_cache_context to a file not exist
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryFileNotExistTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("qnn_context_not_exist.bin");

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);
}

// Create a model with EPContext node. Set the node property ep_cache_context to empty string
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryFileEmptyStringTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("");

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::INVALID_GRAPH);
}

// Run QDQ model on HTP with 2 inputs
// 1st run will generate the Qnn context cache onnx file
// 2nd run directly loads and run from Qnn context cache model
TEST_F(QnnHTPBackendTests, QnnContextBinary2InputsTest) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  const std::string context_binary_file = "./qnn_context_binary_2inputs_test.onnx";
  std::remove(context_binary_file.c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);

  const TestInputDef<float> input_def1({1, 2, 3}, false, -10.0f, 10.0f);
  const TestInputDef<float> input_def2({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Add";

  // Runs model with DQ-> Add-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def1, input_def2}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def1, input_def2}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  // 2nd run directly loads and run from Qnn context cache model
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def1, input_def2}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def1, input_def2}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       context_binary_file);
  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
