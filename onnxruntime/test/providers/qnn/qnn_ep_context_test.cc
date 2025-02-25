// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <string>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/graph/model_saving_options.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static int64_t GetNodeAttr(const Node& node, const std::string& attr_name, int64_t default_val) {
  const auto& attributes = node.GetAttributes();
  if (auto entry = attributes.find(attr_name); entry != attributes.end()) {
    return entry->second.i();
  }

  return default_val;
}

static const std::string& GetNodeAttr(const Node& node, const std::string& attr_name, const std::string& default_val) {
  const auto& attributes = node.GetAttributes();
  if (auto entry = attributes.find(attr_name); entry != attributes.end()) {
    return entry->second.s();
  }

  return default_val;
}

// Create a model with FusedMatMul + Add (quantized)
// input1 -> Add -> Q -> DQ ----
//                              |
//        input2 -> Q -> DQ -> FusedMatMul -> Q -> DQ -> output
static GetTestModelFn BuildGraphWithQAndNonQ(bool single_ep_node = true) {
  return [single_ep_node](ModelTestBuilder& builder) {
    // Creat non-quantized FusedMatMul node1
    std::vector<float> data(200 * 200, 1.0f);
    NodeArg* input1 = MakeTestInput(builder, TestInputDef<float>({200, 200}, false, data));
    NodeArg* add1_ini_input2 = MakeTestInput(builder, TestInputDef<float>({200, 200}, true, data));

    auto* add1_output = builder.MakeIntermediate();
    builder.AddNode("FusedMatMul", {input1, add1_ini_input2}, {add1_output}, kMSDomain);

    // Create quantized Add node2
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    auto* add2_input1_qdq = AddQDQNodePair<uint8_t>(builder, add1_output, q_parameter.scale, q_parameter.zero_point);

    NodeArg* add2_input2 = MakeTestInput(builder, TestInputDef<float>({200, 200}, true, data));
    auto* add2_input2_qdq = AddQDQNodePair<uint8_t>(builder, add2_input2, q_parameter.scale, q_parameter.zero_point);

    auto* add2_output = builder.MakeIntermediate();

    builder.AddNode("Add", {add2_input1_qdq, add2_input2_qdq}, {add2_output});

    if (single_ep_node) {
      // add_output -> Q -> DQ -> output
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add2_output, q_parameter.scale, q_parameter.zero_point);
    } else {
      auto* add3_input1_qdq = AddQDQNodePair<uint8_t>(builder, add2_output, q_parameter.scale, q_parameter.zero_point);
      NodeArg* add3_ini_input2 = MakeTestInput(builder, TestInputDef<float>({200, 200}, true, data));

      auto* add3_output = builder.MakeIntermediate();
      builder.AddNode("FusedMatMul", {add3_input1_qdq, add3_ini_input2}, {add3_output}, kMSDomain);

      // Create quantized Add node4
      auto* add4_input1_qdq = AddQDQNodePair<uint8_t>(builder, add3_output, q_parameter.scale, q_parameter.zero_point);

      NodeArg* add4_input2 = MakeTestInput(builder, TestInputDef<float>({200, 200}, true, data));
      auto* add4_input2_qdq = AddQDQNodePair<uint8_t>(builder, add4_input2, q_parameter.scale, q_parameter.zero_point);

      auto* add4_output = builder.MakeIntermediate();

      builder.AddNode("Add", {add4_input1_qdq, add4_input2_qdq}, {add4_output});
      // add_output -> Q -> DQ -> output
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, add4_output, q_parameter.scale, q_parameter.zero_point);
    }
  };
}

void QnnContextBinaryMultiPartitionTestBody(bool single_ep_node = true) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_binary_file = "./qnn_context_binary_multi_partition_test.onnx";
  std::remove(context_binary_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_binary_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  int ep_context_node_count = 0;
  int non_ep_context_node_count = 0;
  std::shared_ptr<Model> ctx_model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(context_binary_file), ctx_model, nullptr, DefaultLoggingManager().DefaultLogger()));
  auto& ctx_graph = ctx_model->MainGraph();
  for (auto& node : ctx_graph.Nodes()) {
    if (node.OpType() == "EPContext") {
      ++ep_context_node_count;
      // validate the fix for the partition issue relate to QDQ model
      ASSERT_EQ(node.InputDefs().size(), 1);
    } else {
      ++non_ep_context_node_count;
    }
  }

  int expected_node_count = single_ep_node ? 1 : 2;
  ASSERT_EQ(ep_context_node_count, expected_node_count);
  ASSERT_EQ(non_ep_context_node_count, expected_node_count);

  Ort::SessionOptions so2;
  // context file path is required if it's non-embed mode and the model is loaded from memory
  so2.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_binary_file.c_str());
  so2.AppendExecutionProvider("QNN", provider_options);

  std::string ctx_model_data;
  ctx_model->ToProto().SerializeToString(&ctx_model_data);
  Ort::Session session2(*ort_env, ctx_model_data.data(), ctx_model_data.size(), so2);

  // clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

// Test that models with 1 non-quantized FusedMatMul node and 1 quantized Add node can still generate the context binary
// The generated Onnx model has 1 FusedMatMul node and 1 EPContext node
TEST_F(QnnHTPBackendTests, QnnContextBinaryMultiPartitionSupport1) {
  bool single_ep_node = true;
  QnnContextBinaryMultiPartitionTestBody(single_ep_node);
}

// Test that models with 2 non-quantized FusedMatMul nodes and 2 quantized Add nodes can still generate the context binary
// The generated Onnx model has 2 FusedMatMul nodes and 1 EPContext nodes
TEST_F(QnnHTPBackendTests, QnnContextBinaryMultiPartitionSupport2) {
  bool single_ep_node = false;
  QnnContextBinaryMultiPartitionTestBody(single_ep_node);
}

void EpCtxCpuNodeWithExternalIniFileTestBody(bool expect_external_ini_file) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildGraphWithQAndNonQ(true)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  ModelSavingOptions model_saving_options{10};
  // dump the model in testdata folder in case it hides the bug that not able to find model not in current dir
  const std::string model_with_ext = "./testdata/model_external.onnx";
  const std::string model_ext_file = "model_external.bin";
  ASSERT_STATUS_OK(Model::SaveWithExternalInitializers(model, model_with_ext,
                                                       model_ext_file, model_saving_options));

  EXPECT_TRUE(std::filesystem::exists(model_with_ext.c_str()));
  std::string model_ext_file_full_path = "./testdata/" + model_ext_file;
  EXPECT_TRUE(std::filesystem::exists(model_ext_file_full_path.c_str()));

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AppendExecutionProvider("QNN", provider_options);
  const std::string ep_context_model_file = "./qnn_ctx_part_external_ini_ctx.onnx";
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_model_file.c_str());
  const std::string external_ini_file = "./qnn_ctx_part_external_ini.bin";
  if (expect_external_ini_file) {
    // Set the external ini file name will force all initializers to the external file
    so.AddConfigEntry(kOrtSessionOptionsEpContextModelExternalInitializersFileName, external_ini_file.c_str());
  }  // otherwise all initializers are in Onnx file, no external data file generated

  Ort::Session session(*ort_env, ToPathString(model_with_ext).c_str(), so);

  EXPECT_TRUE(std::filesystem::exists(ep_context_model_file.c_str()));
  if (expect_external_ini_file) {
    EXPECT_TRUE(std::filesystem::exists(external_ini_file.c_str()));
    ASSERT_EQ(std::remove(external_ini_file.c_str()), 0);
  } else {
    EXPECT_FALSE(std::filesystem::exists(external_ini_file.c_str()));
  }

  // clean up
  ASSERT_EQ(std::remove(model_with_ext.c_str()), 0);
  ASSERT_EQ(std::remove(model_ext_file_full_path.c_str()), 0);
  ASSERT_EQ(std::remove(ep_context_model_file.c_str()), 0);
}

// Set the external initializer size threshold to 1024 so FusedMatMul (which fallback on CPU)
// will dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithExternalWeights) {
  EpCtxCpuNodeWithExternalIniFileTestBody(true);
}

// Use the default external initializer size threshold (1024000) so FusedMatMul (which fallback on CPU)
// will NOT dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithoutExternalWeights) {
  EpCtxCpuNodeWithExternalIniFileTestBody(false);
}

// Set ep.context_file_path to folder path which is not a valid option, check the error message
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationFolderPathNotExpected) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_folder_not_expected/";
  std::remove(ep_context_onnx_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

// Create session 1 to generate context binary file
// Create session 2 to do same thing, make sure session 2 failed because file exist already
// Make sure no new file over write from session 2
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationNoOverWrite) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_no_over_write.onnx";
  const std::string ep_context_binary_file = "./ep_context_no_over_write.onnx_QNNExecutionProvider_QNN_10880527342279992768_1_0.bin";

  std::remove(ep_context_onnx_file.c_str());
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session1(*ort_env, model_data_span.data(), model_data_span.size(), so);

  auto modify_time_1 = std::filesystem::last_write_time(ep_context_binary_file);

  try {
    Ort::Session session2(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_FAIL);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("exist already."));
    auto modify_time_2 = std::filesystem::last_write_time(ep_context_binary_file);
    ASSERT_EQ(modify_time_1, modify_time_2);
  }

  ASSERT_EQ(std::remove(ep_context_onnx_file.c_str()), 0);
  ASSERT_EQ(std::remove(ep_context_binary_file.c_str()), 0);
}

// Create a model with Cast + Add (quantized)
// cast_input -> Cast -> Q -> DQ ----+
//                                   |
//             input2 -> Q -> DQ -> Add -> Q -> DQ -> output
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

// Create a model with Add (quantized)
// input1 -> Q -> DQ ----
//                       |
// input2 -> Q -> DQ -> Add -> Q -> DQ -> output
static GetTestModelFn BuildAddTestCase() {
  return [](ModelTestBuilder& builder) {
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    NodeArg* add_input1 = MakeTestInput(builder, TestInputDef<float>({2, 3}, false, data));
    auto* add_input1_qdq = AddQDQNodePair<uint8_t>(builder, add_input1, q_parameter.scale, q_parameter.zero_point);

    NodeArg* add_input2 = MakeTestInput(builder, TestInputDef<float>({2, 3}, true, data));
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
  provider_options["offload_graph_io_quantization"] = "0";

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
  std::remove(context_binary_file.c_str());
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
  provider_options["offload_graph_io_quantization"] = "0";

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

TEST_F(QnnHTPBackendTests, QnnContextGenerationNodeNamePrefix) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";
  std::string node_name_prefix = "node_name_prefix_test";

  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  const std::string context_binary_file = "./qnn_ctx_2_inputs_order_test_gen.onnx";
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_binary_file.c_str());
  so.AddConfigEntry(kOrtSessionOptionEpContextNodeNamePrefix, node_name_prefix.c_str());
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx_2_inputs_order_test.onnx"), so);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(context_binary_file), model, nullptr, DefaultLoggingManager().DefaultLogger()));
  for (auto& node : model->MainGraph().Nodes()) {
    if (node.OpType() == "EPContext") {
      EXPECT_TRUE(node.Name().find(node_name_prefix) != std::string::npos);
    }
  }

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
  provider_options["offload_graph_io_quantization"] = "0";
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
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_binary_file = "./testdata/qnn_context_cache_non_embed.onnx";
  std::string qnn_ctx_bin = "./testdata/qnn_context_cache_non_embed.onnx_QNNExecutionProvider_QNN_8283143575221199085_1_0.bin";

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

  std::unordered_map<std::string, std::string> session_option_pairs2;
  // Need to set the context file path since TestQDQModelAccuracy load the model from memory
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  // 2nd run directly loads and run from Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kERROR,
                       context_binary_file,
                       session_option_pairs2);

  // load the model from file
  std::vector<char> buffer;
  {
    std::ifstream file(context_binary_file, std::ios::binary | std::ios::ate);
    if (!file)
      ORT_THROW("Error reading model");
    buffer.resize(narrow<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), buffer.size()))
      ORT_THROW("Error reading model");
  }

  Ort::SessionOptions so;  // No need to set the context file path in so since it's load from file
  so.AppendExecutionProvider("QNN", provider_options);
#ifdef _WIN32
  std::wstring ctx_model_file(context_binary_file.begin(), context_binary_file.end());
#else
  std::string ctx_model_file(context_binary_file.begin(), context_binary_file.end());
#endif
  Ort::Session session(*ort_env.get(), ctx_model_file.c_str(), so);

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
  provider_options["offload_graph_io_quantization"] = "0";
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
  provider_options["offload_graph_io_quantization"] = "0";

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
  provider_options["offload_graph_io_quantization"] = "0";

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
  provider_options["offload_graph_io_quantization"] = "0";

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
  provider_options["offload_graph_io_quantization"] = "0";

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
  provider_options["offload_graph_io_quantization"] = "0";
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

// Context binary only contains a single QNN graph, generated context cache model (detached mode) only has 1 EPContext node
// Create another Onnx model which also reference to the bin file,
// but the node name is not same with the QNN graph name inside the bin file.
// This is to support backward compatible for the models generated before the PR that
// make context generation support multi-partition
TEST_F(QnnHTPBackendTests, QnnContextBinaryCache_SingleNodeNameNotMatchGraphNameInCtx) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";
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

  const std::unordered_map<std::string, int> domain_to_version = {{"", 11}, {kMSDomain, 1}};
  auto& logging_manager = DefaultLoggingManager();
  onnxruntime::Model model("QNN_ctx_model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  std::vector<int64_t> shape = {1, 2, 3};
  NodeArg* graph_input = MakeTestInput(helper, TestInputDef<float>(shape, false, {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}));
  auto* graph_output = helper.MakeOutput<float>(shape);
  Node& ep_context_node = helper.AddNode("EPContext", {graph_input}, {graph_output}, kMSDomain);
  ep_context_node.AddAttribute("embed_mode", static_cast<int64_t>(0));
  ep_context_node.AddAttribute("ep_cache_context", context_bin.string());
  ep_context_node.AddAttribute("partition_name", "QNNExecutionProvider_QNN_1110111000111000111_1_0");
  ep_context_node.AddAttribute("source", "QNNExecutionProvider");
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  // loads and run from Onnx skeleton file + Qnn context cache binary file

  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(QnnExecutionProviderWithOptions(provider_options)));
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  // Verify the return status with code INVALID_GRAPH
  ASSERT_TRUE(session_object.Initialize().Code() == common::StatusCode::OK);

  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
  ASSERT_EQ(std::remove(context_bin.string().c_str()), 0);
}

// Model has 2 EPContext nodes, both with main_context=1 and embedded context binary
TEST_F(QnnHTPBackendTests, QnnMultiContextEmbeded) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_embed.onnx"), so);
}

// Model has 2 EPContext nodes, both with main_context=1 and external context binary
TEST_F(QnnHTPBackendTests, QnnMultiContextExternal) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.AppendExecutionProvider("QNN", provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_external.onnx"), so);
}

static void CreateQdqModel(const std::string& model_file_name, const Logger& logger) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};
  onnxruntime::Model model(model_file_name, false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildAddTestCase()(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  ASSERT_STATUS_OK(onnxruntime::Model::Save(model, ToPathString(model_file_name)));
}

static void DumpModelWithSharedCtx(const ProviderOptions& provider_options,
                                   const std::string& onnx_model_path1,
                                   const std::string& onnx_model_path2) {
  SessionOptions so;
  so.session_logid = "qnn_ctx_model_logger";
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1"));
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0"));
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  auto qnn_ep = QnnExecutionProviderWithOptions(provider_options, &so);
  std::shared_ptr<IExecutionProvider> qnn_ep_shared(std::move(qnn_ep));

  InferenceSessionWrapper session_object1{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object1.RegisterExecutionProvider(qnn_ep_shared));
  ASSERT_STATUS_OK(session_object1.Load(ToPathString(onnx_model_path1)));
  ASSERT_STATUS_OK(session_object1.Initialize());

  InferenceSessionWrapper session_object2{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.RegisterExecutionProvider(qnn_ep_shared));
  ASSERT_STATUS_OK(session_object2.Load(ToPathString(onnx_model_path2)));
  ASSERT_STATUS_OK(session_object2.Initialize());
}

// from the last context ache Onnx model, find the EPContext node with main_context=1,
// and get the QNN context binary file name, thie context binary contains all graphs from all Onnx models
static void GetLastContextBinaryFileName(const std::string last_onnx_ctx_file,
                                         std::string& last_ctx_bin_file,
                                         const Logger& logger) {
  std::shared_ptr<Model> ctx_model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(last_onnx_ctx_file), ctx_model, nullptr, logger));
  auto& ctx_graph = ctx_model->MainGraph();
  for (auto& node : ctx_graph.Nodes()) {
    if (node.OpType() == "EPContext") {
      int64_t is_main_context = GetNodeAttr(node, "main_context", static_cast<int64_t>(0));
      if (1 == is_main_context) {
        last_ctx_bin_file = GetNodeAttr(node, "ep_cache_context", "");
        return;
      }
    }
  }
}

// Update generated context cache Onnx model to make the main EPContext node point to
// the last QNN context binary file
// Remove not used QNN context binary file, only keep the last one which contains all graphs
static void UpdateEpContextModel(const std::vector<std::string>& ep_ctx_files,
                                 const std::string& last_qnn_ctx_binary_file_name,
                                 const Logger& logger) {
  for (auto ep_ctx_file : ep_ctx_files) {
    std::shared_ptr<Model> ctx_model;
    auto path_str = ToPathString(ep_ctx_file);
    ASSERT_STATUS_OK(Model::Load(path_str, ctx_model, nullptr, logger));
    auto& ctx_graph = ctx_model->MainGraph();
    GraphViewer graph_viewer(ctx_graph);
    auto path = std::filesystem::path(path_str);

    for (auto& node : ctx_graph.Nodes()) {
      if (node.OpType() == "EPContext") {
        int64_t is_main_context = GetNodeAttr(node, "main_context", static_cast<int64_t>(0));
        if (1 == is_main_context) {
          std::string old_qnn_ctx_binary_file_name = GetNodeAttr(node, "ep_cache_context", "");
          auto file_path = path.replace_filename(old_qnn_ctx_binary_file_name);
          std::remove(file_path.string().c_str());
          node.ClearAttribute("ep_cache_context");
          node.AddAttribute("ep_cache_context", last_qnn_ctx_binary_file_name);
        }
      }
    }
    std::remove(ep_ctx_file.c_str());
    ASSERT_STATUS_OK(Model::Save(*ctx_model.get(), ToPathString(ep_ctx_file)));
  }
}

static void GetModelInputNames(const std::string& model_path,
                               std::vector<std::string>& input_names,
                               std::vector<std::string>& output_names,
                               const Logger& logger) {
  std::shared_ptr<Model> ctx_model;
  auto path_str = ToPathString(model_path);
  ASSERT_STATUS_OK(Model::Load(path_str, ctx_model, nullptr, logger));
  auto& ctx_graph = ctx_model->MainGraph();

  auto& inputs = ctx_graph.GetInputs();
  for (auto input : inputs) {
    input_names.push_back(input->Name());
  }

  auto& outputs = ctx_graph.GetOutputs();
  for (auto output : outputs) {
    output_names.push_back(output->Name());
  }
}

// 1. Create 2 QDQ models
// 2. Initialize 2 Ort sessions which share the same QNN EP from these 2 QDQ models
// with EpContextEnable = 1, to dump the context binary
// so, the 2nd context binary contains the graph from the 1st model
// 3. Change the 1st context model to point to the 2nd context binary file
// 4. Start 2 ort session from the dumped context model,
// The 2nd session uses graph from 1st session
// 5. Run the 2nd session
TEST_F(QnnHTPBackendTests, QnnContextShareAcrossSessions1) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path, DefaultLoggingManager().DefaultLogger());
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    ctx_model_paths.push_back(model_path + "_ctx.onnx");
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  // Get the last context binary file name
  std::string last_qnn_ctx_binary_file_name;
  GetLastContextBinaryFileName(ctx_model_paths.back(), last_qnn_ctx_binary_file_name,
                               DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!last_qnn_ctx_binary_file_name.empty());

  // Update generated context cache Onnx model to make the main EPContext node point to
  // the last QNN context binary file
  // Remove not used QNN context binary file, only keep the last one which contains all graphs
  std::vector<std::string> ctx_model_paths_to_update(ctx_model_paths);
  ctx_model_paths_to_update.pop_back();
  UpdateEpContextModel(ctx_model_paths_to_update, last_qnn_ctx_binary_file_name,
                       DefaultLoggingManager().DefaultLogger());

  Ort::SessionOptions so1;
  so1.SetLogId("so1");
  so1.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so1.AppendExecutionProvider("QNN", provider_options);
  Ort::SessionOptions so2;
  so2.SetLogId("so2");
  so2.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so2.AppendExecutionProvider("QNN", provider_options);

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif
  Ort::Session session1(*ort_env, ctx_model_file1.c_str(), so1);
  Ort::Session session2(*ort_env, ctx_model_file2.c_str(), so2);

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names,
                     DefaultLoggingManager().DefaultLogger());

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = session1.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names_c.data(), 1);

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(last_qnn_ctx_binary_file_name.c_str());
}

// 1. Create 2 QDQ models
// 2. Initialize 2 Ort sessions which share the same QNN EP from these 2 QDQ models
// with EpContextEnable = 1, to dump the context binary
// so, the 2nd context binary contains the graph from the 1st model
// 3. Change the 1st context model to point to a context binary file which is not exist
// 4. Start 2 ort session from the dumped context model,
// The 1st session uses the 2nd model, the 2nd session uses the 1st model
// so the 2nd session uses graph from the 1st session
// 6. Run the 2nd session
TEST_F(QnnHTPBackendTests, QnnContextShareAcrossSessions2) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  provider_options["offload_graph_io_quantization"] = "0";

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share21.onnx", "./weight_share22.onnx"};
  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path, DefaultLoggingManager().DefaultLogger());
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    ctx_model_paths.push_back(model_path + "_ctx.onnx");
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  // Get the last context binary file name
  std::string last_qnn_ctx_binary_file_name;
  GetLastContextBinaryFileName(ctx_model_paths.back(), last_qnn_ctx_binary_file_name,
                               DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(!last_qnn_ctx_binary_file_name.empty());

  // Update generated context cache Onnx model to make the main EPContext node point to
  // the last QNN context binary file
  // Remove not used QNN context binary file, only keep the last one which contains all graphs
  std::vector<std::string> ctx_model_paths_to_update(ctx_model_paths);
  ctx_model_paths_to_update.pop_back();
  // The 2nd model still point to the context binary which includes all graphs
  // The 1st model point to file not exists
  UpdateEpContextModel(ctx_model_paths_to_update, "file_not_exist.bin",
                       DefaultLoggingManager().DefaultLogger());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so.AppendExecutionProvider("QNN", provider_options);

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif
  // Create session from the 2nd model first
  Ort::Session session1(*ort_env, ctx_model_file2.c_str(), so);
  Ort::Session session2(*ort_env, ctx_model_file1.c_str(), so);

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names,
                     DefaultLoggingManager().DefaultLogger());

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = session1.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names_c.data(), 1);

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(last_qnn_ctx_binary_file_name.c_str());
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
