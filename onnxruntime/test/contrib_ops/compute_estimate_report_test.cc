// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <gsl/gsl>
#include "nlohmann/json.hpp"
#include "core/common/path_string.h"
#include "core/framework/allocator.h"
#include "core/framework/compute_estimate_report.h"
#include "core/graph/constants.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/model.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {
namespace {

using Json = nlohmann::json;

class ComputeEstimateReportTest : public ::testing::Test {
 protected:
  ComputeEstimateReportTest()
      : model_("compute_report", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
               {{kOnnxDomain, 17}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()),
        builder_(model_.MainGraph()) {
    static std::atomic<uint64_t> sequence{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("ort_compute_report_" + std::to_string(stamp) + "_" + std::to_string(sequence++) + ".json");
    cleanup_ = ScopedFileDeleter(path_.native());
  }

  Node& AddMatMul(std::optional<std::vector<int64_t>> shape = std::vector<int64_t>{2, 3, 64},
                  bool add_identity = false) {
    auto* a = builder_.MakeInput<float>(shape);
    auto* b = builder_.MakeInitializer<uint8_t>({32, 2, 16}, std::vector<uint8_t>(1024, 0x88));
    auto* scales = builder_.MakeInitializer<float>({32, 2}, std::vector<float>(64, 1.0f));
    auto* output = add_identity ? builder_.MakeIntermediate() : builder_.MakeOutput();
    auto& node = builder_.graph_.AddNode(builder_.graph_.GenerateNodeName("projection \"quoted\"\\\n"),
                                         "MatMulNBits", "", {a, b, scales}, {output}, nullptr, kMSDomain);
    node.AddAttribute("K", int64_t{64});
    node.AddAttribute("N", int64_t{32});
    node.AddAttribute("bits", int64_t{4});
    node.AddAttribute("block_size", int64_t{32});
    if (add_identity) {
      builder_.AddNode("Identity", {output}, {builder_.MakeOutput()});
    }
    builder_.SetGraphOutputs();
    return node;
  }

  Json Report() {
    std::string value;
    const auto status = BuildComputeEstimateReport(model_.MainGraph(), value);
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    return Json::parse(value);
  }

  Json ReadReport() {
    std::ifstream stream(path_);
    EXPECT_TRUE(stream.is_open());
    return Json::parse(stream);
  }

  std::string ModelBytes() {
    return model_.ToProto().SerializeAsString();
  }

  Status Initialize(InferenceSession& session) {
    std::istringstream stream(ModelBytes());
    ORT_RETURN_IF_ERROR(session.Load(stream));
    return session.Initialize();
  }

  SessionOptions ReportOptions() {
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    const auto utf8_path = ToUTF8String(path_.native());
    EXPECT_TRUE(options.config_options.AddConfigEntry(kOrtSessionOptionsComputeEstimateReportPath,
                                                      utf8_path.c_str())
                    .IsOK());
    return options;
  }

  Model model_;
  ModelTestBuilder builder_;
  std::filesystem::path path_;
  ScopedFileDeleter cleanup_;
};

TEST_F(ComputeEstimateReportTest, ComponentContractAndDeterminism) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  EXPECT_EQ(report["schema_version"], 1);
  EXPECT_EQ(report["estimator_support"], "matmul_nbits_core_only");
  ASSERT_EQ(report["nodes"].size(), 1u);
  const auto& entry = report["nodes"][0];
  EXPECT_EQ(entry["node_name"], node.Name());
  EXPECT_EQ(entry["graph_path"], "main");
  EXPECT_EQ(entry["opset_version"], 1);
  const auto& component = entry["components"][0];
  EXPECT_EQ(component["operation_count"], "24576");
  EXPECT_EQ(component["metric"], "dense_equivalent_ops");
  EXPECT_EQ(component["input_basis"], "static_concrete");
  EXPECT_EQ(component["execution_phase"], "per_invocation");
  EXPECT_EQ(component["basis"], "logical_semantics");
  EXPECT_EQ(component["convention_version"], "core_arithmetic_v1");
  EXPECT_EQ(component["excluded_components"],
            Json::array({"dequantization", "bias", "packing", "conversion", "padding", "route_specific_extra_work"}));
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "24576");
  EXPECT_EQ(report["summary"]["complete_for_declared_graph_and_component_scope"], true);
  EXPECT_EQ(report["summary"]["whole_operator_work_complete"], false);
  EXPECT_EQ(report, Report());
}

TEST_F(ComputeEstimateReportTest, UnknownShapesHaveNoCount) {
  AddMatMul(std::vector<int64_t>{-1, 64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  const auto& entry = report["nodes"][0];
  EXPECT_EQ(entry["unavailable_reason"], "unknown_shape");
  EXPECT_FALSE(entry["components"][0].contains("operation_count"));
  EXPECT_EQ(entry["input_shapes"][0]["dimensions"], Json::array({nullptr, 64}));
  EXPECT_EQ(report["summary"]["unavailable_reason_counts"]["unknown_shape"], 1);
  EXPECT_EQ(report["summary"]["complete_for_declared_graph_and_component_scope"], false);
}

TEST_F(ComputeEstimateReportTest, PartialCoverage) {
  AddMatMul(std::vector<int64_t>{2, 3, 64}, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  EXPECT_EQ(report["summary"]["reported_node_count"], 2);
  EXPECT_EQ(report["summary"]["estimated_node_count"], 1);
  EXPECT_EQ(report["summary"]["unavailable_node_count"], 1);
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "24576");
  EXPECT_EQ(report["summary"]["complete_for_declared_graph_and_component_scope"], false);
}

TEST_F(ComputeEstimateReportTest, OpaquePartitionDoesNotCountUnderlyingCore) {
  const auto& node = AddMatMul(std::vector<int64_t>{2, 3, 64}, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  IndexedSubGraph partition;
  auto definition = std::make_unique<IndexedSubGraph::MetaDef>();
  definition->name = "CompiledProjection";
  definition->domain = "test.partition";
  definition->since_version = 1;
  for (const auto* input : node.InputDefs()) {
    definition->inputs.push_back(input->Name());
  }
  definition->outputs.push_back(model_.MainGraph().GetOutputs()[0]->Name());
  for (const auto& original : model_.MainGraph().Nodes()) {
    partition.nodes.push_back(original.Index());
  }
  partition.SetMetaDef(std::move(definition));
  model_.MainGraph().FuseSubGraph(partition, "compiled_projection");
  const auto report = Report();
  ASSERT_EQ(report["nodes"].size(), 1u);
  EXPECT_EQ(report["nodes"][0]["unavailable_reason"], "opaque_partition");
  EXPECT_TRUE(report["nodes"][0]["components"].empty());
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "0");
}

TEST_F(ComputeEstimateReportTest, SubtotalOverflowIsExplicit) {
  constexpr int64_t m = (int64_t{1} << 61) - 1;
  // Each count is 2 * ((2^61) - 1); add enough nodes to exceed uint64 without any node overflowing.
  for (int i = 0; i < 5; ++i) {
    auto& node = AddMatMul(std::vector<int64_t>{m, 1});
    node.AddAttribute("K", int64_t{1});
    node.AddAttribute("N", int64_t{1});
  }
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  EXPECT_EQ(report["summary"]["estimated_node_count"], 5);
  EXPECT_EQ(report["summary"]["subtotal_unavailable_reason"], "overflow");
  EXPECT_FALSE(report["summary"].contains("known_dense_equivalent_ops"));
}

TEST_F(ComputeEstimateReportTest, NestedGraphsAreSeparateScopes) {
  auto* condition = builder_.MakeInput<bool>(std::vector<int64_t>{});
  auto* captured = builder_.MakeInput<float>(std::vector<int64_t>{2, 3, 64});
  model_.MainGraph().SetInputs({condition, captured});
  auto* output = builder_.MakeOutput();
  auto& node = builder_.AddNode("If", {condition}, {output});
  for (const auto* attribute : {"then_branch", "else_branch"}) {
    Model branch("branch", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                 {{kOnnxDomain, 17}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger());
    ModelTestBuilder branch_builder(branch.MainGraph());
    auto* input = branch_builder.MakeInput<float>(std::vector<int64_t>{2, 3, 64}, captured->Name());
    auto* weights = branch_builder.MakeInitializer<uint8_t>({32, 2, 16}, std::vector<uint8_t>(1024, 0x88));
    auto* scales = branch_builder.MakeInitializer<float>({32, 2}, std::vector<float>(64, 1.0f));
    auto& matmul = branch_builder.AddNode("MatMulNBits", {input, weights, scales},
                                          {branch_builder.MakeOutput()}, kMSDomain);
    matmul.AddAttribute("K", int64_t{64});
    matmul.AddAttribute("N", int64_t{32});
    matmul.AddAttribute("bits", int64_t{4});
    matmul.AddAttribute("block_size", int64_t{32});
    branch_builder.SetGraphOutputs();
    ASSERT_STATUS_OK(branch.MainGraph().Resolve());
    auto proto = branch.ToProto().graph();
    proto.clear_input();
    node.AddAttribute(attribute, proto);
  }
  builder_.SetGraphOutputs();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  ASSERT_EQ(report["graphs"].size(), 3u);
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "0");
  EXPECT_EQ(report["summary"]["contains_subgraphs"], true);
  EXPECT_EQ(report["nodes"][0]["unavailable_reason"], "runtime_execution_required");
  EXPECT_EQ(report["graphs"][1]["graph_path"], "main/0/else_branch");
  EXPECT_EQ(report["graphs"][2]["graph_path"], "main/0/then_branch");
  for (size_t i = 1; i < 3; ++i) {
    EXPECT_EQ(report["graphs"][i]["summary"]["known_dense_equivalent_ops"], "24576");
    EXPECT_EQ(report["graphs"][i]["summary"]["per_run_total_available"], false);
  }
  auto options = ReportOptions();
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  const auto exported = ReadReport();
  EXPECT_EQ(exported["graphs"].size(), 3u);
  EXPECT_EQ(exported["summary"]["known_dense_equivalent_ops"], "0");
  RecordProperty("compute_estimate_report", exported.dump(2));
}

TEST_F(ComputeEstimateReportTest, SessionExportsFinalGraphAndRetainsInference) {
  AddMatMul(std::vector<int64_t>{2, 3, 64}, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  auto options = ReportOptions();
  options.graph_optimization_level = TransformerLevel::Level1;
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  const auto report = ReadReport();
  EXPECT_EQ(report["nodes"].size(), session.GetGraph().NumberOfNodes());
  EXPECT_EQ(report["summary"]["reported_node_count"], 1);
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "24576");
  EXPECT_EQ(report["nodes"][0]["provider"], kCpuExecutionProvider);
  RecordProperty("compute_estimate_report", report.dump(2));

  const auto& input_name = session.GetGraph().GetInputs()[0]->Name();
  const auto& output_name = session.GetGraph().GetOutputs()[0]->Name();
  std::vector<float> values(2 * 3 * 64, 1.0f);
  OrtValue input;
  auto allocator = std::make_shared<CPUAllocator>();
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({2, 3, 64}), allocator, input);
  std::copy(values.begin(), values.end(), input.GetMutable<Tensor>()->MutableData<float>());
  std::vector<OrtValue> outputs;
  const std::vector<std::string> output_names{output_name};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, {{input_name, input}}, output_names, &outputs));
  ASSERT_EQ(outputs.size(), 1u);
  const auto& result = outputs[0].Get<Tensor>();
  EXPECT_EQ(result.Shape(), TensorShape({2, 3, 32}));
  for (size_t i = 0; i < 2 * 3 * 32; ++i) {
    EXPECT_EQ(result.Data<float>()[i], 0.0f);
  }
}

TEST_F(ComputeEstimateReportTest, DisabledExportDoesNotWrite) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  SessionOptions options;
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  EXPECT_FALSE(std::filesystem::exists(path_));
}

TEST_F(ComputeEstimateReportTest, WriteFailureWarnsAndInitializationSucceeds) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  auto options = ReportOptions();
  // Opening an existing directory as a file fails without requiring permission changes.
  const auto invalid_path = ToUTF8String(std::filesystem::temp_directory_path().native());
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsComputeEstimateReportPath,
                                                         invalid_path.c_str()));
  auto sink = std::make_unique<CapturingSink>();
  auto* captured = sink.get();
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::move(sink), logging::Severity::kWARNING, false, logging::LoggingManager::InstanceType::Temporal);
  std::unique_ptr<Environment> environment;
  ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), environment));
  InferenceSession session(options, *environment);
  ASSERT_STATUS_OK(Initialize(session));
  EXPECT_TRUE(std::any_of(captured->Messages().begin(), captured->Messages().end(), [](const std::string& message) {
    return message.find("Failed to open compute estimate report") != std::string::npos;
  }));
}

TEST_F(ComputeEstimateReportTest, Utf8PathAndOverwrite) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto unicode_path = path_.parent_path() /
                            ToPathString(path_.stem().string() + "_\xE6\xB5\x8B\xE8\xAF\x95.json");
  auto cleanup = gsl::finally([&]() { std::filesystem::remove(unicode_path); });
  const auto utf8_path = ToUTF8String(unicode_path.native());
  ASSERT_STATUS_OK(WriteComputeEstimateReport(model_.MainGraph(), utf8_path));
  ASSERT_STATUS_OK(WriteComputeEstimateReport(model_.MainGraph(), utf8_path));
  std::ifstream stream(unicode_path);
  const auto report = Json::parse(stream);
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "24576");
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
#endif
