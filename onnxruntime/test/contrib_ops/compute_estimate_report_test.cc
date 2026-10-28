// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include <gsl/gsl>
#include "nlohmann/json.hpp"
#include "core/common/path_string.h"
#include "core/framework/allocator.h"
#include "core/framework/compute_estimate_report.h"
#include "core/framework/runtime_workload_collector.h"
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

  ~ComputeEstimateReportTest() override {
    for (const auto& path : RuntimeReportPaths()) {
      std::error_code error;
      std::filesystem::remove(path, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  std::vector<std::filesystem::path> RuntimeReportPaths() const {
    std::vector<std::filesystem::path> paths;
    const auto prefix = path_.filename().native() + ToPathString("_");
    for (const auto& entry : std::filesystem::directory_iterator(path_.parent_path())) {
      if (entry.path().filename().native().find(prefix) == 0 && entry.path().extension() == ".json") {
        paths.push_back(entry.path());
      }
    }
    return paths;
  }

  Node& AddMatMul(std::optional<std::vector<int64_t>> shape = std::vector<int64_t>{2, 3, 64},
                  bool add_identity = false, bool bias = false) {
    auto* a = builder_.MakeInput<float>(shape);
    auto* b = builder_.MakeInitializer<uint8_t>({32, 2, 16}, std::vector<uint8_t>(1024, 0x88));
    auto* scales = builder_.MakeInitializer<float>({32, 2}, std::vector<float>(64, 1.0f));
    auto* output = add_identity ? builder_.MakeIntermediate() : builder_.MakeOutput();
    std::vector<NodeArg*> inputs{a, b, scales};
    if (bias) {
      auto& missing = builder_.graph_.GetOrCreateNodeArg("", nullptr);
      inputs.push_back(&missing);
      inputs.push_back(&missing);
      inputs.push_back(builder_.MakeInitializer<float>({32}, std::vector<float>(32, 1.0f)));
    }
    auto& node = builder_.graph_.AddNode(builder_.graph_.GenerateNodeName("projection \"quoted\"\\\n"),
                                         "MatMulNBits", "", inputs, {output}, nullptr, kMSDomain);
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
    const auto paths = RuntimeReportPaths();
    ORT_ENFORCE(paths.size() == 1, "Expected exactly one runtime report, found ", paths.size());
    std::ifstream stream(paths[0]);
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
    options.intra_op_param.thread_pool_size = 1;
    const auto utf8_path = ToUTF8String(path_.native());
    EXPECT_TRUE(options.config_options.AddConfigEntry(kOrtSessionOptionsComputeEstimateReportPath,
                                                      utf8_path.c_str())
                    .IsOK());
    return options;
  }

  Status Run(InferenceSession& session, const std::vector<int64_t>& shape = {2, 3, 64},
             bool terminate = false, RuntimeWorkloadCollector* collector = nullptr) {
    OrtValue input;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape(shape), std::make_shared<CPUAllocator>(), input);
    std::fill_n(input.GetMutable<Tensor>()->MutableData<float>(),
                static_cast<size_t>(input.Get<Tensor>().Shape().Size()), 1.0f);
    RunOptions options;
    options.terminate = terminate;
    options.runtime_workload_collector = collector;
    std::vector<OrtValue> outputs;
    const std::vector<std::string> output_names{model_.MainGraph().GetOutputs()[0]->Name()};
    return session.Run(options, {{model_.MainGraph().GetInputs()[0]->Name(), input}},
                       output_names, &outputs);
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
  EXPECT_EQ(report["schema_version"], 2);
  EXPECT_EQ(report["estimator_support"], "matmul_nbits_core_and_bias");
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
  ASSERT_EQ(entry["components"].size(), 2u);
  const auto& bias = entry["components"][1];
  EXPECT_EQ(bias["component"], "bias");
  EXPECT_EQ(bias["metric"], "floating_point_ops");
  EXPECT_EQ(bias["operation_count"], "0");
  EXPECT_EQ(bias["input_basis"], "optional_input_absent");
  EXPECT_EQ(bias["convention_version"], "bias_add_v1");
  EXPECT_EQ(report, Report());
}

TEST_F(ComputeEstimateReportTest, BiasIsSeparateFromCoreSubtotal) {
  AddMatMul(std::vector<int64_t>{2, 3, 64}, false, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  const auto& bias = report["nodes"][0]["components"][1];
  EXPECT_EQ(bias["operation_count"], "192");
  EXPECT_EQ(bias["input_basis"], "static_concrete");
  EXPECT_EQ(bias["basis"], "logical_semantics");
  EXPECT_EQ(bias["execution_phase"], "per_invocation");
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "24576");
  const auto& summary = report["summary"]["component_summaries"][1];
  EXPECT_EQ(summary["component"], "bias");
  EXPECT_EQ(summary["known_operation_count"], "192");
  EXPECT_EQ(summary["estimated_component_count"], 1);
  EXPECT_EQ(summary["unavailable_component_count"], 0);
  EXPECT_EQ(summary["complete_for_declared_component_scope"], true);
  EXPECT_EQ(report["summary"]["whole_operator_work_complete"], false);
}

TEST_F(ComputeEstimateReportTest, UnknownBiasGeometryHasExplicitCoverageGap) {
  AddMatMul(std::vector<int64_t>{-1, 64}, false, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  const auto& entry = report["nodes"][0];
  EXPECT_EQ(entry["availability"], "unavailable");
  EXPECT_EQ(entry["components"][1]["unavailable_reason"], "unknown_shape");
  EXPECT_FALSE(entry["components"][1].contains("operation_count"));
  const auto& summary = report["summary"]["component_summaries"][1];
  EXPECT_EQ(summary["known_operation_count"], "0");
  EXPECT_EQ(summary["unavailable_component_count"], 1);
  EXPECT_EQ(summary["unavailable_reason_counts"]["unknown_shape"], 1);
  EXPECT_EQ(summary["complete_for_declared_component_scope"], false);
}

TEST_F(ComputeEstimateReportTest, EmptyOutputHasZeroBias) {
  AddMatMul(std::vector<int64_t>{0, 64}, false, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  EXPECT_EQ(report["nodes"][0]["components"][1]["operation_count"], "0");
  EXPECT_EQ(report["summary"]["complete_for_declared_graph_and_component_scope"], true);
}

TEST_F(ComputeEstimateReportTest, CoreOverflowRetainsKnownBias) {
  auto& node = AddMatMul(std::vector<int64_t>{1, std::numeric_limits<int64_t>::max()}, false, true);
  node.AddAttribute("K", std::numeric_limits<int64_t>::max());
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  EXPECT_EQ(report["nodes"][0]["availability"], "partial");
  EXPECT_EQ(report["nodes"][0]["components"][0]["unavailable_reason"], "overflow");
  EXPECT_EQ(report["nodes"][0]["components"][1]["operation_count"], "32");
  EXPECT_EQ(report["summary"]["partially_estimated_node_count"], 1);
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "0");
  EXPECT_EQ(report["summary"]["component_summaries"][1]["known_operation_count"], "32");
  EXPECT_EQ(report["summary"]["complete_for_declared_graph_and_component_scope"], false);
}

TEST_F(ComputeEstimateReportTest, BiasSubtotalOverflowIsSeparateAndExplicit) {
  for (int i = 0; i < 3; ++i) {
    auto& node = AddMatMul(std::vector<int64_t>{std::numeric_limits<int64_t>::max() / 32, 0}, false, true);
    node.AddAttribute("K", int64_t{0});
  }
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "0");
  const auto& summary = report["summary"]["component_summaries"][1];
  EXPECT_EQ(summary["subtotal_availability"], "unavailable");
  EXPECT_EQ(summary["subtotal_unavailable_reason"], "overflow");
  EXPECT_FALSE(summary.contains("known_operation_count"));
  EXPECT_EQ(report["summary"]["complete_for_declared_graph_and_component_scope"], false);
}

TEST_F(ComputeEstimateReportTest, KnownCoreDoesNotHideBiasOverflow) {
  auto& node = AddMatMul(std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 0}, false, true);
  node.AddAttribute("K", int64_t{0});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  const auto& entry = report["nodes"][0];
  EXPECT_EQ(entry["availability"], "partial");
  EXPECT_EQ(entry["unavailable_reason"], "overflow");
  EXPECT_EQ(entry["components"][0]["operation_count"], "0");
  EXPECT_EQ(entry["components"][1]["unavailable_reason"], "overflow");
  EXPECT_FALSE(entry["components"][1].contains("operation_count"));
  EXPECT_EQ(report["summary"]["known_dense_equivalent_ops"], "0");
  EXPECT_EQ(report["summary"]["component_summaries"][1]["unavailable_component_count"], 1);
  EXPECT_EQ(report["summary"]["complete_for_declared_graph_and_component_scope"], false);
}

TEST_F(ComputeEstimateReportTest, UnknownShapesHaveNoCount) {
  AddMatMul(std::vector<int64_t>{-1, 64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  const auto report = Report();
  const auto& entry = report["nodes"][0];
  EXPECT_EQ(entry["unavailable_reason"], "unknown_shape");
  EXPECT_EQ(entry["availability"], "partial");
  EXPECT_EQ(entry["components"][1]["operation_count"], "0");
  EXPECT_EQ(report["summary"]["partially_estimated_node_count"], 1);
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
}

TEST_F(ComputeEstimateReportTest, SessionExportsExecutedInvocationsAndRetainsInference) {
  AddMatMul(std::vector<int64_t>{2, 3, 64}, true, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  auto options = ReportOptions();
  options.graph_optimization_level = TransformerLevel::Level1;
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  EXPECT_FALSE(std::filesystem::exists(path_));
  EXPECT_TRUE(RuntimeReportPaths().empty());

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
    EXPECT_EQ(result.Data<float>()[i], 1.0f);
  }
  const auto report = ReadReport();
  EXPECT_EQ(report["schema_version"], 3);
  EXPECT_EQ(report["run_state"], "succeeded");
  EXPECT_EQ(report["session_id"], session.GetCurrentSessionId());
  EXPECT_EQ(report["collection"]["max_records"], 10000);
  EXPECT_EQ(report["collection"]["max_payload_bytes"], 16777216);
  ASSERT_EQ(report["invocations"].size(), 1u);
  EXPECT_EQ(report["invocations"][0]["provider"], kCpuExecutionProvider);
  EXPECT_EQ(report["invocations"][0]["input_shape"], Json::array({2, 3, 64}));
  EXPECT_EQ(report["invocations"][0]["components"][0]["operation_count"], "24576");
  EXPECT_EQ(report["invocations"][0]["components"][1]["operation_count"], "192");
  EXPECT_TRUE(report["invocations"][0].contains("route"));
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], true);
  EXPECT_EQ(report["whole_run_work_complete"], false);
  RecordProperty("runtime_compute_estimate_report", report.dump(2));
}

TEST_F(ComputeEstimateReportTest, DisabledExportDoesNotWrite) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  SessionOptions options;
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  ASSERT_STATUS_OK(Run(session));
  EXPECT_FALSE(std::filesystem::exists(path_));
  EXPECT_TRUE(RuntimeReportPaths().empty());
}

TEST_F(ComputeEstimateReportTest, WriteFailureWarnsAndInferenceSucceeds) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  auto options = ReportOptions();
  // A file cannot be the parent directory of a report.
  {
    std::ofstream parent(path_);
  }
  const auto invalid_path = ToUTF8String((path_ / "report").native());
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
  EXPECT_FALSE(std::any_of(captured->Messages().begin(), captured->Messages().end(), [](const std::string& message) {
    return message.find("Failed to open compute estimate report") != std::string::npos;
  }));
  ASSERT_STATUS_OK(Run(session));
  EXPECT_TRUE(std::any_of(captured->Messages().begin(), captured->Messages().end(), [](const std::string& message) {
    return message.find("Failed to open compute estimate report") != std::string::npos;
  }));
}

TEST_F(ComputeEstimateReportTest, DynamicAndConcurrentRunsWriteDistinctReports) {
  AddMatMul(std::vector<int64_t>{-1, 3, 64}, false, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  InferenceSession session(ReportOptions(), GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  std::atomic<size_t> failures{0};
  std::array<std::thread, 4> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i]() {
      if (!Run(session, {static_cast<int64_t>(i + 1), 3, 64}).IsOK()) ++failures;
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(failures, 0u);
  const auto paths = RuntimeReportPaths();
  ASSERT_EQ(paths.size(), threads.size());
  std::vector<uint64_t> ids;
  for (const auto& path : paths) {
    std::ifstream stream(path);
    const auto report = Json::parse(stream);
    const auto batches = report["invocations"][0]["input_shape"][0].get<uint64_t>();
    EXPECT_EQ(report["component_summaries"][0]["known_operation_count"], std::to_string(batches * 12288));
    EXPECT_EQ(report["component_summaries"][1]["known_operation_count"], std::to_string(batches * 96));
    ids.push_back(report["run_id"].get<uint64_t>());
  }
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(std::adjacent_find(ids.begin(), ids.end()), ids.end());
  ASSERT_STATUS_OK(Run(session));
  EXPECT_EQ(RuntimeReportPaths().size(), threads.size() + 1);
  InferenceSession another_session(ReportOptions(), GetEnvironment());
  ASSERT_STATUS_OK(Initialize(another_session));
  ASSERT_STATUS_OK(Run(another_session));
  EXPECT_EQ(RuntimeReportPaths().size(), threads.size() + 2);
}

TEST_F(ComputeEstimateReportTest, AutomaticLimitsAndFailedRunsAreExplicit) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  for (const auto* limit : {kOrtSessionOptionsComputeEstimateMaxRecords,
                            kOrtSessionOptionsComputeEstimateMaxPayloadBytes}) {
    auto options = ReportOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(limit, "0"));
    InferenceSession session(options, GetEnvironment());
    ASSERT_STATUS_OK(Initialize(session));
    ASSERT_STATUS_OK(Run(session));
    auto report = ReadReport();
    EXPECT_TRUE(report["invocations"].empty());
    EXPECT_EQ(report["run_state"], "succeeded");
    EXPECT_EQ(report["collection"]["collection_complete"], false);
    EXPECT_EQ(report["collection"][limit == kOrtSessionOptionsComputeEstimateMaxRecords
                                       ? "record_limit_drops"
                                       : "payload_limit_drops"],
              1);
    EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], false);
    for (const auto& path : RuntimeReportPaths()) ASSERT_TRUE(std::filesystem::remove(path));
  }
  InferenceSession session(ReportOptions(), GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  EXPECT_FALSE(Run(session, {2, 3, 64}, true).IsOK());
  const auto report = ReadReport();
  EXPECT_EQ(report["run_state"], "failed");
  EXPECT_TRUE(report["invocations"].empty());
  EXPECT_EQ(report["collection"]["collection_complete"], true);
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], false);
}

TEST_F(ComputeEstimateReportTest, InvalidLimitsFailInitialization) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  for (const auto* key : {kOrtSessionOptionsComputeEstimateMaxRecords,
                          kOrtSessionOptionsComputeEstimateMaxPayloadBytes}) {
    for (const auto* value : {"", "-1", "1x", " 2", "18446744073709551616"}) {
      auto options = ReportOptions();
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(key, value));
      InferenceSession session(options, GetEnvironment());
      EXPECT_FALSE(Initialize(session).IsOK()) << key << "=" << value;
    }
  }
  EXPECT_TRUE(RuntimeReportPaths().empty());
}

TEST_F(ComputeEstimateReportTest, ExplicitCollectorIsExportedWithoutChangingItsLimits) {
  AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  auto options = ReportOptions();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsComputeEstimateMaxRecords, "0"));
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  RuntimeWorkloadCollector collector({10, 8192});
  ASSERT_STATUS_OK(Run(session, {2, 3, 64}, false, &collector));
  const auto report = ReadReport();
  EXPECT_EQ(report["invocations"].size(), 1u);
  EXPECT_EQ(report["collection"]["max_records"], 10);
  EXPECT_EQ(report["collection"]["max_payload_bytes"], 8192);
  ASSERT_STATUS_OK(Run(session, {2, 3, 64}, false, &collector));
  EXPECT_EQ(RuntimeReportPaths().size(), 1u);
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
  path_ = unicode_path;
  InferenceSession session(ReportOptions(), GetEnvironment());
  ASSERT_STATUS_OK(Initialize(session));
  EXPECT_TRUE(RuntimeReportPaths().empty());
  ASSERT_STATUS_OK(Run(session));
  const auto first = ReadReport();
  EXPECT_EQ(first["component_summaries"][0]["known_operation_count"], "24576");
  ASSERT_STATUS_OK(Run(session));
  EXPECT_EQ(RuntimeReportPaths().size(), 2u);
}

RuntimeWorkloadSnapshot RuntimeSnapshot() {
  RuntimeWorkloadSnapshot snapshot;
  snapshot.session_id = 7;
  snapshot.run_id = 11;
  snapshot.run_state = RuntimeWorkloadRunState::Succeeded;
  snapshot.limits = {10, 8192};
  MatMulNBitsRuntimeMetadata metadata;
  metadata.input_shape = {2, 3, 64};
  metadata.output_shape = {2, 3, 32};
  metadata.m = 6;
  metadata.n = 32;
  metadata.k = 64;
  metadata.batch_count = 1;
  metadata.bias_present = true;
  metadata.bits = 4;
  metadata.block_size = 32;
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT;
  snapshot.records.push_back({0, "main/0/body", 1, kCpuExecutionProvider, metadata});
  return snapshot;
}

Json RuntimeReport(const RuntimeWorkloadSnapshot& snapshot) {
  std::string text;
  ORT_THROW_IF_ERROR(BuildComputeEstimateReport(snapshot, text));
  return Json::parse(text);
}

TEST(RuntimeComputeEstimateReportTest, CountsAndRouteProvenanceAreIndependent) {
  auto snapshot = RuntimeSnapshot();
  for (const auto route : {MatMulNBitsRuntimeRoute::PackedLut, MatMulNBitsRuntimeRoute::PackedQNBit,
                           MatMulNBitsRuntimeRoute::Unpacked, MatMulNBitsRuntimeRoute::EmptyOutput}) {
    snapshot.records[0].matmul_nbits.route = route;
    const auto report = RuntimeReport(snapshot);
    EXPECT_EQ(report["invocations"][0]["components"][0]["operation_count"], "24576");
    EXPECT_EQ(report["invocations"][0]["components"][1]["operation_count"], "192");
    EXPECT_EQ(report["component_summaries"][0]["known_operation_count"], "24576");
    EXPECT_EQ(report["route_adjustment_applied"], false);
    EXPECT_EQ(report, RuntimeReport(snapshot));
  }
  snapshot.records[0].matmul_nbits.route.reset();
  const auto report = RuntimeReport(snapshot);
  EXPECT_FALSE(report["invocations"][0].contains("route"));
  EXPECT_FALSE(report["invocations"][0].contains("compute_mode"));
  EXPECT_EQ(report["invocations"][0]["availability"], "available");
}

TEST(RuntimeComputeEstimateReportTest, SubgraphsAndRepeatedInvocationsSumOnlyExecutedRecords) {
  auto snapshot = RuntimeSnapshot();
  auto record = snapshot.records[0];
  record.invocation_id = 1;
  record.graph_path = "main/2/then_branch";
  snapshot.records.push_back(record);
  record.invocation_id = 2;
  snapshot.records.push_back(record);
  const auto report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"].size(), 3u);
  EXPECT_EQ(report["component_summaries"][0]["known_operation_count"], "73728");
  EXPECT_EQ(report["component_summaries"][1]["known_operation_count"], "576");
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], true);
}

TEST(RuntimeComputeEstimateReportTest, UnknownInvalidAndUnsupportedAreNotZero) {
  auto snapshot = RuntimeSnapshot();
  snapshot.records[0].matmul_nbits.input_shape = {-1, 3, 64};
  auto report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"][0]["availability"], "unavailable");
  EXPECT_EQ(report["invocations"][0]["components"][0]["unavailable_reason"], "unknown_shape");
  EXPECT_FALSE(report["invocations"][0]["components"][0].contains("operation_count"));
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], false);
  snapshot.records[0].matmul_nbits.input_shape = {2, 3, 63};
  report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"][0]["components"][0]["unavailable_reason"], "invalid_metadata");
  snapshot.records[0].provider = kCudaExecutionProvider;
  report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"][0]["components"][0]["unavailable_reason"], "unsupported_provider");
}

TEST(RuntimeComputeEstimateReportTest, EmptyOutputAndZeroReductionKeepBiasSemantics) {
  auto snapshot = RuntimeSnapshot();
  snapshot.records[0].matmul_nbits.input_shape = {2, 0, 64};
  auto report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"][0]["components"][0]["operation_count"], "0");
  EXPECT_EQ(report["invocations"][0]["components"][1]["operation_count"], "0");
  snapshot.records[0].matmul_nbits.input_shape = {2, 3, 0};
  snapshot.records[0].matmul_nbits.k = 0;
  report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"][0]["components"][0]["operation_count"], "0");
  EXPECT_EQ(report["invocations"][0]["components"][1]["operation_count"], "192");
  snapshot.records[0].matmul_nbits.bias_present = false;
  EXPECT_EQ(RuntimeReport(snapshot)["invocations"][0]["components"][1]["operation_count"], "0");
}

TEST(RuntimeComputeEstimateReportTest, InvocationAndSubtotalOverflowRemainDistinct) {
  auto snapshot = RuntimeSnapshot();
  auto& metadata = snapshot.records[0].matmul_nbits;
  metadata.input_shape = {std::numeric_limits<int64_t>::max(), 1};
  metadata.k = 1;
  metadata.n = 2;
  auto report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"][0]["components"][0]["unavailable_reason"], "overflow");
  EXPECT_EQ(report["invocations"][0]["components"][1]["operation_count"], "18446744073709551614");
  EXPECT_EQ(report["invocations"][0]["availability"], "partial");
  metadata.n = 1;
  auto record = snapshot.records[0];
  record.invocation_id = 1;
  snapshot.records.push_back(record);
  report = RuntimeReport(snapshot);
  EXPECT_EQ(report["invocations"][0]["components"][0]["operation_count"], "18446744073709551614");
  EXPECT_EQ(report["component_summaries"][0]["subtotal_unavailable_reason"], "overflow");
  EXPECT_FALSE(report["component_summaries"][0].contains("known_operation_count"));
  EXPECT_EQ(report["component_summaries"][1]["known_operation_count"], "18446744073709551614");
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], false);
}

TEST(RuntimeComputeEstimateReportTest, FailedRunningAndIncompleteDoNotClaimCompleteTotals) {
  auto snapshot = RuntimeSnapshot();
  for (const auto state : {RuntimeWorkloadRunState::Failed, RuntimeWorkloadRunState::Running,
                           RuntimeWorkloadRunState::NotStarted}) {
    snapshot.run_state = state;
    const auto report = RuntimeReport(snapshot);
    EXPECT_EQ(report["component_summaries"][0]["known_operation_count"], "24576");
    EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], false);
  }
  snapshot.run_state = RuntimeWorkloadRunState::Succeeded;
  snapshot.stats.collection_complete = false;
  snapshot.stats.record_limit_drops = 3;
  snapshot.stats.payload_limit_drops = 4;
  snapshot.stats.metadata_failure_drops = 5;
  snapshot.stats.allocation_failure_drops = 6;
  snapshot.stats.run_reuse_rejections = 7;
  const auto report = RuntimeReport(snapshot);
  EXPECT_EQ(report["collection"]["record_limit_drops"], 3);
  EXPECT_EQ(report["collection"]["payload_limit_drops"], 4);
  EXPECT_EQ(report["collection"]["metadata_failure_drops"], 5);
  EXPECT_EQ(report["collection"]["allocation_failure_drops"], 6);
  EXPECT_EQ(report["collection"]["run_reuse_rejections"], 7);
  EXPECT_EQ(report["component_summaries"][0]["complete_for_captured_component_scope"], false);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
#endif
