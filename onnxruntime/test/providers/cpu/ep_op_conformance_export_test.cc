#include <algorithm>
#include <atomic>
#include <filesystem>
#include <random>

#include "gtest/gtest.h"
#include "core/platform/env.h"
#include "test/ep_op_conformance/runner/case_io.h"
#include "test/unittest_util/op_tester.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

constexpr const char* kMulId = "ai.onnx.Mul.opset14.float32.vector";

struct MulCase {
  const std::vector<int64_t> shape{2};
  const std::vector<float> left{2.0f, 3.0f};
  const std::vector<float> right{4.0f, 5.0f};
  const std::vector<float> expected{8.0f, 15.0f};
  const float absolute = 1e-6f;
  const float relative = 2e-6f;

  void AddTo(OpTester& tester, bool right_is_initializer = false, bool sorted = false) const {
    const BaseTester::DimsVariant dimensions = shape;
    tester.AddInput<float>("A", dimensions, left);
    tester.AddInput<float>("B", dimensions, right, right_is_initializer);
    tester.AddOutput<float>("Y", dimensions, expected, sorted, relative, absolute);
  }
};

void RunOnCpu(OpTester& tester) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

class MissingTargetTester : public OpTester {
 public:
  MissingTargetTester() : OpTester("Mul", 14) {}

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>& attributes) override {
    OpTester::AddNodes(graph, inputs, outputs, attributes);
    auto& missing = graph.GetOrCreateNodeArg("missing_target", outputs.front()->TypeAsProto());
    graph.AddNode("extra_node", "Identity", "", {outputs.front()}, {&missing});
    graph.SetOutputs({outputs.front(), &missing});
  }
};

class EpOpConformanceExportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<unsigned> sequence{0};
    root_ = std::filesystem::temp_directory_path() /
            ("ort_ep_export_" + std::to_string(std::random_device{}()) + "_" + std::to_string(sequence++));
    ASSERT_TRUE(std::filesystem::create_directory(root_));
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
    EXPECT_FALSE(error) << error.message();
  }

  std::filesystem::path CaseDirectory() const { return root_ / "case"; }

  void ExpectNoExport() const {
    EXPECT_FALSE(std::filesystem::exists(CaseDirectory() / "case.json"));
    EXPECT_FALSE(std::filesystem::exists(CaseDirectory()));
  }

  std::filesystem::path root_;
  const MulCase authored_;
};

TEST_F(EpOpConformanceExportTest, OrdinaryRunWithoutExport) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  ASSERT_NO_THROW(RunOnCpu(tester));
  ASSERT_EQ(tester.GetFetches().size(), 1U);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, ExplicitOptionsWithoutExport) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  SessionOptions options;
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  ASSERT_NO_THROW(tester.Run(options, BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers));
  ASSERT_EQ(tester.GetFetches().size(), 1U);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, MulFloat32Vector) {
  const auto configured_root = Env::Default().GetEnvironmentVar("ORT_EP_OP_CONFORMANCE_EXPORT_DIR");
  const auto directory = configured_root.empty() ? CaseDirectory() : std::filesystem::path(configured_root) / kMulId;
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(directory.string(), kMulId);
  ASSERT_NO_THROW(RunOnCpu(tester));
  const auto fetches = tester.GetFetches();
  ASSERT_EQ(fetches.size(), 1U);
  const auto& fetched = fetches.front().Get<Tensor>();
  ASSERT_EQ(fetched.Shape().Size(), static_cast<int64_t>(authored_.expected.size()));
  EXPECT_TRUE(std::equal(authored_.expected.begin(), authored_.expected.end(), fetched.Data<float>()));

  const auto exported = ep_op_conformance::ReadCase(directory);
  EXPECT_EQ(exported.manifest.at("id"), kMulId);
  EXPECT_EQ(exported.manifest.at("schema_version"), 1);
  EXPECT_FALSE(exported.manifest.contains("repeat_count"));
  ASSERT_EQ(exported.datasets.size(), 1U);
  const auto& dataset = exported.datasets.front();
  ASSERT_EQ(dataset.inputs.size(), 2U);
  EXPECT_EQ(dataset.inputs[0].name, "A");
  EXPECT_EQ(dataset.inputs[0].shape, authored_.shape);
  EXPECT_EQ(dataset.inputs[0].values, authored_.left);
  EXPECT_EQ(dataset.inputs[1].name, "B");
  EXPECT_EQ(dataset.inputs[1].shape, authored_.shape);
  EXPECT_EQ(dataset.inputs[1].values, authored_.right);
  ASSERT_EQ(dataset.outputs.size(), 1U);
  EXPECT_EQ(dataset.outputs.front().name, "Y");
  EXPECT_EQ(dataset.outputs.front().shape, authored_.shape);
  EXPECT_EQ(dataset.outputs.front().values, authored_.expected);
  const auto policy = ep_op_conformance::ReadComparison(exported.manifest.at("comparison").at("Y"));
  EXPECT_DOUBLE_EQ(policy.absolute, authored_.absolute);
  EXPECT_DOUBLE_EQ(policy.relative, authored_.relative);
  EXPECT_TRUE(policy.nan_equal);
  ONNX_NAMESPACE::ModelProto model;
  ASSERT_TRUE(model.ParseFromString(exported.model));
  ASSERT_EQ(model.graph().node_size(), 1);
  EXPECT_EQ(model.graph().node(0).op_type(), "Mul");
  EXPECT_EQ(exported.manifest.at("required_nodes"),
            ep_op_conformance::Json::array({model.graph().node(0).name()}));
  bool has_opset = false;
  for (const auto& import : model.opset_import()) {
    if (import.domain().empty()) {
      EXPECT_EQ(import.version(), 14);
      has_opset = true;
    }
  }
  EXPECT_TRUE(has_opset);
  const auto& source = exported.manifest.at("source");
  const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  EXPECT_EQ(source.at("suite"), test_info->test_suite_name());
  EXPECT_EQ(source.at("test"), test_info->name());
  EXPECT_EQ(source.at("line"), test_info->line());
  const auto source_path = source.at("file").get<std::string>();
  EXPECT_FALSE(std::filesystem::path(source_path).is_absolute());
  EXPECT_EQ(source_path, "onnxruntime/test/providers/cpu/ep_op_conformance_export_test.cc");
}

TEST_F(EpOpConformanceExportTest, RejectsMultipleRuns) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.SetNumRunCalls(2);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, InitializerRemainsInGraph) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester, true);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  ASSERT_NO_THROW(RunOnCpu(tester));
  const auto exported = ep_op_conformance::ReadCase(CaseDirectory());
  ASSERT_EQ(exported.datasets.front().inputs.size(), 1U);
  EXPECT_EQ(exported.datasets.front().inputs.front().name, "A");
  ONNX_NAMESPACE::ModelProto model;
  ASSERT_TRUE(model.ParseFromString(exported.model));
  ASSERT_EQ(model.graph().initializer_size(), 1);
  const auto& initializer = model.graph().initializer(0);
  EXPECT_EQ(initializer.name(), "B");
  const auto path = root_ / "initializer.pb";
  ep_op_conformance::WriteFile(path, initializer.SerializeAsString());
  const auto tensor = ep_op_conformance::ReadTensor(path);
  EXPECT_EQ(tensor.shape, authored_.shape);
  EXPECT_EQ(tensor.values, authored_.right);
}

TEST_F(EpOpConformanceExportTest, CapturesOnlyOneInvocation) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  ASSERT_NO_THROW(RunOnCpu(tester));
  const auto first_manifest = ep_op_conformance::ReadFile(CaseDirectory() / "case.json");
  ASSERT_NO_THROW(RunOnCpu(tester));
  EXPECT_EQ(ep_op_conformance::ReadFile(CaseDirectory() / "case.json"), first_manifest);
}

TEST_F(EpOpConformanceExportTest, ExistingDirectoryIsNotOverwritten) {
  ASSERT_TRUE(std::filesystem::create_directory(CaseDirectory()));
  const std::string sentinel = "existing authored case";
  ep_op_conformance::WriteFile(CaseDirectory() / "case.json", sentinel);
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  EXPECT_EQ(ep_op_conformance::ReadFile(CaseDirectory() / "case.json"), sentinel);
  EXPECT_FALSE(std::filesystem::exists(CaseDirectory() / "model.onnx"));
}

TEST_F(EpOpConformanceExportTest, RejectsCustomVerifier) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsDisabledVerification) {
  OpTester tester("Mul", 14, kOnnxDomain, false);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsNonFloat32FeedAndOutput) {
  OpTester tester("Mul", 14);
  tester.AddInput<int32_t>("A", {1}, {1});
  tester.AddInput<int32_t>("B", {1}, {1});
  tester.AddOutput<int32_t>("Y", {1}, {1});
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsNonFloat32Initializer) {
  OpTester tester("Reshape", 14);
  tester.AddInput<float>("data", BaseTester::DimsVariant(authored_.shape), authored_.left);
  tester.AddInput<int64_t>("shape", {1}, authored_.shape, true);
  tester.AddOutput<float>("reshaped", BaseTester::DimsVariant(authored_.shape), authored_.left);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsNonFloat32Output) {
  OpTester tester("Cast", 13);
  tester.AddAttribute<int64_t>("to", ONNX_NAMESPACE::TensorProto::INT64);
  tester.AddInput<float>("input", BaseTester::DimsVariant(authored_.shape), authored_.left);
  std::vector<int64_t> expected;
  expected.reserve(authored_.left.size());
  for (const auto value : authored_.left) expected.push_back(static_cast<int64_t>(value));
  tester.AddOutput<int64_t>("output", BaseTester::DimsVariant(authored_.shape), expected);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsCustomRegistry) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.AddCustomOpRegistry(std::make_shared<CustomRegistry>());
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsMissingOptionalInput) {
  OpTester tester("Clip", 13);
  tester.AddInput<float>("input", BaseTester::DimsVariant(authored_.shape), authored_.left);
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<float>("max", {}, {10.0f});
  tester.AddOutput<float>("output", BaseTester::DimsVariant(authored_.shape), authored_.left);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsMissingOutputTarget) {
  MissingTargetTester tester;
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsSortedOutputs) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester, false, true);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsCustomConfigurationBeforeLegacyRun) {
  for (int configuration = 0; configuration != 3; ++configuration) {
    OpTester tester("Mul", 14);
    authored_.AddTo(tester);
    tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
    if (configuration == 0) tester.Config(SessionOptions{});
    if (configuration == 1) tester.Config(static_cast<const RunOptions*>(nullptr));
    if (configuration == 2) tester.Config(Graph::ResolveOptions{});
    EXPECT_THROW(RunOnCpu(tester), OnnxRuntimeException);
    ExpectNoExport();
  }
}

TEST_F(EpOpConformanceExportTest, RejectsDefaultOptimizedRunWithConfig) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId).ConfigEp(DefaultCpuExecutionProvider());
  EXPECT_THROW(tester.RunWithConfig(), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsExplicitSessionOptionsOverload) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  EXPECT_THROW(tester.Run(options), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsExpectedFailure) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(tester.Run(BaseTester::ExpectResult::kExpectFailure), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsLegacyRunOptions) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  RunOptions options;
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(tester.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, &options), OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsLegacyParallelExecution) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(tester.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, nullptr,
                          ExecutionMode::ORT_PARALLEL),
               OnnxRuntimeException);
  ExpectNoExport();
}

TEST_F(EpOpConformanceExportTest, RejectsLegacyResolveOptions) {
  OpTester tester("Mul", 14);
  authored_.AddTo(tester);
  Graph::ResolveOptions options;
  options.override_types = true;
  tester.ConfigEpOpConformanceExport(CaseDirectory().string(), kMulId);
  EXPECT_THROW(tester.Run(BaseTester::ExpectResult::kExpectSuccess, "", {}, nullptr, nullptr,
                          ExecutionMode::ORT_SEQUENTIAL, options),
               OnnxRuntimeException);
  ExpectNoExport();
}

}  // namespace
}  // namespace onnxruntime::test