// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/providers/migraphx/migraphx_provider_factory_creator.h"
#include "core/providers/migraphx/migraphx_execution_provider_utils.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {

namespace {

constexpr const char* kInt8EnableEnv = "ORT_MIGRAPHX_INT8_ENABLE";
constexpr const char* kInt8CalibrationTableNameEnv = "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME";
constexpr const char* kInt8UseNativeCalibrationTableEnv = "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE";
constexpr const char* kInt8EnableOption = "migraphx_int8_enable";
constexpr const char* kInt8CalibrationTableNameOption = "migraphx_int8_calibration_table_name";
constexpr const char* kInt8UseNativeCalibrationTableOption = "migraphx_int8_use_native_calibration_table";

std::unique_ptr<IExecutionProvider> CreateMIGraphXProvider(const ProviderOptions& options = {}) {
  return MIGraphXProviderFactoryCreator::Create(options)->CreateProvider();
}

}  // namespace

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with two inputs and one initializer.
 * input: "X", "Y" and "Z"
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Ini"  Add
 *     | \  /
 *     |  Add
 *     |      \
 *      \     Shape
 *       \    /
 *       Reshape
 *          |
 *          M
 */
void CreateBaseModel(onnxruntime::Model& model, std::vector<int> dims) {
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  for (auto dim : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  // INT tensor
  ONNX_NAMESPACE::TypeProto int64_tensor;
  int64_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  // constant
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  value_tensor.set_name("Ini");
  graph.AddInitializedTensor(value_tensor);

  // Create node1 (Add)
  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  // Create node2 (Add)
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Ini", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  // Create node3 (Shape)
  inputs.clear();
  outputs.clear();
  inputs.push_back(&output_arg_2);
  auto& output_arg_3 = graph.GetOrCreateNodeArg("S", &int64_tensor);
  outputs.push_back(&output_arg_3);
  graph.AddNode("node_3", "Shape", "node 3.", inputs, outputs);

  // Create node4 (Reshape)
  inputs.clear();
  outputs.clear();
  inputs.push_back(&input_arg_3);
  inputs.push_back(&output_arg_3);
  auto& output_arg_4 = graph.GetOrCreateNodeArg("R", &float_tensor);
  outputs.push_back(&output_arg_4);
  graph.AddNode("node_4", "Reshape", "node 4.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
}

TEST(MIGraphXExecutionProviderTest, GraphInputName) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  ASSERT_EQ(IsGraphInput(gv, "X"), true);
}

TEST(MIGraphXExecutionProviderTest, GraphInitializer) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  ASSERT_EQ(IsGraphInitializer(gv, "Ini"), true);
}

TEST(MIGraphXExecutionProviderTest, NodeInputNum) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node0 = gv.GetNode(0);
  const auto& node1 = gv.GetNode(1);

  ASSERT_EQ(getNodeInputNum(*node0), 0);
  ASSERT_EQ(getNodeInputNum(*node1), 1);
}

TEST(MIGraphXExecutionProviderTest, IsNodeInput) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node2 = gv.GetNode(1);
  ASSERT_EQ(isInputNode(node2, "M"), true);
}

TEST(MIGraphXExecutionProviderTest, canEvalArgument) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node2 = gv.GetNode(3);
  std::vector<NodeIndex> input_nodes;
  ASSERT_EQ(canEvalNodeArgument(gv, node2, {1}, input_nodes), true);
}

class MIGraphXInt8CalibrationTableTest : public testing::Test {
 protected:
  void SetUp() override {
    calibration_table_path_ =
        std::filesystem::path{testing::TempDir()} / "migraphx_int8_calibration_table_test.txt";
    std::ofstream calibration_table{calibration_table_path_};
    calibration_table << "TRT-8400-EntropyCalibration2\n"
                      << "input: 3f800000\n";
    ASSERT_TRUE(calibration_table.good());

    missing_calibration_table_path_ = calibration_table_path_;
    missing_calibration_table_path_ += ".missing";
    std::error_code ec;
    std::filesystem::remove(missing_calibration_table_path_, ec);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove(calibration_table_path_, ec);
  }

  EnvVarMap Int8Environment(optional<std::string> calibration_table_name) const {
    return {
        {kInt8EnableEnv, "1"},
        {kInt8CalibrationTableNameEnv, std::move(calibration_table_name)},
        {kInt8UseNativeCalibrationTableEnv, "1"},
    };
  }

  EnvVarMap ClearedInt8Environment() const {
    return {
        {kInt8EnableEnv, nullopt},
        {kInt8CalibrationTableNameEnv, nullopt},
        {kInt8UseNativeCalibrationTableEnv, nullopt},
    };
  }

  std::filesystem::path calibration_table_path_;
  std::filesystem::path missing_calibration_table_path_;
};

TEST_F(MIGraphXInt8CalibrationTableTest, EnvironmentTableIsEffective) {
  ScopedEnvironmentVariables environment{Int8Environment(calibration_table_path_.string())};

  const auto provider = CreateMIGraphXProvider();
  const auto options = provider->GetProviderOptions();

  EXPECT_EQ(options.at(kInt8CalibrationTableNameOption), calibration_table_path_.string());
}

TEST_F(MIGraphXInt8CalibrationTableTest, EnvironmentTableOverridesProviderOption) {
  ScopedEnvironmentVariables environment{Int8Environment(calibration_table_path_.string())};
  const ProviderOptions provider_options{
      {kInt8EnableOption, "1"},
      {kInt8CalibrationTableNameOption, missing_calibration_table_path_.string()},
      {kInt8UseNativeCalibrationTableOption, "1"},
  };

  const auto provider = CreateMIGraphXProvider(provider_options);
  const auto options = provider->GetProviderOptions();

  EXPECT_EQ(options.at(kInt8CalibrationTableNameOption), calibration_table_path_.string());
}

TEST_F(MIGraphXInt8CalibrationTableTest, ProviderOptionIsUsedWithoutEnvironmentTable) {
  ScopedEnvironmentVariables environment{ClearedInt8Environment()};
  const ProviderOptions provider_options{
      {kInt8EnableOption, "1"},
      {kInt8CalibrationTableNameOption, calibration_table_path_.string()},
      {kInt8UseNativeCalibrationTableOption, "1"},
  };

  const auto provider = CreateMIGraphXProvider(provider_options);
  const auto options = provider->GetProviderOptions();

  EXPECT_EQ(options.at(kInt8CalibrationTableNameOption), calibration_table_path_.string());
}

TEST_F(MIGraphXInt8CalibrationTableTest, EmptyEnvironmentTableFallsBackToProviderOption) {
  ScopedEnvironmentVariables environment{Int8Environment("")};
  const ProviderOptions provider_options{
      {kInt8EnableOption, "1"},
      {kInt8CalibrationTableNameOption, calibration_table_path_.string()},
      {kInt8UseNativeCalibrationTableOption, "1"},
  };

  const auto provider = CreateMIGraphXProvider(provider_options);
  const auto options = provider->GetProviderOptions();

  EXPECT_EQ(options.at(kInt8CalibrationTableNameOption), calibration_table_path_.string());
}

TEST_F(MIGraphXInt8CalibrationTableTest, EmptyTableWithoutEnvironmentDoesNotLoad) {
  ScopedEnvironmentVariables environment{ClearedInt8Environment()};
  const ProviderOptions provider_options{{kInt8EnableOption, "1"}};

  const auto provider = CreateMIGraphXProvider(provider_options);
  const auto options = provider->GetProviderOptions();

  EXPECT_TRUE(options.at(kInt8CalibrationTableNameOption).empty());
}

TEST_F(MIGraphXInt8CalibrationTableTest, InvalidEnvironmentTablePathIsRejected) {
  ScopedEnvironmentVariables environment{Int8Environment(missing_calibration_table_path_.string())};

  EXPECT_THROW(CreateMIGraphXProvider(), std::runtime_error);
}

TEST_F(MIGraphXInt8CalibrationTableTest, InvalidProviderOptionTablePathIsRejected) {
  ScopedEnvironmentVariables environment{ClearedInt8Environment()};
  const ProviderOptions provider_options{
      {kInt8EnableOption, "1"},
      {kInt8CalibrationTableNameOption, missing_calibration_table_path_.string()},
      {kInt8UseNativeCalibrationTableOption, "1"},
  };

  EXPECT_THROW(CreateMIGraphXProvider(provider_options), std::runtime_error);
}

#if defined(WIN32)
static bool SessionHasEp(Ort::Session& session, const char* ep_name) {
  // Access the underlying InferenceSession.
  const OrtSession* ort_session = session;
  const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);
  bool has_ep = false;

  for (const auto& provider : s->GetRegisteredProviderTypes()) {
    if (provider == ep_name) {
      has_ep = true;
      break;
    }
  }
  return has_ep;
}

// Tests autoEP feature to automatically select an EP that supports the GPU.
// Currently only works on Windows.
TEST(MIGraphXExecutionProviderTest, AutoEp_PreferGpu) {
  PathString model_name = ORT_TSTR("migraphx_basic_test.onnx");

  onnxruntime::Model model("test", false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};
  CreateBaseModel(model, dims);

  auto status = onnxruntime::Model::Save(model, model_name);
  ASSERT_TRUE(status.IsOK());

  auto env = Ort::Env();
  env.UpdateEnvWithCustomLogLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);

  {
    env.RegisterExecutionProviderLibrary(kMIGraphXExecutionProvider, ORT_TSTR("onnxruntime_providers_migraphx.dll"));

    Ort::SessionOptions so;
    so.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
    Ort::Session session_object(env, model_name.c_str(), so);
    EXPECT_TRUE(SessionHasEp(session_object, kMIGraphXExecutionProvider));
  }

  env.UnregisterExecutionProviderLibrary(kMIGraphXExecutionProvider);
}
#endif

}  // namespace test
}  // namespace onnxruntime
