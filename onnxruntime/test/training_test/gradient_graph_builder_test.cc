// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/gradient_op_test_utils.h"
#include "core/training/training_optimizer.h"
#include "core/training/weight_updater.h"
#include "core/providers/cpu/cpu_execution_provider.h"

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace google::protobuf::util;

namespace onnxruntime {
namespace test {

constexpr auto ORIGINAL_MODEL_PATH = "testdata/test_training_model.onnx";
constexpr auto BACKWARD_MODEL_PATH = "backward_model.onnx";

const std::string TAB = "\t";

AllocatorPtr GetAllocator() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return cpu_provider.GetAllocator(0, OrtMemTypeDefault);
}
template <typename T>
static void CreateMLValue(const AllocatorPtr& alloc,
                          const std::vector<int64_t>& dims,
                          const std::vector<T>& value,
                          MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto location = alloc->Info();
  auto element_type = DataTypeImpl::GetType<T>();
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  if (!value.empty()) {
    memcpy(buffer, &value[0], element_type->Size() * shape.Size());
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

static std::string BuildBackPropGraph(const std::string& forward_model_file, const LossFunctionInfo& loss_func_info) {
  const std::string backward_model_file = BACKWARD_MODEL_PATH;

  std::unique_ptr<Environment> env;
  EXPECT_TRUE(Environment::Create(env).IsOK());

  SessionOptions so;
  TrainingSession training_session{so};

  std::cout << "Loading source model file = " << forward_model_file << std::endl;

  EXPECT_TRUE(training_session.Load(forward_model_file).IsOK());

  auto model_inputs = training_session.GetModelInputNames();
  std::cout << "Model input names = [" << std::endl;
  for (auto& n : model_inputs) {
    std::cout << TAB << n << std::endl;
  }
  std::cout << "]" << std::endl;

  auto model_outputs = training_session.GetModelOutputNames();
  std::cout << "Model output names = [" << std::endl;
  for (auto& n : model_outputs) {
    std::cout << TAB << n << std::endl;
  }
  std::cout << "]" << std::endl;

  EXPECT_TRUE(training_session.BuildLossFunction(loss_func_info).IsOK());
  EXPECT_TRUE(training_session.BuildGradientGraph(model_inputs, loss_func_info.loss_name).IsOK());

  EXPECT_TRUE(training_session.Save(backward_model_file,
                                    TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS)
                  .IsOK());

  return backward_model_file;
}

static std::unique_ptr<TrainingSession> RunTrainingSessionWithChecks(
    SessionOptions& so, const std::string& backprop_model_file) {
  std::unique_ptr<Environment> env;
  EXPECT_TRUE(Environment::Create(env).IsOK());

  const auto& log_manager = so.session_log_verbosity_level > 0 ? &DefaultLoggingManager() : nullptr;

  std::unique_ptr<TrainingSession> training_session = std::make_unique<TrainingSession>(so, log_manager);

  EXPECT_TRUE(training_session->Load(backprop_model_file).IsOK());

  std::pair<common::Status, const ModelMetadata*> res = training_session->GetModelMetadata();
  EXPECT_TRUE(res.first.IsOK());
  EXPECT_TRUE(res.second != nullptr);
  auto model_metadata = res.second;
  std::cout << "Loaded " << model_metadata->graph_name << std::endl;

  EXPECT_TRUE(training_session->Initialize().IsOK());

  // Create a WeightUpdater powered by GradientDescent algorithm.
  const static float LEARNING_RATE = 0.5f;

  WeightUpdater<out_graph_optimizer::GradientDescent> weight_updater(*training_session, {LEARNING_RATE, GetAllocator()});

  std::vector<MLValue> gradient_fetches;
  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;

  // Create dummy feeds
  std::vector<int64_t> image_dims = {1, 784};
  std::vector<int64_t> label_dims = {1, 10};
  std::vector<float> image_value(784, 1);
  std::vector<float> label_value(10, 1);

  MLValue imageMLValue;
  CreateMLValue(GetAllocator(), image_dims, image_value, &imageMLValue);
  MLValue labelMLValue;
  CreateMLValue(GetAllocator(), label_dims, label_value, &labelMLValue);

  auto fw_feeds = std::make_pair<std::vector<std::string>, std::vector<MLValue>>({"X", "labels"}, {imageMLValue, labelMLValue});

  auto output_names_include_gradients = training_session->GetModelOutputNames();
  std::vector<std::string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  EXPECT_TRUE(training_session->Run(run_options, fw_feeds.first, fw_feeds.second, training_output_names, &gradient_fetches).IsOK());

  // Get gradients
  NameMLValMap grad;
  for (size_t i = 0; i < training_output_names.size(); i++) {
    if (training_output_names[i] == "loss") continue;
    if (training_output_names[i] == "predictions") continue;

    grad.insert(make_pair(training_output_names[i], gradient_fetches[i]));
  }

  EXPECT_TRUE(weight_updater.Update(grad, 1).IsOK());

  return training_session;
}

TEST(GradientGraphBuilderTest, BuildGradientGraphTest) {
  const auto loss_func_info = LossFunctionInfo(
      OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  const std::string& backprop_model_file = BuildBackPropGraph(ORIGINAL_MODEL_PATH, loss_func_info);

  std::shared_ptr<Model> pModel;
  EXPECT_TRUE(Model::Load(backprop_model_file, pModel).IsOK());

  Graph& graph = pModel->MainGraph();
  EXPECT_FALSE(graph.GraphResolveNeeded());
  EXPECT_TRUE(graph.NumberOfNodes() > 0);
  EXPECT_TRUE(graph.MaxNodeIndex() > 0);

  std::cout << "Graph input names = [" << std::endl;
  for (const NodeArg* p_node_arg : graph.GetInputs()) {
    std::cout << TAB << p_node_arg->Name() << std::endl;
  }
  std::cout << "]" << std::endl;

  std::cout << "Graph output names = [" << std::endl;
  for (const NodeArg* p_node_arg : graph.GetOutputs()) {
    std::cout << TAB << p_node_arg->Name() << std::endl;
  }
  std::cout << "]" << std::endl;

  for (Node& node : graph.Nodes()) {
    std::cout << "Operation node:"
              << " Index=" << node.Index()
              << (node.NodeType() == Node::Type::Fused ? "-(FUSED)" : "")
              << " OpType=" << node.OpType()
              << " Name=" << node.Name()
              << std::endl;
  }
}

TEST(GradientGraphBuilderTest, RunTrainingSessionTest_Basic) {
  const auto loss_func_info = LossFunctionInfo(
      OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  const std::string& backprop_model_file = BuildBackPropGraph(ORIGINAL_MODEL_PATH, loss_func_info);

  SessionOptions so;
  RunTrainingSessionWithChecks(so, backprop_model_file);
}

TEST(GradientGraphBuilderTest, RunTrainingSessionTest_WithLogging) {
  const auto& log_manager = DefaultLoggingManager();
  const auto& default_logger = log_manager.DefaultLogger();
  log_manager.SetDefaultLoggerSeverity(Severity::kINFO);

  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kERROR, DataType::USER)) << "ERROR level logging enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kWARNING, DataType::USER)) << "WARNING level logging enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kINFO, DataType::USER)) << "INFO level logging enabled.";

  const auto loss_func_info = LossFunctionInfo(
      OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  const std::string& backprop_model_file = BuildBackPropGraph(ORIGINAL_MODEL_PATH, loss_func_info);

  SessionOptions so;
  so.session_logid = "training_session_with_logging";
  so.session_log_verbosity_level = 1;  // 1 == detailed logging

  std::unique_ptr<TrainingSession> training_session = RunTrainingSessionWithChecks(so, backprop_model_file);

  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kERROR, DataType::USER)) << "ERROR level logging still enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kWARNING, DataType::USER)) << "WARNING level logging still enabled.";
  EXPECT_TRUE(default_logger.OutputIsEnabled(Severity::kINFO, DataType::USER)) << "INFO level logging still enabled.";

  std::string profile_file = training_session->EndProfiling();

  log_manager.SetDefaultLoggerSeverity(Severity::kWARNING);

  EXPECT_EQ(profile_file, std::string()) << "There should be no profile output file.";
}

TEST(GradientGraphBuilderTest, RunTrainingSessionTest_WithProfiler) {
  const auto loss_func_info = LossFunctionInfo(
      OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  const std::string& backprop_model_file = BuildBackPropGraph(ORIGINAL_MODEL_PATH, loss_func_info);

  SessionOptions so;
  so.enable_profiling = true;
  so.profile_file_prefix = ORT_TSTR("onnx_training_profiler_test");

  std::unique_ptr<TrainingSession> training_session = RunTrainingSessionWithChecks(so, backprop_model_file);

  std::string profile_file = training_session->EndProfiling();

  std::cout << "Profile output file = " << profile_file << std::endl;

  std::ifstream profile(profile_file);
  ASSERT_TRUE(profile);

  std::vector<std::string> tags = {"pid", "dur", "ts", "ph", "X", "name", "args"};
  int count = 0;
  std::string line;
  while (std::getline(profile, line)) {
    if (count == 0) {
      ASSERT_TRUE(line.find('[') != std::string::npos);
      // Opening array marker found.
    } else if (line.find(']') != std::string::npos) {
      // Closing array marker found.
      break;
    } else if (count >= 1) {
#ifdef DEBUG
      std::cout << count << ": " << line << std::endl;
#endif
      if (count == 1) {
        ASSERT_TRUE(line.find("model_loading_uri") != std::string::npos);
      }

      for (auto& s : tags) {
        ASSERT_TRUE(line.find(s) != std::string::npos);
      }
    }

    count++;
  }
  ASSERT_TRUE(count > 1);
}
}  // namespace test
}  // namespace onnxruntime
