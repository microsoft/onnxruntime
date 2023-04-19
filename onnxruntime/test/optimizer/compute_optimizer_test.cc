// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <random>
#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "asserts.h"
#include "core/common/span_utils.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"

#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/compute_optimizer/compute_optimizer.h"
#include "core/optimizer/utils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/util/math.h"

#include "test/compare_ortvalue.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

// LayerNormalization/Gelu implementation are in contrib namespace (OnnxDomain 1), so
// Without contib_ops enabled, we cannot parse the graph correctly.
#ifndef DISABLE_CONTRIB_OPS
static void GatherNDComputationReductionTest(const std::string& op_type,
                                             const logging::Logger& logger,
                                             std::function<void(Graph&, std::string op_type)> validation_func) {
  std::string op_type_lower = op_type;
  std::transform(op_type_lower.begin(), op_type_lower.end(), op_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string file_path = std::string("testdata/transform/computation_reduction/gathernd/gathernd_") + op_type_lower +
                          std::string(".onnx");
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(file_path), model, nullptr, logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

  validation_func(graph, op_type);
}

void SingleOpDefaultValidationFunc(Graph& graph, std::string op_type) {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  Node* gathernd_node = nullptr;
  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    ASSERT_FALSE(p_node == nullptr);
    if (p_node->OpType().compare("GatherND") == 0) {
      gathernd_node = p_node;
      EXPECT_EQ(gathernd_node->MutableInputDefs()[0]->Name(), "input");
      const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
      EXPECT_EQ(consumers[0]->OpType(), op_type);
    }
  }

  ASSERT_FALSE(gathernd_node == nullptr);
}

TEST(ComputeOptimizerTests, GatherND_Gelu) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("Gelu", *logger, SingleOpDefaultValidationFunc);
}

TEST(ComputeOptimizerTests, GatherND_Add) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("Add", *logger, [](Graph& graph, std::string op_type) -> void {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    Node* gathernd_node = nullptr;
    bool found_gathernd_around_graph_output = false;
    for (auto node_index : node_topology_list) {
      Node* p_node = graph.GetNode(node_index);
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("GatherND") == 0) {
        if (p_node->OutputDefs()[0]->Name().compare("output") != 0) {
          gathernd_node = p_node;
          EXPECT_EQ(gathernd_node->MutableInputDefs()[0]->Name(), "input");
          const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
          EXPECT_EQ(consumers[0]->OpType(), op_type);
        } else {
          found_gathernd_around_graph_output = true;
        }
      }
    }
    ASSERT_FALSE(gathernd_node == nullptr);
    EXPECT_TRUE(found_gathernd_around_graph_output); });
}

TEST(ComputeOptimizerTests, GatherND_LayerNormalization) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("LayerNormalization", *logger, SingleOpDefaultValidationFunc);
}

TEST(ComputeOptimizerTests, GatherND_MatMul) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("MatMul", *logger, SingleOpDefaultValidationFunc);
}

/**
 * @brief Class represent a input data (dimensions, data type and value).
 */
struct TestInputData {
  template <typename T>
  TestInputData(const std::string& name, const TensorShapeVector& dims, const std::vector<T>& values)
      : name_(name), dims_(dims), values_(values) {}

  OrtValue ToOrtValue() {
    OrtValue ortvalue;
    std::vector<int64_t> dims;
    dims.reserve(dims_.size());
    dims.insert(dims.end(), dims_.begin(), dims_.end());
    std::visit([&ortvalue, &dims](auto&& arg) {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                    std::is_same_v<T, std::vector<float>> ||
                    std::is_same_v<T, std::vector<MLFloat16>>)
        CreateMLValue<typename T::value_type>(
            TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims, arg, &ortvalue);
      else
        static_assert("Unspported types!");
    },
               values_);

    return ortvalue;
  }

  std::string GetName() const {
    return name_;
  }

 private:
  std::string name_;
  TensorShapeVector dims_;
  std::variant<std::vector<float>, std::vector<MLFloat16>, std::vector<int64_t>> values_;
};

void RandomFillFloatVector(const TensorShapeVector& shape, std::vector<float>& data) {
  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;
  data.resize(TensorShape(shape).Size());
  std::default_random_engine generator_float{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution_float{mean, scale};

  std::for_each(data.begin(), data.end(),
                [&generator_float, &distribution_float](float& value) {
                  value = distribution_float(generator_float);
                });
}

void RandomFillHalfVector(const TensorShapeVector& shape, std::vector<MLFloat16>& data) {
  std::vector<float> data_float(TensorShape(shape).Size());
  std::transform(data_float.begin(), data_float.end(), data.begin(),
                 [](float value) { return MLFloat16(math::floatToHalf(value)); });
}

struct InputContainer {
  InputContainer() = default;

  template <typename T>
  TestInputData& AddInput(const std::string& name, const TensorShapeVector dims, const std::vector<T>& values) {
    inputs_.emplace_back(TestInputData(name, dims, values));
    return inputs_.back();
  }

  template <typename T>
  TestInputData& AddInput(const std::string& name, TensorShapeVector dims,
                          std::function<
                              void(const TensorShapeVector& shape, std::vector<T>& data)>
                              func = nullptr) {
    std::vector<T> values(TensorShape(dims).Size());
    if (func) {
      func(dims, values);
    }

    inputs_.emplace_back(TestInputData(name, dims, values));
    return inputs_.back();
  }

  void ToInputMap(NameMLValMap& feeds) const {
    for (auto input : inputs_) {
      feeds.insert({input.GetName(), input.ToOrtValue()});
    }
  }

 private:
  std::vector<TestInputData> inputs_;
};

static void RunModelWithData(const PathString& model_uri, const std::string session_log_id,
                             const std::string& provider_type, const InputContainer& input_container,
                             const std::vector<std::string>& output_names,
                             std::vector<OrtValue>& run_results) {
  SessionOptions so;
  // we don't want any transformation here.
  so.graph_optimization_level = TransformerLevel::Default;
  so.session_logid = session_log_id;

  InferenceSession session_object{so, GetEnvironment()};
  std::unique_ptr<IExecutionProvider> execution_provider;
  if (provider_type == onnxruntime::kCpuExecutionProvider)
    execution_provider = DefaultCpuExecutionProvider();
  else if (provider_type == onnxruntime::kCudaExecutionProvider)
    execution_provider = DefaultCudaExecutionProvider();
  else if (provider_type == onnxruntime::kRocmExecutionProvider)
    execution_provider = DefaultRocmExecutionProvider();
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

  Status st;
  ASSERT_TRUE((st = session_object.Load(model_uri)).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st.ErrorMessage();

  NameMLValMap feeds;
  input_container.ToInputMap(feeds);

  // Now run
  RunOptions run_options;
  st = session_object.Run(run_options, feeds, output_names, &run_results);

  ASSERT_TRUE(st.IsOK()) << "RunModelWithData  run graph failed with error: " << st.ErrorMessage();
}

TEST(ComputeOptimizerTests, GatherND_E2E) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gathernd/e2e.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  // check the expected node orders.
  {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    Node* gathernd_node = nullptr;
    for (auto node_index : node_topology_list) {
      Node* p_node = graph.GetNode(node_index);
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("GatherND") == 0) {
        gathernd_node = p_node;
        const Node* layer_norm_node = graph.GetProducerNode(gathernd_node->MutableInputDefs()[0]->Name());
        EXPECT_EQ(layer_norm_node->OpType(), "LayerNormalization");
        EXPECT_EQ(layer_norm_node->Name(), "layer_norm_1");
        const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
        EXPECT_EQ(consumers[0]->OpType(), "MatMul");
        EXPECT_EQ(consumers[0]->Name(), "matmul_1");
        break;
      }
    }

    ASSERT_FALSE(gathernd_node == nullptr);
  }

  // check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("computation_reduction_transformer_after.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  InputContainer input_container;

  int batch_size = 8;
  int sequence = 128;
  int hidden_size = 128;
  int dynamic_predict_count = 20;
  input_container.AddInput<float>("input", {batch_size, sequence, hidden_size}, RandomFillFloatVector);

  const TensorShapeVector dims_unsqueezed_masked_lm_positions{batch_size, dynamic_predict_count, 1};
  std::vector<int64_t> values_unsqueezed_masked_lm_positions(TensorShape(dims_unsqueezed_masked_lm_positions).Size());

  std::random_device rd;                                   // obtain a random number from hardware
  std::mt19937 eng(rd());                                  // seed the generator
  std::uniform_int_distribution<> distr(0, sequence - 1);  // define the range
  std::for_each(values_unsqueezed_masked_lm_positions.begin(), values_unsqueezed_masked_lm_positions.end(),
                [&distr, &eng](int64_t& value) { value = distr(eng); });

  input_container.AddInput<int64_t>("unsqueezed_masked_lm_positions",
                                    dims_unsqueezed_masked_lm_positions,
                                    values_unsqueezed_masked_lm_positions);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };
  const std::vector<std::string> output_names{"output", "gather_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"), provider_type,
                     input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherMatMul_ScalarSlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_scalar_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Unsqueeze");

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Unsqueeze");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_matmul_scalar_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherMatMul_SlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Gather");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_matmul_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherMatMul_ScalarSlicingOnLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_scalar_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 2);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_TRUE(lhs_input == nullptr);

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Unsqueeze");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_matmul_scalar_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherMatMul_SlicingOnLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 2);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_TRUE(lhs_input == nullptr);

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Gather");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_matmul_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherMatMul_ScalarSlicingOnSecondLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_scalar_second_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  // Check the second branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check MatMul(who gathers on the second last dim)'s input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Unsqueeze");

    ASSERT_TRUE(rhs_input == nullptr);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(
      tmp_dir.Path(),
      ORT_TSTR("gather_matmul_scalar_second_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherMatMul_SlicingOnSecondLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_second_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  // Check the second branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_matmul_second_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherReshape_ScalarSlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_scalar_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 3U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 16);
    ASSERT_EQ(new_shape_const_values[2], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_reshape_scalar_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherReshape_SlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 4U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 0);
    ASSERT_EQ(new_shape_const_values[2], 16);
    ASSERT_EQ(new_shape_const_values[3], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_reshape_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherReshape_ScalarSlicingOnSeqlenDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_scalar_seqlen_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 3U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 16);
    ASSERT_EQ(new_shape_const_values[2], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_reshape_scalar_seqlen_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherReshape_SlicingOnSeqlenDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_seqlen_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 4U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 0);
    ASSERT_EQ(new_shape_const_values[2], 16);
    ASSERT_EQ(new_shape_const_values[3], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_reshape_seqlen_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherReshape_SlicingOnSeqlenDim2) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_seqlen_dim2.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 4U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 31);
    ASSERT_EQ(new_shape_const_values[2], 16);
    ASSERT_EQ(new_shape_const_values[3], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_reshape_seqlen_dim2_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 128;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

TEST(ComputeOptimizerTests, GatherRobertaE2E) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  // Be noted, all dropout have ratio be 0.0, to make it easier to compare when running with session.
  // This did not affect the transformer tests, because we did not remove the Dropout of ratio 0. in the middle.
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_roberta_e2e.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{3};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ComputeOptimizer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("c1_out");
    const Node* gather_node = nullptr;
    for (auto p_node : consumers) {
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("Gather") == 0) {
        gather_node = p_node;
        const Node* cast_node = graph.GetProducerNode(gather_node->InputDefs()[0]->Name());
        EXPECT_EQ(cast_node->OpType(), "Cast");
        EXPECT_EQ(cast_node->Name(), "c1");
        const auto& gather_consumers = graph.GetConsumerNodes(gather_node->OutputDefs()[0]->Name());
        EXPECT_EQ(gather_consumers[0]->OpType(), "Unsqueeze");
        break;
      }
    }

    ASSERT_FALSE(gather_node == nullptr);
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("d1_out");
    const Node* gather_node = nullptr;
    for (auto p_node : consumers) {
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("Gather") == 0) {
        gather_node = p_node;
        const Node* dropout_node = graph.GetProducerNode(gather_node->InputDefs()[0]->Name());
        EXPECT_EQ(dropout_node->OpType(), "Dropout");
        EXPECT_EQ(dropout_node->Name(), "d1");
        const auto& gather_consumers = graph.GetConsumerNodes(gather_node->OutputDefs()[0]->Name());
        EXPECT_EQ(gather_consumers[0]->OpType(), "Add");
        EXPECT_EQ(gather_consumers[0]->Name(), "a6");
        break;
      }
    }

    ASSERT_FALSE(gather_node == nullptr);
  }

  // Check the input/output of the original Gather node.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("layernorm2_out");
    ASSERT_TRUE(consumers.size() == 1);
    ASSERT_FALSE(consumers[0] == nullptr);
    EXPECT_EQ(consumers[0]->OpType(), "Dropout");
    EXPECT_EQ(consumers[0]->Name(), "d6");
  }

  // Check MatMul(who gathers on the second last dim)'s input and output.
  {
    const Node* m5 = graph.GetProducerNode("m5_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m5");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Unsqueeze");

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Transpose");
    EXPECT_EQ(rhs_input->Name(), "transpose1");
  }

  // Check Add(who has broadcastable dim on gather axis)'s input and output.
  {
    const Node* a4 = graph.GetProducerNode("a4_out");
    ASSERT_FALSE(a4 == nullptr);
    EXPECT_EQ(a4->OpType(), "Add");
    EXPECT_EQ(a4->Name(), "a4");

    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("a4_out");
    ASSERT_TRUE(consumers.size() == 1);
    ASSERT_FALSE(consumers[0] == nullptr);
    EXPECT_EQ(consumers[0]->OpType(), "Squeeze");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(tmp_dir.Path(),
                                                         ORT_TSTR("gather_roberta_e2e_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  const TensorShapeVector dims_mask = {batch_size, sequence_length};
  std::vector<int64_t> attention_mask(TensorShape(dims_mask).Size(), 1);
  input_container.AddInput<int64_t>("attention_mask", dims_mask, attention_mask);

  input_container.AddInput<MLFloat16>("matmul1.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add1.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul2.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add2.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul3.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add3.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul4.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add4.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<float>("layer_norm1.weight", {hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("layer_norm1.bias", {hidden_size}, RandomFillFloatVector);

  input_container.AddInput<MLFloat16>("matmul7.weight", {hidden_size, hidden_size * 4}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add7.bias", {hidden_size * 4}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul8.weight", {hidden_size * 4, hidden_size}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add8.bias", {hidden_size}, RandomFillHalfVector);

  input_container.AddInput<float>("layer_norm2.weight", {hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("layer_norm2.bias", {hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());

    // "expected 0.793675 (3f4b2e44), got 0.79232 (3f4ad584), diff: 0.00135422, tol=0.000179367 idx=4276.
    // 1713 of 8192 differ"
    // Loose the atol a bit because we see the MatMuls results differs once we move Gather before it.
    constexpr double per_sample_tolerance = 2e-3;
    constexpr double relative_per_sample_tolerance = 2e-3;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime

#endif
