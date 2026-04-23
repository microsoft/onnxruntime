// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>  // needed for std::transform
#include "gtest/gtest.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/common/path_utils.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/fuse_initializers_transformer.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/common/random_generator.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

namespace test {

/**
 *
 *   "fuse_fp16_initializers.onnx"
 *
 *           --------
 *          | X_Fp16 |
 *           --------
 *              |
 *              |
 *              |
 *              |
 *              v
 *  ---------------------------
 * |        Conv_Fp16          |
 * |        --W_Fp16--         |
 * |        --B_Fp16--         |
 *  ---------------------------
 *              |
 *              |
 *              |
 *              |
 *              v
 *           --------
 *          | Y_Fp16 |
 *           --------
 */

/**
 *
 *    "fuse_fp16_initializers_with_graph_outputs.onnx"
 *
 *        --------              ---------------
 *       | A_fp32 |            |     Cast      |
 *        --------             | (to: Float32) |
 *         |                   |  --X_fp16--   |
 *         |                    ---------------
 *         |                      |         |
 *         |                      |         |
 *         |                      |         |
 *         |                      v         |
 *         |   +---------------(B_fp32)     |
 *         |   |                            |
 *         v   v                            |
 *        -------                           |
 *       |  Add  |                          |
 *        -------                           |
 *           |                              |
 *           |                              |
 *           v                              v
 *        --------                       --------
 *       | C_fp32 |                     | B_fp32 |
 *        --------                       --------
 *     (graph output)                 (graph output)
 */

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

unsigned int CountNoOfInitializersInGraph(const Graph& graph, const onnxruntime::MLDataType _data_type) {
  // init
  unsigned int num_initializers = 0;

  // Get nodes in topological order
  const GraphViewer graph_viewer(graph);
  auto nodes_indexes_in_topological_order = graph_viewer.GetNodesInTopologicalOrder();

  // For each Node
  for (auto node_index : nodes_indexes_in_topological_order) {
    // Get Node
    auto node = graph.GetNode(node_index);

    // Get input defs
    auto node_input_defs = node->InputDefs();

    // For each Node Args
    for (NodeIndex node_arg_index = 0; node_arg_index < node_input_defs.size(); ++node_arg_index) {
      // Continue if the current arg is not an initialized tensor
      if (!(graph.IsInitializedTensor(node_input_defs[node_arg_index]->Name()))) continue;

      // Continue if initialzed tensor is not of specific type
      if (!(_data_type == DataTypeImpl::TypeFromProto(*(node_input_defs[node_arg_index]->TypeAsProto())))) continue;

      // increment
      num_initializers += 1;
    }
  }

  return num_initializers;
}

unsigned int CountNoOfNodesInGraph(const Graph& graph, const onnxruntime::MLDataType _data_type) {
  // init
  unsigned int num_nodes = 0;
  unsigned int num_args_in_a_node = 0;

  // Get nodes in topological order
  const GraphViewer graph_viewer(graph);
  auto nodes_indexes_in_topological_order = graph_viewer.GetNodesInTopologicalOrder();

  // For each Node
  for (auto node_index : nodes_indexes_in_topological_order) {
    // Get Node
    auto node = graph.GetNode(node_index);

    // Get input defs
    auto node_input_defs = node->InputDefs();

    // For each Node Args
    num_args_in_a_node = 0;
    for (NodeIndex node_arg_index = 0; node_arg_index < node_input_defs.size(); ++node_arg_index) {
      // Continue if current arg is not of specific type
      if (!(_data_type == DataTypeImpl::TypeFromProto(*(node_input_defs[node_arg_index]->TypeAsProto())))) continue;

      num_args_in_a_node += 1;
    }

    // increment
    num_nodes += ((node_input_defs.size() == num_args_in_a_node) ? 1 : 0);
  }

  return num_nodes;
}

void test_graph_structure_at_init(const Graph& graph) {
  // Count ops
  auto op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(0, op_to_count["Cast"]);
  // Count no. of initializers of FP16 type
  auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(2, num_initializers_fp16);
  // Count no. of initializers of FP32 type
  auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(0, num_initializers_fp32);
  // Count no. of FP16 nodes
  auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(1, num_nodes_fp16);
  // Count no. of FP32 nodes
  auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(0, num_nodes_fp32);
  // Check if all conditions are met
  ASSERT_TRUE((0 == op_to_count["Cast"]) && (2 == num_initializers_fp16) && (0 == num_initializers_fp32) && (1 == num_nodes_fp16) && (0 == num_nodes_fp32));
}

void test_graph_structure_before_fusion(const Graph& graph) {
  // Count ops
  auto op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(4, op_to_count["Cast"]);
  // Count no. of initializers of FP16 type
  auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(2, num_initializers_fp16);
  // Count no. of initializers of FP32 type
  auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(0, num_initializers_fp32);
  // Count no. of FP16 nodes
  auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(3, num_nodes_fp16);
  // Count no. of FP32 nodes
  auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(2, num_nodes_fp32);
  // Check if all conditions are met
  ASSERT_TRUE((4 == op_to_count["Cast"]) && (2 == num_initializers_fp16) && (0 == num_initializers_fp32) && (3 == num_nodes_fp16) && (2 == num_nodes_fp32));
}

void test_graph_structure_after_fusion(const Graph& graph) {
  // Count ops
  auto op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(2, op_to_count["Cast"]);
  // Count no. of initializers of FP16 type
  auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(0, num_initializers_fp16);
  // Count no. of initializers of FP32 type
  auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(2, num_initializers_fp32);
  // Count no. of FP16 nodes
  auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(1, num_nodes_fp16);
  // Count no. of FP32 nodes
  auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(2, num_nodes_fp32);
  // Check if all conditions are met
  ASSERT_TRUE((2 == op_to_count["Cast"]) && (0 == num_initializers_fp16) && (2 == num_initializers_fp32) && (1 == num_nodes_fp16) && (2 == num_nodes_fp32));
}

void test_graph_structure_after_session_init_without_graph_optimization_loop(const Graph& graph) {
  // Note: Unable to calc no. of fp16/fp32 initializers, as when session
  // state is finalized after init it removes initializers from graph.
  // Look for "session_state_->FinalizeSessionState" method in
  // inference_session.cc for more explanation.

  // Count ops
  auto op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(2, op_to_count["Cast"]);
  // Count no. of initializers of FP16 type
  auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(0, num_initializers_fp16);
  // Count no. of initializers of FP32 type
  auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(0, num_initializers_fp32);
  // Count no. of FP16 nodes
  auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(1, num_nodes_fp16);
  // Count no. of FP32 nodes
  auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(2, num_nodes_fp32);
  // Check if all conditions are met
  ASSERT_TRUE((2 == op_to_count["Cast"]) && (0 == num_initializers_fp16) && (0 == num_initializers_fp32) && (1 == num_nodes_fp16) && (2 == num_nodes_fp32));
}

void test_graph_structure_after_session_init_with_graph_optimization_loop(const Graph& graph) {
  // Note: Unable to calc no. of fp16/fp32 initializers, as when session
  // state is finalized after init it removes initializers from graph.
  // Look for "session_state_->FinalizeSessionState" method in
  // inference_session.cc for more explanation.

  // Count ops
  auto op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(2, op_to_count["Cast"]);
  // Count no. of initializers of FP16 type
  auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(0, num_initializers_fp16);
  // Count no. of initializers of FP32 type
  auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
  EXPECT_EQ(0, num_initializers_fp32);
  // Count no. of FP16 nodes
  auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(1, num_nodes_fp16);
  // Count no. of FP32 nodes
  auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
  // NOTE: On platforms where the NCHWC transformer is not supported, no reorder node will be added
  // to the final optimized graph. Consequently, there will be "two" FP32 nodes instead of "three".
  // This is an expected behavior and can be verified by testing if the Nchwc block size is greater than 1.
  unsigned int expected_num_nodes_fp32 = (MlasNchwcGetBlockSize() > 1) ? 3 : 2;
  EXPECT_EQ(expected_num_nodes_fp32, num_nodes_fp32);
  // Check if all conditions are met
  ASSERT_TRUE((2 == op_to_count["Cast"]) && (0 == num_initializers_fp16) && (0 == num_initializers_fp32) && (1 == num_nodes_fp16) && (expected_num_nodes_fp32 == num_nodes_fp32));
}

TEST(TransformerTest, FuseFp16InitializersWithFp32Node) {
  // Init
  auto test_logger = DefaultLoggingManager().DefaultLogger();
  auto model_uri = MODEL_FOLDER ORT_TSTR("fuse_fp16_initializers.onnx");  // Graph for this model is drawn at beginning of this file
  std::shared_ptr<Model> model;

  // Load model
  auto status_at_load = Model::Load(model_uri, model, nullptr, test_logger);
  ASSERT_TRUE(status_at_load.IsOK()) << status_at_load;

  // Load Graph
  Graph& graph = model->MainGraph();

  // check graph initial structure
  test_graph_structure_at_init(graph);

  // apply insert cast transforms
  InsertCastTransformer insert_cast_transformer("TransformerTest.FusedInitializers",
                                                DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool graph_modified_by_insert_cast_transforms = false;
  auto status_insert_cast_transforms = insert_cast_transformer.Apply(graph,
                                                                     graph_modified_by_insert_cast_transforms,
                                                                     test_logger);

  EXPECT_TRUE(status_insert_cast_transforms.IsOK()) << status_insert_cast_transforms;
  auto status_insert_cast_transforms_resolve = graph.Resolve();
  EXPECT_TRUE(status_insert_cast_transforms_resolve.IsOK()) << status_insert_cast_transforms_resolve;

  // check graph structure before fusion
  if (graph_modified_by_insert_cast_transforms) {
    test_graph_structure_before_fusion(graph);
  }

  // apply fused initializer transforms
  FuseInitializersTransformer fused_initializers_transformer("TransformerTest.FusedInitializers",
                                                             DataTypeImpl::GetTensorType<MLFloat16>(),
                                                             DataTypeImpl::GetTensorType<float>());

  bool graph_modified_by_fused_initializers_transforms = false;
  auto status_fused_initializers_transforms = fused_initializers_transformer.Apply(graph,
                                                                                   graph_modified_by_fused_initializers_transforms,
                                                                                   test_logger);

  EXPECT_TRUE(status_fused_initializers_transforms.IsOK()) << status_fused_initializers_transforms;
  auto status_fused_initializers_transforms_resolve = graph.Resolve();
  EXPECT_TRUE(status_fused_initializers_transforms_resolve.IsOK()) << status_fused_initializers_transforms_resolve;

  // If insert cast transforms is applied then FP16 compute is not supported
  if (graph_modified_by_insert_cast_transforms) {
    // If fp16 compute is not supported, Fusion is performed.
    // The fp16 node/s is/are transformed to fp32 node/s.
    // For each fp16 initializer in fp16 node/s, a cast node is created, converting fp16 tensors to fp32
    // tensors everytime during each inference.
    // Each of fp16 cast nodes will point to newly created fp32 nodes. Running nodes with fp32 kernel.
    // From input to next node there will be one FP16 to FP32 cast node. Totaling two FP32 node.
    // From last node to output there will be one FP32 to FP16 cast node. Totaling one FP16 node.
    EXPECT_TRUE(graph_modified_by_fused_initializers_transforms) << status_fused_initializers_transforms_resolve;

    // check if graph structure is changed from initial structure
    test_graph_structure_after_fusion(graph);

  } else {
    // If fp16 compute is supported, Fusion is not performed, keeping the graph as it is.
    EXPECT_FALSE(graph_modified_by_fused_initializers_transforms) << status_fused_initializers_transforms_resolve;

    // check if graph structure is same as initial structure
    test_graph_structure_at_init(graph);
  }

}  // FuseFp16InitializersWithFp32Node

// NOTE: All the below tests for graph optimizations loop level will "FAIL" when FP16 Conv nodes are supported.
// In order to avoid this situation we disabled these testcases whenever FP16 Acceleration is supported.

TEST(TransformerTest, FuseFp16InitializersWithFp32Node_with_graph_optimizations_loop_level_set_to_0) {
  if (MlasFp16AccelerationSupported()) {
    GTEST_SKIP() << "Skipping test because FP16 acceleration support was detected.";
  }

  // Make model inputs and outputs
  auto model_uri = MODEL_FOLDER ORT_TSTR("fuse_fp16_initializers.onnx");  // Graph for this model is drawn at beginning of this file
  RandomValueGenerator random{123};
  std::vector<int64_t> x_dims{1, 1, 5, 5};
  std::vector<float> x_data = random.Gaussian<float>(x_dims, 0.0f, 1.0f);
  std::vector<MLFloat16> x_data_fp16(detail::SizeFromDims(x_dims));
  std::transform(x_data.begin(), x_data.end(), x_data_fp16.begin(),
                 [](float value) -> MLFloat16 { return static_cast<MLFloat16>(value); });
  OrtValue x_fp16;
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                           x_dims, x_data_fp16, &x_fp16);
  NameMLValMap inputs{{"X", x_fp16}};

  std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> outputs;

  // set session options
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::MaxLevel;
  // Add graph optimization loop level session option and set it to 0.
  // Hence, during the session initialization only fused initializer graph transforms will be applied
  // as we are disabling the graph optimization loop by setting this session option.
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsGraphOptimizationsLoopLevel, "0"));

  // Create session and check graph before / after initiation
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_uri));
  test_graph_structure_at_init(session.GetGraph());
  ASSERT_STATUS_OK(session.Initialize());
  test_graph_structure_after_session_init_without_graph_optimization_loop(session.GetGraph());
  ASSERT_STATUS_OK(session.Run(inputs, output_names, &outputs));
}  // FuseFp16InitializersWithFp32Node_with_graph_optimizations_loop_level_set_to_0

TEST(TransformerTest, FuseFp16InitializersWithFp32Node_with_graph_optimizations_loop_level_set_to_1) {
  if (MlasFp16AccelerationSupported()) {
    GTEST_SKIP() << "Skipping test because FP16 acceleration support was detected.";
  }

  // Make model inputs and outputs
  auto model_uri = MODEL_FOLDER ORT_TSTR("fuse_fp16_initializers.onnx");  // Graph for this model is drawn at beginning of this file
  RandomValueGenerator random{123};
  std::vector<int64_t> x_dims{1, 1, 5, 5};
  std::vector<float> x_data = random.Gaussian<float>(x_dims, 0.0f, 1.0f);
  std::vector<MLFloat16> x_data_fp16(detail::SizeFromDims(x_dims));
  std::transform(x_data.begin(), x_data.end(), x_data_fp16.begin(),
                 [](float value) -> MLFloat16 { return static_cast<MLFloat16>(value); });
  OrtValue x_fp16;
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                           x_dims, x_data_fp16, &x_fp16);
  NameMLValMap inputs{{"X", x_fp16}};

  std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> outputs;

  // set session options
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::MaxLevel;
  // Add graph optimization loop level session option and set it to 1.
  // Hence, during the session initialization after fused initializer graph transforms is applied
  // the graph optimization loop will run one more time to see if there is any valid graph transforms
  // after the fusion which can be applied. In this case NchwcTransformer is applied.
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsGraphOptimizationsLoopLevel, "1"));

  // Create session and check graph before / after initiation
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_uri));
  test_graph_structure_at_init(session.GetGraph());
  ASSERT_STATUS_OK(session.Initialize());
  test_graph_structure_after_session_init_with_graph_optimization_loop(session.GetGraph());
  ASSERT_STATUS_OK(session.Run(inputs, output_names, &outputs));
}  // FuseFp16InitializersWithFp32Node_with_graph_optimizations_loop_level_set_to_1

TEST(TransformerTest, FuseFp16InitializersWithFp32Node_with_graph_optimizations_loop_level_set_to_2) {
  if (MlasFp16AccelerationSupported()) {
    GTEST_SKIP() << "Skipping test because FP16 acceleration support was detected.";
  }

  // Make model inputs and outputs
  auto model_uri = MODEL_FOLDER ORT_TSTR("fuse_fp16_initializers.onnx");  // Graph for this model is drawn at beginning of this file
  RandomValueGenerator random{123};
  std::vector<int64_t> x_dims{1, 1, 5, 5};
  std::vector<float> x_data = random.Gaussian<float>(x_dims, 0.0f, 1.0f);
  std::vector<MLFloat16> x_data_fp16(detail::SizeFromDims(x_dims));
  std::transform(x_data.begin(), x_data.end(), x_data_fp16.begin(),
                 [](float value) -> MLFloat16 { return static_cast<MLFloat16>(value); });
  OrtValue x_fp16;
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                           x_dims, x_data_fp16, &x_fp16);
  NameMLValMap inputs{{"X", x_fp16}};

  std::vector<std::string> output_names{"Y"};
  std::vector<OrtValue> outputs;

  // set session options
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::MaxLevel;
  // Add graph optimization loop level session option and set it to 2.
  // Hence, during the session initialization after fused initializer graph transforms is applied
  // the graph optimization loop will run one more time to see if there is any valid graph transforms
  // after the fusion which can be applied. In this case NchwcTransformer is applied. Again the graph
  // optimization loop will run one more time to check if there is any valid graph transforms which
  // can be applied after nchwc transforms. This running one more time will not change anything.
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsGraphOptimizationsLoopLevel, "2"));

  // Create session and check graph before / after initiation
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_uri));
  test_graph_structure_at_init(session.GetGraph());
  ASSERT_STATUS_OK(session.Initialize());
  test_graph_structure_after_session_init_with_graph_optimization_loop(session.GetGraph());
  ASSERT_STATUS_OK(session.Run(inputs, output_names, &outputs));
}  // FuseFp16InitializersWithFp32Node_with_graph_optimizations_loop_level_set_to_2

TEST(TransformerTest, SimplifiedLayerNormWithFp16ConstantDoesNotRevisitLevel3AtDefaultLoopLevel) {
  // Build a minimal SimplifiedLayerNorm pattern where the Pow exponent comes from a float16 Constant node.
  // InsertCast will add a Cast between the Constant and Pow (to run Pow in fp32 on CPU). When Level 3 reruns
  // after InsertCast (old behavior), SimplifiedLayerNormFusion would crash while fusing because the extra Cast
  // input name is not present in the replacement node. With the default loop level (1), L2/L3 run only once,
  // before InsertCast, so initialization must succeed.

  constexpr int opset = 14;
  const onnxruntime::logging::Logger& logger = logging::LoggingManager::DefaultLogger();
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, opset}};
  Model model("SimplifiedLayerNormFp16Const", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  // Inputs and initializers
  std::vector<MLFloat16> x_data(24, MLFloat16(0.5f));
  auto* x = builder.MakeInput<MLFloat16>({1, 2, 3, 4}, x_data);
  auto* add_initializer = builder.MakeInitializer<float>({}, {1e-5f});
  auto* weight_initializer =
      builder.MakeInitializer<MLFloat16>({4}, std::vector<MLFloat16>(4, MLFloat16(1.0f)));

  // Intermediate values
  auto* reduce_mean_out = builder.MakeIntermediate();
  auto* sub_out = builder.MakeIntermediate();
  auto* cast_out = builder.MakeIntermediate();
  auto* pow_const = builder.MakeIntermediate<MLFloat16>({});
  auto* pow_out = builder.MakeIntermediate();
  auto* reduce_mean_out_2 = builder.MakeIntermediate();
  auto* add_out = builder.MakeIntermediate();
  auto* sqrt_out = builder.MakeIntermediate();
  auto* div_out = builder.MakeIntermediate();
  auto* cast_back_fp16 = builder.MakeIntermediate();
  auto* mul_out = builder.MakeOutput();

  // Build graph
  builder.AddNode("ReduceMean", std::vector<NodeArg*>{x}, std::vector<NodeArg*>{reduce_mean_out})
      .AddAttribute("axes", std::vector<int64_t>{-1});
  builder.AddNode("Sub", std::vector<NodeArg*>{x, reduce_mean_out}, std::vector<NodeArg*>{sub_out});
  builder.AddNode("Cast", std::vector<NodeArg*>{sub_out}, std::vector<NodeArg*>{cast_out})
      .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  // float16 Constant (value 2.0) feeding Pow exponent
  ONNX_NAMESPACE::TensorProto pow_two_tensor;
  pow_two_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  pow_two_tensor.add_dims(1);
  const MLFloat16 fp16_two(2.0f);
  pow_two_tensor.set_raw_data(&fp16_two, sizeof(fp16_two));
  builder.AddNode("Constant", std::vector<NodeArg*>{}, std::vector<NodeArg*>{pow_const}).AddAttribute("value", pow_two_tensor);

  builder.AddNode("Pow", std::vector<NodeArg*>{cast_out, pow_const}, std::vector<NodeArg*>{pow_out});
  builder.AddNode("ReduceMean", std::vector<NodeArg*>{pow_out}, std::vector<NodeArg*>{reduce_mean_out_2})
      .AddAttribute("axes", std::vector<int64_t>{-1});
  builder.AddNode("Add", std::vector<NodeArg*>{reduce_mean_out_2, add_initializer}, std::vector<NodeArg*>{add_out});
  builder.AddNode("Sqrt", std::vector<NodeArg*>{add_out}, std::vector<NodeArg*>{sqrt_out});
  builder.AddNode("Div", std::vector<NodeArg*>{cast_out, sqrt_out}, std::vector<NodeArg*>{div_out});
  builder.AddNode("Cast", std::vector<NodeArg*>{div_out}, std::vector<NodeArg*>{cast_back_fp16})
      .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
  builder.AddNode("Mul", std::vector<NodeArg*>{cast_back_fp16, weight_initializer}, std::vector<NodeArg*>{mul_out});

  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  // Save model to a temp directory.
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("simplified_layer_norm_fp16_const")};
  PathString model_path = ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("model.onnx"));
  ASSERT_STATUS_OK(Model::Save(model, model_path));

  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::MaxLevel;  // enables Level2/3 and InsertCast/Level4 path
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_path));
  // Should not crash during Initialize with default graph_optimizations_loop_level (=1).
  ASSERT_STATUS_OK(session.Initialize());
}

TEST(TransformerTest, FuseFp16InitializersWithGraphOutputs) {
  if (MlasFp16AccelerationSupported()) {
    GTEST_SKIP() << "Skipping test because FP16 acceleration support was detected.";
  }

  auto model_uri = MODEL_FOLDER ORT_TSTR("fuse_fp16_initializers_with_graph_outputs.onnx");  // Graph for this model is drawn at beginning of this file
  RandomValueGenerator random{123};
  std::vector<int64_t> a_dims{1, 1, 5, 5};
  std::vector<float> a_data = random.Gaussian<float>(a_dims, 0.0f, 1.0f);
  OrtValue a_fp32;
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                       a_dims, a_data, &a_fp32);
  NameMLValMap inputs{{"A", a_fp32}};

  std::vector<std::string> output_names{"B", "C"};
  std::vector<OrtValue> outputs;

  auto _graph_structure_at_load = [](const Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(1, op_to_count["Add"]);
    ASSERT_EQ(1, op_to_count["Cast"]);
    auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
    ASSERT_EQ(1, num_initializers_fp16);
  };

  auto _graph_structure_at_initialized = [](const Graph& graph) {
    auto op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(1, op_to_count["Add"]);
    ASSERT_EQ(1, op_to_count["Cast"]);
    // Note: Unable to calc no. of fp16/fp32 initializers, as when session
    // state is finalized after init it removes initializers from graph.
    // Look for "session_state_->FinalizeSessionState" method in
    // inference_session.cc for more explanation.
  };

  // set session options
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::MaxLevel;
  // Create session and check graph before / after initiation
  InferenceSessionWrapper session{so, GetEnvironment()};
  // Disabling ConstantFolding optimizer as it will remove the Cast node
  // by folding it with Add node. This will not allow us to test the
  // scenario where Cast node is producing graph output and need to
  // kept untouched by FuseInitializersTransformer.
  ASSERT_STATUS_OK(session.FilterEnabledOptimizers({"ConstantFolding"}));
  ASSERT_STATUS_OK(session.Load(model_uri));
  _graph_structure_at_load(session.GetGraph());
  ASSERT_STATUS_OK(session.Initialize());
  _graph_structure_at_initialized(session.GetGraph());
  ASSERT_STATUS_OK(session.Run(inputs, output_names, &outputs));
}  // FuseFp16InitializersWithGraphOutputs

}  // namespace test
}  // namespace onnxruntime
