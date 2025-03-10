/*
 * Copyright (c) Intel Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include "gtest/gtest.h"
#include "test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/asserts.h"
#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/fuse_initializers_transformer.h"

namespace onnxruntime {

    namespace test {

        #define MODEL_FOLDER ORT_TSTR("testdata/transform/")

        unsigned int CountNoOfInitializersInGraph(const Graph& graph, const onnxruntime::MLDataType _data_type) {

            //init
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

            //init
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
            EXPECT_TRUE(0==op_to_count["Cast"]);
            // Count no. of initializers of FP16 type
            auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
            EXPECT_TRUE(2==num_initializers_fp16);
            // Count no. of initializers of FP32 type
            auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
            EXPECT_TRUE(0==num_initializers_fp32);
            // Count no. of FP16 nodes
            auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
            EXPECT_TRUE(1==num_nodes_fp16);
            // Count no. of FP32 nodes
            auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
            EXPECT_TRUE(0==num_nodes_fp32);
            // Check if all conditions are met
            ASSERT_TRUE((0==op_to_count["Cast"])
                     && (2==num_initializers_fp16)
                     && (0==num_initializers_fp32)
                     && (1==num_nodes_fp16)
                     && (0==num_nodes_fp32));
        }

        void test_graph_structure_before_fusion(const Graph& graph) {
            // Count ops
            auto op_to_count = CountOpsInGraph(graph);
            EXPECT_TRUE(4==op_to_count["Cast"]);
            // Count no. of initializers of FP16 type
            auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
            EXPECT_TRUE(2==num_initializers_fp16);
            // Count no. of initializers of FP32 type
            auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
            EXPECT_TRUE(0==num_initializers_fp32);
            // Count no. of FP16 nodes
            auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
            EXPECT_TRUE(3==num_nodes_fp16);
            // Count no. of FP32 nodes
            auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
            EXPECT_TRUE(2==num_nodes_fp32);
            // Check if all conditions are met
            ASSERT_TRUE((4==op_to_count["Cast"])
                     && (2==num_initializers_fp16)
                     && (0==num_initializers_fp32)
                     && (3==num_nodes_fp16)
                     && (2==num_nodes_fp32));
        }

        void test_graph_structure_after_fusion(const Graph& graph) {
            // Count ops
            auto op_to_count = CountOpsInGraph(graph);
            EXPECT_TRUE(2==op_to_count["Cast"]);
            // Count no. of initializers of FP16 type
            auto num_initializers_fp16 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
            EXPECT_TRUE(0==num_initializers_fp16);
            // Count no. of initializers of FP32 type
            auto num_initializers_fp32 = CountNoOfInitializersInGraph(graph, DataTypeImpl::GetTensorType<float>());
            EXPECT_TRUE(2==num_initializers_fp32);
            // Count no. of FP16 nodes
            auto num_nodes_fp16 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<MLFloat16>());
            EXPECT_TRUE(1==num_nodes_fp16);
            // Count no. of FP32 nodes
            auto num_nodes_fp32 = CountNoOfNodesInGraph(graph, DataTypeImpl::GetTensorType<float>());
            EXPECT_TRUE(2==num_nodes_fp32);
            // Check if all conditions are met
            ASSERT_TRUE((2==op_to_count["Cast"])
                     && (0==num_initializers_fp16)
                     && (2==num_initializers_fp32)
                     && (1==num_nodes_fp16)
                     && (2==num_nodes_fp32));
        }

        TEST(TransformerTest, FuseFp16InitializersWithFp32Node) {

            // Init
            auto test_logger = DefaultLoggingManager().DefaultLogger();
            auto model_uri = MODEL_FOLDER ORT_TSTR("fuse_fp16_initializers.onnx");
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
            if(graph_modified_by_insert_cast_transforms) {
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
            if(graph_modified_by_insert_cast_transforms) {

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

        } // FuseFp16InitializersWithFp32Node

    }  // namespace test
}  // namespace onnxruntime
