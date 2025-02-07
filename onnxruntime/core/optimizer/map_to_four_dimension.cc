// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/optimizer/map_to_four_dimension.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types.h"

using namespace onnxruntime::common;

namespace onnxruntime {

MapToFourDimensions::MapToFourDimensions(const IExecutionProvider& execution_provider,
                                       bool skip_dequantize_linear,
                                       bool dequantize_initializer_for_dequantize_linear,
                                       const ConfigOptions& config_options,
                                       const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                       const InlinedHashSet<std::string>& excluded_initializers) noexcept
    : GraphTransformer("MapToFourDimensions", compatible_execution_providers),
      skip_dequantize_linear_(skip_dequantize_linear),
      dequantize_initializer_for_dequantize_linear_(dequantize_initializer_for_dequantize_linear),
      config_options_(config_options),
      excluded_initializers_(excluded_initializers),
      execution_provider_(execution_provider) {
}

onnxruntime::NodeArg* AddSliceReduceConcatNodes(onnxruntime::Graph& graph,
                                                onnxruntime::Node& reshape,
                                                onnxruntime::Node& reduce_sum,
                                                onnxruntime::NodeArg* old_arg,
                                                ONNX_NAMESPACE::TypeProto* new_type,
                                                bool new_on_input,
                                                int64_t to_type,
                                                onnxruntime::ProviderType providerType) {
  // Insert 2 Slice nodes, 2 ReduceSum nodes and 1 Concat node.

  // Create 2 Slice nodes
  std::string slice_node_0_name = graph.GenerateNodeName(reshape.Name() + "_slice_0");
  std::string slice_node_1_name = graph.GenerateNodeName(reshape.Name() + "_slice_1");

  // The Slice node type should be the same as the Reshape node (going to be removed) type.
  auto* slice_node_0_arg = &graph.GetOrCreateNodeArg(slice_node_0_name, reshape.OutputDefs()[0]->TypeAsProto());
  auto* slice_node_1_arg = &graph.GetOrCreateNodeArg(slice_node_1_name, reshape.OutputDefs()[0]->TypeAsProto());

  std::vector<onnxruntime::NodeArg*> slice_node_0_output_defs = {slice_node_0_arg};
  std::vector<onnxruntime::NodeArg*> slice_node_1_output_defs = {slice_node_1_arg};

  auto& slice_node_0 = graph.AddNode(slice_node_0_name, "Slice", "Map 5D/6D to 4D",
                                  reshape.MutableInputDefs(), slice_node_0_output_defs);
  auto& slice_node_1 = graph.AddNode(slice_node_1_name, "Slice", "Map 5D/6D to 4D",
                                    reshape.MutableInputDefs(), slice_node_1_output_defs);

  // Create 2 ReduceSum nodes
  std::string reduce_sum_node_0_name = graph.GenerateNodeName(reduce_sum.Name() + "_0");
  std::string reduce_sum_node_1_name = graph.GenerateNodeName(reduce_sum.Name() + "_1");

  // The ReduceSum node type should be the same as the original ReduceSum node (going to be removed) type.
  auto* reduce_sum_node_0_arg = &graph.GetOrCreateNodeArg(reduce_sum_node_0_name, reduce_sum.OutputDefs()[0]->TypeAsProto());
  auto* reduce_sum_node_1_arg = &graph.GetOrCreateNodeArg(reduce_sum_node_1_name, reduce_sum.OutputDefs()[0]->TypeAsProto());

  std::vector<onnxruntime::NodeArg*> reduce_sum_node_0_output_defs = {reduce_sum_node_0_arg};
  std::vector<onnxruntime::NodeArg*> reduce_sum_node_1_output_defs = {reduce_sum_node_1_arg};

  auto& reduce_sum_node_0 = graph.AddNode(reduce_sum_node_0_name, "ReduceSum", "Map 5D/6D to 4D",
                                    slice_node_0_output_defs, reduce_sum_node_0_output_defs);
  auto& reduce_sum_node_1 = graph.AddNode(reduce_sum_node_1_name, "ReduceSum", "Map 5D/6D to 4D",
                                    slice_node_1_output_defs, reduce_sum_node_1_output_defs);

  // Create 1 Concat
  std::string concat_node_name = graph.GenerateNodeName(reduce_sum_node_0_name + "_concat");
  auto* concat_node_arg = &graph.GetOrCreateNodeArg(concat_node_name, reduce_sum.OutputDefs()[0]->TypeAsProto());
  std::vector<onnxruntime::NodeArg*> concat_node_arg_input_defs = {reduce_sum_node_0_arg, reduce_sum_node_1_arg};
  std::vector<onnxruntime::NodeArg*> concat_node_arg_output_defs = {concat_node_arg};
  auto& concat_node = graph.AddNode(concat_node_name, "Concat", "Map 5D/6D to 4D",
                                    concat_node_arg_input_defs, concat_node_arg_output_defs);

  return concat_node_arg;
}

Status MapToFourDimensions::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  bool have_updated_nodes = false;
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }
    
    bool map_tensor_to_4d = false;

    // Requirements:
    // 1. Map 2D/3D to 4D. Replace 2D Gemms with Transpose/Reshape and 1x1 Conv.
    // 2. Map 5D/6D to 4D
    if (node->OpType() == "MatMul") {
      const auto* input_0 = node->InputDefs()[0];
      const auto* input_1 = node->InputDefs()[1];
      if ((input_0->Shape()->dim_size() == 2 || input_0->Shape()->dim_size() == 3) &&
          (input_1->Shape()->dim_size() == 2 || input_1->Shape()->dim_size() == 3)) {
        //map_tensor_to_4d = true;
      }
    } else if (node->OpType() == "ReduceSum") {
      // assume Reshape -> Q -> DQ -> ReduceSum since we don't remove Q/DQ for now
      // TODO: Make sure Reshape, Q and DQ does exist
      const Node& node_x = *node->InputNodesBegin();  // Q
      const Node& node_y = *node_x.InputNodesBegin(); // DQ
      const Node& node_z = *node_y.InputNodesBegin(); // Reshape
      if (node_z.OpType() == "Reshape") {
        const auto* output_0 = node_z.OutputDefs()[0];
        if (output_0->Shape()->dim_size() == 5) {
          map_tensor_to_4d = true;
        }
      }
    }

    if (!map_tensor_to_4d) {
      continue;
    }

    if (node->OpType() == "MatMul") {
      const auto* input_0 = node->InputDefs()[0]; // X
      const auto* input_1 = node->InputDefs()[1]; // W 
    } else if (node->OpType() == "ReduceSum") {
      // assume Reshape -> Q -> DQ -> ReduceSum since we don't remove Q/DQ for now
      // TODO: Make sure Reshape, Q and DQ does exist
      const Node& node_x = *node->InputNodesBegin();   // Q
      const Node& node_y = *node_x.InputNodesBegin();  // DQ
      const Node& node_z = *node_y.InputNodesBegin();  // Reshape

      const auto* input_0 = node_z.InputDefs()[0];
      const auto* output_0 = node_z.OutputDefs()[0];
    }

  }

  return Status::OK();
}
}  // namespace onnxruntime
