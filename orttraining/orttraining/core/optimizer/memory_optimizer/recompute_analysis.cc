// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/transformer_specific.h"
#include "orttraining/core/optimizer/memory_optimizer/recompute_analysis.h"
#include "core/common/string_utils.h"
#include "core/framework/data_types.h"
#include "core/optimizer/utils.h"

namespace onnxruntime::optimizer::memory_optimizer {

namespace {

constexpr int32_t MAXIMUM_RECOMPUTE_NODE_COUNT = 50;

static size_t GetElementSize(const ONNX_NAMESPACE::DataType& tensor_type) {
  const ONNX_NAMESPACE::TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(tensor_type);
  MLDataType ml_data_type = DataTypeImpl::TypeFromProto(type_proto);
  const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
  ORT_ENFORCE(nullptr != tensor_type_base);
  MLDataType elt_type = tensor_type_base->GetElementType();
  return elt_type->Size();
}

// TODO(pengwa): extent this function to be more general.
float InputOutputSizeRatio(const Node* node) {
  if (node->OpType().compare("Cast") == 0) {
    const NodeArg* input = node->InputDefs()[0];
    const NodeArg* output = node->OutputDefs()[0];
    if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING ||
        output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      return 1.0f;
    }
    const auto& ptype1 = input->Type();
    const auto& ptype2 = output->Type();
    float ratio = static_cast<float>(GetElementSize(ptype1)) / static_cast<float>(GetElementSize(ptype2));
    return ratio;
  }

  return 1.0f;
}

using IgnorableInputIndices = InlinedVector<int>;
using OpsetToIgnorableIndicesMap = InlinedHashMap<int, IgnorableInputIndices>;

/**
 * @brief Get the Allowed Recompute Ops object
 *
 * The supported op types are predefined.
 * Most recent revisited for ONNX v1.15.0 release - https://github.com/onnx/onnx/blob/b86cc54efce19530fb953e4b21f57e6b3888534c/docs/Operators.md
 *
 * We defined supported list explicitly instead of using a excluding list for the following reasons:
 * 1. Some ops generate indeterministic results (for example using random number generator). We need evaluate whether
 *   this is a problem for recompute before adding the support, instead of fixing this after we find and try to
 *   fix convergence issues (which will be very hard if we have multiple indeterministic operators by default supported.)
 * 2. Some ops schema will be changed in new opsets, we need also check manually whether it is applicable to recompute
 *   or not.
 * 3. Some ops are not supported in older opsets, we need to check whether it is applicable to recompute or not.
 */
const InlinedHashMap<std::string, OpsetToIgnorableIndicesMap>& GetAllowedRecomputeOps(int probe_op_level) {
  static InlinedHashMap<int, InlinedHashMap<std::string, OpsetToIgnorableIndicesMap>> recomputable_op_table_map;
  if (recomputable_op_table_map.find(probe_op_level) != recomputable_op_table_map.end()) {
    return recomputable_op_table_map.at(probe_op_level);
  }

  recomputable_op_table_map.insert({probe_op_level, InlinedHashMap<std::string, OpsetToIgnorableIndicesMap>()});
  auto& recomputable_op_table = recomputable_op_table_map.at(probe_op_level);
  if (probe_op_level >= static_cast<int>(ProbeLevel::Basic)) {
    recomputable_op_table.insert({
        {
            utils::GetFullQualifiedOpName("Add", kOnnxDomain),
            {
                {1, {}},
                {6, {}},
                {7, {}},
                {13, {}},
                {14, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("BatchNormalization", kOnnxDomain),
            {
                {1, {}},
                {6, {}},
                {7, {}},
                {9, {}},
                {14, {}},
                {15, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("BiasGelu", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("BiasDropout", kMSDomain),
            {
                {1, {3, 4}},  // ignore ratio (optional) and training mode (optional)
            },
        },
        {
            utils::GetFullQualifiedOpName("BitmaskBiasDropout", kMSDomain),
            {
                {1, {3, 4}},  // ignore ratio (optional) and training mode (optional)
            },
        },
        {
            utils::GetFullQualifiedOpName("BitmaskDropout", kMSDomain),
            {
                {1, {1, 2}},  // ignore ratio (optional) and training mode (optional)
            },
        },
        {
            utils::GetFullQualifiedOpName("Cast", kOnnxDomain),
            {
                {1, {}},
                {6, {}},
                {9, {}},
                {13, {}},
                {19, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("ConcatTraining", kMSDomain),
            {
                {1, {}},

            },
        },
        {
            utils::GetFullQualifiedOpName("ConstantOfShape", kOnnxDomain),
            {
                {9, {0}},  // ignore the `input`, e.g. the shape of the expected output tensor
                {20, {0}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Dropout", kOnnxDomain),
            {
                // ONNX Dropout 1, 6, 7, 10 do not have seed attribute, so we remove them from the recompute support.
                {12, {1, 2}},  // ignore ratio and training_mode
                {13, {1, 2}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Div", kOnnxDomain),
            {
                {1, {}},
                {6, {}},
                {7, {}},
                {13, {}},
                {14, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Expand", kOnnxDomain),
            {
                {8, {1}},  // Ignore the shape.
                {13, {1}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Cos", kOnnxDomain),
            {
                {7, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("CumSum", kOnnxDomain),
            {
                // The axis input is trivial
                {11, {1}},
                {14, {1}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Einsum", kOnnxDomain),
            {
                {12, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Equal", kOnnxDomain),
            {
                {1, {}},
                {7, {}},
                {11, {}},
                {13, {}},
                {19, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("FastGelu", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Gather", kOnnxDomain),
            {
                {1, {1}},  // ignore the indices
                {11, {1}},
                {13, {1}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Gelu", kOnnxDomain),
            {
                {20, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Gelu", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Less", kOnnxDomain),
            {
                {1, {}},
                {7, {}},
                {9, {}},
                {13, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Mul", kOnnxDomain),
            {
                {1, {}},
                {6, {}},
                {7, {}},
                {13, {}},
                {14, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Range", kOnnxDomain),
            {
                {11, {0, 1, 2}},  // ignore start, end, delta, because they are scalars.
            },
        },
        {
            utils::GetFullQualifiedOpName("Reshape", kOnnxDomain),
            {
                {1, {}},
                {5, {}},  // ignore the shape.
                {13, {}},
                {14, {}},
                {19, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Sin", kOnnxDomain),
            {
                {7, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Slice", kOnnxDomain),
            {
                {1, {}},
                {10, {1, 2, 3, 4}},  // ignore starts, ends, axes (optional) and steps (optional)
                {11, {1, 2, 3, 4}},
                {13, {1, 2, 3, 4}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Split", kOnnxDomain),
            {
                {1, {1}},  // ignore split (optional)
                {2, {}},
                {11, {}},
                {13, {1}},  // ignore the split (optional)
                {18, {1}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Squeeze", kOnnxDomain),
            {
                {1, {}},
                {11, {}},
                {13, {1}},  // ignore the axes (optional)
            },
        },
        {
            utils::GetFullQualifiedOpName("Sub", kOnnxDomain),
            {
                {1, {}},
                {6, {}},
                {7, {}},
                {13, {}},
                {14, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Tile", kOnnxDomain),
            {
                {1, {1, 2}},
                {6, {1}},
                {13, {1}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Transpose", kOnnxDomain),
            {
                {1, {}},
                {13, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Trilu", kOnnxDomain),
            {
                {14, {1}},  // ignore k (optional)
            },
        },
        {
            utils::GetFullQualifiedOpName("QuickGelu", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Unsqueeze", kOnnxDomain),
            {
                {1, {}},
                {11, {}},
                {13, {1}},  // ignore the axes (optional)
            },
        },
        {
            utils::GetFullQualifiedOpName("Where", kOnnxDomain),
            {
                {9, {}},
                {16, {}},
            },
        },

    });
  }

  if (probe_op_level >= static_cast<int>(ProbeLevel::Advanced)) {
    recomputable_op_table.insert({
        {
            utils::GetFullQualifiedOpName("BiasSoftmax", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("BiasSoftmaxDropout", kMSDomain),
            {
                {1, {2}},  // ignore ratio (optional)
            },
        },
        {
            utils::GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
            {
                // Opset 1 in ONNX official does not have LayerNormalization,
                // while our contrib op defined LayerNormalization in opset 1 in ONNX domain.
                {1, {}},
                {17, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("MatMul", kOnnxDomain),
            {
                {1, {}},
                {9, {}},
                {13, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("FusedMatMul", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("SimplifiedLayerNormalization", kOnnxDomain),
            {
                // Opset 1 in ONNX official does not have SimplifiedLayerNormalization,
                // while our contrib op defined SimplifiedLayerNormalization in opset 1 in ONNX domain.
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("SkipLayerNormalization", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("SkipSimplifiedLayerNormalization", kMSDomain),
            {
                {1, {}},
            },
        },
        {
            utils::GetFullQualifiedOpName("Softmax", kOnnxDomain),
            {
                {1, {}},
                {11, {}},
                {13, {}},
            },
        },
    });
  }

  return recomputable_op_table;
}

/**
 * @brief Check whether a node is a recomputable node at given probe level.
 */
bool IsRecomputable(const Node& node, ProbeLevel probe_level) {
  const InlinedHashMap<std::string, OpsetToIgnorableIndicesMap>& op_table = GetAllowedRecomputeOps(static_cast<int>(probe_level));
  auto it = op_table.find(utils::GetFullQualifiedOpName(node.OpType(), node.Domain()));
  if (it == op_table.end()) {
    return false;
  }
  return it->second.count(node.SinceVersion());
}

const InlinedVector<int>& GetIgnorableInputIndices(const Node& node, ProbeLevel probe_level) {
  const InlinedHashMap<std::string, OpsetToIgnorableIndicesMap>& op_table = GetAllowedRecomputeOps(static_cast<int>(probe_level));
  auto it = op_table.find(utils::GetFullQualifiedOpName(node.OpType(), node.Domain()));
  ORT_ENFORCE(it != op_table.end(), "Cannot get ignorable indices since the node type is supported in the list.");
  ORT_ENFORCE(it->second.count(node.SinceVersion()) > 0, "Cannot get ignorable indices since the opset is supported");
  return it->second.at(node.SinceVersion());
}

/**
 * @brief Find recomputable subgraphs (has at least one nodes, at most MAXIMUM_RECOMPUTE_NODE_COUNT nodes).
 *
 * @param entry_node The entry node to start the subgraph matching (bottom-up), usually the last node of found subgraphs.
 * @param probe_config The probe config to control recomputable subgraph detecting.
 * @param node_output_index_candidates Candidate output indices of "node", which are consumed by both fw and bw ops.
 * @param fw_op_output_arg_used_map The activation usage (in fw and bw) mapping.
 * @param node_index_to_its_order_in_topological_sort_map The mapping of node index to its order in topological sort.
 *   Used to re-order the collected subgraph nodes.
 * @param nodes_in_topological_order Collected vector of nodes of found subgraph, in the order of the topological
 *  sorted.
 * @param logger Logger.
 * @param compromise_stashed_activation Whether to compromise stashed activation, e.g. if we cannot find a
 * recomputable subgraph to save a stashed activation, we can compromise to find a recomputable subgraph to reduce the
 * size of stashed activation.
 * @param can_compromise_stashed_activation A bool return value, to indicate there are opportunities for finding a
 * compromised subgraph.
 * @param save_ratio The ratio of memory saving if we can find a recomputable subgraph.
 * @return Status
 */
Status SelectRecomputeSubgraph(const Node& entry_node,
                               const ProbeConfig& probe_config,
                               const InlinedVector<size_t>& node_output_index_candidates,
                               const ActivationUsedMap& fw_op_output_arg_used_map,
                               const InlinedHashMap<NodeIndex, ptrdiff_t>&
                                   node_index_to_its_order_in_topological_sort_map,
                               const logging::Logger& logger,
                               InlinedVector<const Node*>& nodes,
                               bool compromise_stashed_activation,
                               bool& can_compromise_stashed_activation,
                               float& save_ratio) {
  const ProbeLevel probe_level = probe_config.probe_level;

  can_compromise_stashed_activation = false;

  MO_LOG_DEBUG_INFO(logger, "Enter SelectRecomputeSubgraph for Node " + entry_node.Name() +
                                "(" + entry_node.OpType() + ")");
  nodes.clear();

  std::deque<NodeOutputPort> q;
  for (auto output_index : node_output_index_candidates) {
    q.push_back(NodeOutputPort(&entry_node, output_index));
  }

  bool early_stop = false;
  std::set<NodeOutputPort> visited_output_arg_set;
  std::set<const Node*> visited_node_set;

  // For the initial activations in queue, they are stashed ones, so we do differently when scanning the queue for them.
  bool is_first_queue_scan = true;
  while (nodes.size() < MAXIMUM_RECOMPUTE_NODE_COUNT && !q.empty() && !early_stop) {
    // Loop all candidate NodeOutputPort, and find the next layer of input nodes.
    size_t current_queue_size = q.size();
    for (size_t i = 0; i < current_queue_size; ++i) {
      NodeOutputPort p = q.front();
      q.pop_front();
      const Node* curr_node = p.first;

      // Skip if the node output is already visited.
      if (std::find(visited_output_arg_set.begin(), visited_output_arg_set.end(), p) !=
          visited_output_arg_set.end()) {
        continue;
      }

      visited_output_arg_set.insert({p});

      // If the node is already visited by from its other output index, skip it.
      if (visited_node_set.find(curr_node) != visited_node_set.end()) {
        continue;
      }

      visited_node_set.insert(curr_node);

      // Bottom-up search rules.
      // If current op is entry output node (that generates stashed activations):
      //   1. If the op is not in recomputable_op_table, skip it.
      // Otherwise:
      //  If current op is in allowed list, check its input args, and append the producers' NodeOutputPorts to next_q.
      //  If current op is NOT in allowed list:
      //    1). the output does not exist in backward, we cannot find a good solution for so, the search terminates.
      //    2). the output is used in backward, we don't need to trace back further, so continue searching.
      bool is_recomputable = IsRecomputable(*curr_node, probe_level);
      auto cur_output_arg_name = curr_node->OutputDefs()[p.second]->Name();
      if (is_first_queue_scan) {
        // We handle the entry node outputs differently because, we don't want this case falls into and succeed one of
        // the checks in the other branch
        // 1. "op is not in recompute op list, but its output is used in backward"
        // 2. "op is in recompute op list, but its output is used in backward"
        // (either of the above checks is true for entry node outputs)
        if (!is_recomputable) {
          early_stop = true;
          MO_LOG_DEBUG_INFO(logger, "Entry Node " + curr_node->Name() + "(" + curr_node->OpType() +
                                        ") is **NOT** in recompute op list, search terminates.");
          break;
        }
      } else {
        if (!is_recomputable) {
          if (fw_op_output_arg_used_map.at(cur_output_arg_name).second) {
            MO_LOG_DEBUG_INFO(logger, "Node " + curr_node->Name() + "(" + curr_node->OpType() +
                                          ") is **NOT** in recompute op list, but its output [" +
                                          cur_output_arg_name +
                                          "] is used in backward, we don't need trace bottom-up further. Entry node: " +
                                          entry_node.Name() + "(" + entry_node.OpType() + ")");
            continue;
          } else {
            early_stop = true;
            MO_LOG_DEBUG_INFO(logger, "Node " + curr_node->Name() + "(" + curr_node->OpType() + ") is **NOT** in " +
                                          "recompute op list, and its output [" + cur_output_arg_name +
                                          "] does not exist in backward, search terminates. Entry node: " +
                                          entry_node.Name() + "(" + entry_node.OpType() + ")");
            break;
          }
        }

        if (fw_op_output_arg_used_map.at(cur_output_arg_name).second) {
          MO_LOG_DEBUG_INFO(logger, "Node " + curr_node->Name() + "(" + curr_node->OpType() + ") " +
                                        "is in recompute op list, while its output [" + cur_output_arg_name +
                                        "] is used in backward, we don't need trace bottom-up further. Entry node: " +
                                        entry_node.Name() + "(" + entry_node.OpType() + ")");
          continue;
        }
      }

      // Append node to the selected graph.
      if (std::find(nodes.begin(), nodes.end(), curr_node) == nodes.end()) {
        nodes.push_back(curr_node);
        MO_LOG_DEBUG_INFO(logger, "Node " + curr_node->Name() + "(" + curr_node->OpType() +
                                      ") is added in selected subgraph");
      }

      // This check is not matured now, subject to change.
      float ratio = InputOutputSizeRatio(curr_node);
      float saving_ratio = 1.0f - ratio;
      float is_current_node_compromisable = (ratio < 1.f);
      can_compromise_stashed_activation = can_compromise_stashed_activation || is_current_node_compromisable;
      if (is_current_node_compromisable) {
        MO_LOG_DEBUG_INFO(logger, "Node " + curr_node->Name() + "(" + curr_node->OpType() +
                                      ") has input/output size " + std::to_string(ratio) +
                                      " < 1.f, can compromise stashed activation");
      }

      if (is_current_node_compromisable && compromise_stashed_activation) {
        MO_LOG_DEBUG_INFO(logger, "Node " + curr_node->Name() + "(" + curr_node->OpType() + ") is in " +
                                      "recompute op list, and its output [" + cur_output_arg_name +
                                      "] does not exist in backward, while it meets compromised check, we don't need trace " +
                                      "bottom-up further.");
        save_ratio = saving_ratio;
        continue;
      }

      // Iterate all input nodes according to allowed input arg index of the entry node.
      const auto& igorable_input_arg_indices = GetIgnorableInputIndices(*curr_node, probe_level);
      for (auto it = curr_node->InputEdgesBegin(), end = curr_node->InputEdgesEnd(); it != end; ++it) {
        const Node::EdgeEnd& input_edge = *it;
        const auto& parent_node = input_edge.GetNode();
        const auto parent_node_output_index = input_edge.GetSrcArgIndex();
        const auto current_node_input_index = input_edge.GetDstArgIndex();
        if (std::find(igorable_input_arg_indices.begin(), igorable_input_arg_indices.end(), current_node_input_index) ==
            igorable_input_arg_indices.end()) {
          // If the tensor size is constant and very small (Now < 1M), we stop adding the input edge into queue.
          auto output_shape = parent_node.OutputDefs()[parent_node_output_index]->Shape();
          if (output_shape) {
            bool all_constant_dim = true;
            int64_t num_elem = 1;
            for (int k = 0, dim_size = output_shape->dim_size(); k < dim_size; ++k) {
              if (!output_shape->dim(k).has_dim_value()) {
                all_constant_dim = false;
                num_elem *= output_shape->dim(k).dim_value();
              }
            }
            if (all_constant_dim && num_elem < 1 * 1024 * 1024) {
              // Skip this input index.
              continue;
            }
          }
          NodeOutputPort next_p = std::make_pair(&parent_node, parent_node_output_index);

          MO_LOG_DEBUG_INFO(logger, "Node " + parent_node.Name() + "(" + parent_node.OpType() + ")'s " +
                                        std::to_string(parent_node_output_index) + "th output [" +
                                        parent_node.OutputDefs()[parent_node_output_index]->Name() +
                                        "] is added in recompute search list");

          q.push_back(next_p);
        }
      }
    }
    // After handling all entry node outputs, we set the flag to false.
    is_first_queue_scan = false;
  }

  // If input args are not found in bw, but op count exceed MAXIMUM_RECOMPUTE_NODE_COUNT, skip recompute.
  if (!q.empty() || early_stop) {
    MO_LOG_DEBUG_INFO(logger, "Fail to find a solution for recompute: current node count is " +
                                  std::to_string(nodes.size()) + ", queue size: " + std::to_string(q.size()) +
                                  ", early stop: " + std::to_string(early_stop));
    nodes.clear();
  } else {
    // Re-order the nodes in topological order.
    std::sort(nodes.begin(), nodes.end(),
              [&node_index_to_its_order_in_topological_sort_map](const Node*& lhs, const Node*& rhs) {
                return node_index_to_its_order_in_topological_sort_map.at(lhs->Index()) <
                       node_index_to_its_order_in_topological_sort_map.at(rhs->Index());
              });
  }
  return Status::OK();
}

/**
 * @brief Convert the recompute subgraph to its string representation.
 *
 * @param nodes_in_topological_order The subgraph nodes in topological order.
 * @param subgraph_string_representation Returns subgraph string representation.
 * @param log_info Returns log info for users.
 */
void NodesInTopoOrderToString(gsl::span<const Node* const> nodes_in_topological_order,
                              std::string& subgraph_string_representation,
                              std::string& log_info) {
  std::ostringstream oss;
  std::ostringstream subgraph_string_representation_oss;
  size_t node_count = nodes_in_topological_order.size();
  for (size_t i = 0; i < node_count; ++i) {
    if (i < node_count - 1) {  // Ignore the last node.
      oss << "(name:" << nodes_in_topological_order[i]->Name() << ", type:" << nodes_in_topological_order[i]->OpType()
          << "),";
    }

    subgraph_string_representation_oss << nodes_in_topological_order[i]->OpType() << "+";
  }

  subgraph_string_representation = subgraph_string_representation_oss.str();
  log_info = oss.str();
  if (log_info.size() > 0) {
    log_info = " with its precedent nodes: " + log_info;
  }
}

}  // namespace

Status ParseProbeConfigFromString(std::string_view recompute_probe_config, ProbeConfig& probe_config) {
  int transformer_layer_as_boundary = 0;
  if (!recompute_probe_config.empty()) {
    const auto probe_configs = utils::SplitString(recompute_probe_config, ":");
    ORT_ENFORCE(probe_configs.size() >= 1, "Probe config information is not complete.");
    int probe_level_int = ParseIntValueFromString(probe_configs[0]);
    ORT_ENFORCE(probe_level_int <
                        static_cast<int>(ProbeLevel::LevelMax) &&
                    probe_level_int >= 0,
                "Invalid probe level specified: ", probe_configs[0]);

    if (probe_configs.size() > 1) {
      transformer_layer_as_boundary = ParseIntValueFromString(probe_configs[1]);
      ORT_ENFORCE(transformer_layer_as_boundary == 0 || transformer_layer_as_boundary == 1,
                  "Invalid transformer_layer_as_boundary specified: ", probe_configs[1]);
    }

    probe_config.probe_level = static_cast<ProbeLevel>(probe_level_int);
  }

  probe_config.enable_transformer_layer_as_boundary = transformer_layer_as_boundary == 1;

  return Status::OK();
}

std::unique_ptr<NodeRecomputePlan> CheckNodeForRecompute(const GraphViewer& graph_viewer,
                                                         const Node& node,
                                                         const ProbeConfig& probe_config,
                                                         const ActivationUsedMap& fw_op_output_arg_used_map,
                                                         const InlinedHashMap<NodeIndex, ptrdiff_t>&
                                                             node_index_to_its_order_in_topological_sort_map,
                                                         const InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                                             candidate_output_args_map,
                                                         const InlinedVector<const Node*>& layer_boundary_ln_nodes,
                                                         const logging::Logger& logger,
                                                         bool compromise_stashed_activation,
                                                         bool& can_compromise_stashed_activation) {
  if (!IsRecomputable(node, probe_config.probe_level)) {
    return nullptr;
  }

  if (probe_config.enable_transformer_layer_as_boundary) {
    // Check whether the node's stashed activation outputs are used by LayerNormalization's inputs.
    // If yes, for Transformers, we don't need to recompute the node, because we treated
    // LayerNormalization of Attention as the boundary for subgraph searching.
    // Check at least one of the stashed activation output is used as the 1st input
    // of LayerNormalization, e.g. will be used as input of LayerNormalizationGrad.
    for (auto& output_index : candidate_output_args_map.at(&node)) {
      auto output_name = node.OutputDefs()[output_index]->Name();
      auto consumers = graph_viewer.GetConsumerNodes(output_name);
      for (auto& consumer : consumers) {
        if (std::find(layer_boundary_ln_nodes.begin(), layer_boundary_ln_nodes.end(), consumer) !=
            layer_boundary_ln_nodes.end()) {
          int dest_in_index = optimizer_utils::IndexOfNodeInput(*consumer, *node.OutputDefs()[output_index]);
          if (dest_in_index == 0) {
            MO_LOG_DEBUG_INFO(logger, "Node " + node.Name() + "(" + node.OpType() +
                                          ") is a Attention+MLP layer boundary node, " +
                                          "its stashed activation outputs are used by LayerNormalization's inputs, " +
                                          "we don't need to recompute it.");
            return nullptr;
          }
        }
      }
    }
  }

  InlinedVector<const Node*> nodes_in_topological_order;
  float save_ratio = 1.f;
  ORT_ENFORCE(SelectRecomputeSubgraph(node,
                                      probe_config,
                                      candidate_output_args_map.at(&node),
                                      fw_op_output_arg_used_map,
                                      node_index_to_its_order_in_topological_sort_map,
                                      logger,
                                      nodes_in_topological_order,
                                      compromise_stashed_activation,
                                      can_compromise_stashed_activation,
                                      save_ratio)
                  .IsOK());
  if (nodes_in_topological_order.size() == 0) {
    return nullptr;
  }

  std::string subgraph_str_representation, log_info;
  NodesInTopoOrderToString(nodes_in_topological_order, subgraph_str_representation, log_info);

  MO_LOG_DEBUG_INFO(logger, "Node " + node.Name() + "(" + node.OpType() + ") can be recomputed" + log_info);

  return std::make_unique<NodeRecomputePlan>(&node, candidate_output_args_map.at(&node),
                                             nodes_in_topological_order,
                                             compromise_stashed_activation,
                                             save_ratio);
}

std::string NodeRecomputePlan::GetClusterId() const {
  std::ostringstream oss;
  oss << GetNodesInTopoOrderStr();
  return oss.str();
}

std::string NodeRecomputePlan::NormalizeForNodeClusterId() const {
  std::ostringstream oss;
  oss << "recompute:" << node->OpType() << "-"
      << compromise_recompute_ << "-";
  for (auto& output_index : GetActivationOutputIndices()) {
    oss << output_index << ":" << GetActivationOutputDimParamString(output_index);
    oss << ":" << node->OutputDefs()[output_index]->TypeAsProto()->tensor_type().elem_type() << "-";
  }

  oss << GetNodesInTopoOrderStr();
  return oss.str();
}

std::string NodeRecomputePlan::GetNodesInTopoOrderStr() const {
  std::string subgraph_str_representation, log_info;
  NodesInTopoOrderToString(nodes_in_topological_order_, subgraph_str_representation, log_info);
  return subgraph_str_representation;
}

}  // namespace onnxruntime::optimizer::memory_optimizer
