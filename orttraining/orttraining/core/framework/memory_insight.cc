// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/framework/memory_insight.h"

#include "core/framework/random_seed.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/ort_value_name_idx_map.h"
// #include "core/framework/ort_value_name_idx_map.h"
// using namespace ONNX_NAMESPACE;
#include <iomanip>
#include <functional>
#include <algorithm>
#include <cctype>
#include <locale>

namespace onnxruntime {

namespace training {

// trim from start (in place)
static inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

// trim from end (in place)
static inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
}

// trim from both ends (in place)
static inline void trim(std::string& s) {
  rtrim(s);
  ltrim(s);
}

// // trim from start (copying)
// static inline std::string ltrim_copy(std::string s) {
//   ltrim(s);
//   return s;
// }

// // trim from end (copying)
// static inline std::string rtrim_copy(std::string s) {
//   rtrim(s);
//   return s;
// }

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
  trim(s);
  return s;
}

std::string empty_dim_param_placeholder = "empty_dim_param";
static int64_t index_empty_dim = 0;

struct TensorShapeAndType {
  TensorShapeAndType() = default;

  TensorShapeAndType(const ONNX_NAMESPACE::TensorShapeProto& shape,
                     const ONNX_NAMESPACE::TypeProto& type_proto)
      : shape(shape), type_proto(type_proto) {}

  std::string Normalize() const {
    std::vector<std::string> dim_params;
    // int64_t dim_value_factor = 1;
    for (int dim_index = 0; dim_index < shape.dim_size(); dim_index++) {
      auto dim = shape.dim(dim_index);
      if (utils::HasDimValue(dim)) {
        // if (keep_raw_shape) {
        dim_params.push_back(std::to_string(dim.dim_value()));
        // }
        // dim_value_factor *= static_cast<int64_t>(dim.dim_value());
      } else {
        std::string trimmed_dim_param = trim_copy(dim.dim_param());
        if (trimmed_dim_param.empty()) {
          dim_params.push_back(empty_dim_param_placeholder + std::to_string(index_empty_dim++));
        } else {
          dim_params.push_back(trimmed_dim_param);
        }
      }
    }

    if (shape.dim_size() == 0) {
      dim_params.push_back("(1)");  // Scalar
    }

    // if (!keep_raw_shape) {
    //   std::sort(dim_params.begin(), dim_params.end());
    // }

    std::ostringstream oss;
    oss << "(";
    for (auto it = dim_params.begin(); it != dim_params.end(); ++it) {
      oss << "(" << *it << ")";
      if (it != (dim_params.end() - 1)) {
        oss << "*";
      }
    }

    MLDataType ml_data_type = DataTypeImpl::TypeFromProto(type_proto);
    ORT_ENFORCE(ml_data_type->IsTensorType(), "ml_type must be a tensor type, but it is ", DataTypeImpl::ToString(ml_data_type));
    const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
    ORT_ENFORCE(nullptr != tensor_type_base);
    MLDataType elt_type = tensor_type_base->GetElementType();
    oss << ") * " << elt_type->Size();
    return oss.str();
  }

 private:
  ONNX_NAMESPACE::TensorShapeProto shape;
  ONNX_NAMESPACE::TypeProto type_proto;
};

struct OpTypePort {
  OpTypePort() = default;
  OpTypePort(const std::string& op_type, int port_index) : op_type(op_type), port_index(port_index) {}
  OpTypePort(const OpTypePort& other) : op_type(other.op_type), port_index(other.port_index) {}

  std::string Normalize() const {
    std::ostringstream oss;
    oss << op_type << ":" << port_index;
    return oss.str();
  }

  // To make OpTypePort become a valid key in std map
  bool operator<(const OpTypePort& other) const {
    if (op_type != other.op_type)
      return op_type.compare(other.op_type) < 0;
    return port_index < other.port_index;
  }

  std::string op_type;
  int port_index;
};

inline bool operator==(const OpTypePort& left, const OpTypePort& other) {
  return left.op_type == other.op_type && left.port_index == other.port_index;
}

inline bool operator!=(const OpTypePort& left, const OpTypePort& other) {
  return !(left == other);
}
// class OpTypePortHash {
//  public:
//   size_t operator()(const OpTypePort& op_type_port) const {
//     return std::hash<std::string>()(op_type_port.Normalize());
//   }
// };

struct TensorRepresentation {
  TensorRepresentation(const GraphViewer& graph_viewer,
                       const OrtValueNameIdxMap& ortvalue_name_to_idx_map,
                       const SequentialExecutionPlan& p_seq_exec_plan,
                       InlinedHashMap<std::string, InlinedVector<std::pair<const Node*, int>>>&
                           node_arg_to_bw_consumer_map,
                       const NodeArg* node_arg) : p_seq_exec_plan(p_seq_exec_plan) {
    ORT_ENFORCE(node_arg != nullptr && node_arg->Exists(), "node_arg cannot be null");
    ORT_ENFORCE(node_arg->Type() != nullptr, "node_arg->Type cannot be null", node_arg->Name());
    ORT_ENFORCE(node_arg->Shape() != nullptr, "node_arg->Shape cannot be null", node_arg->Name());
    shape_and_type = TensorShapeAndType(*node_arg->Shape(), *node_arg->TypeAsProto());

    name = node_arg->Name();
    ORT_ENFORCE(!node_arg->Name().empty());

    int ort_value_idx;
    ORT_ENFORCE(ortvalue_name_to_idx_map.GetIdx(name, ort_value_idx).IsOK());
    const auto& alloc_plan = p_seq_exec_plan.allocation_plan;
    ORT_ENFORCE(ort_value_idx >= 0 && static_cast<size_t>(ort_value_idx) < alloc_plan.size());
    const auto& per_alloc_plan = alloc_plan[ort_value_idx];
    alloc_plan_per_value = per_alloc_plan;
    ort_value_idx_ = ort_value_idx;

    if (alloc_plan_per_value.alloc_kind == AllocKind::kReuse) {
      reused_buffer_idx_ = alloc_plan_per_value.reused_buffer;
    }

    std::ostringstream oss;
    oss << alloc_plan_per_value.alloc_kind;
    alloc_type_str = oss.str();

    const Node* p_node = graph_viewer.GetProducerNode(node_arg->Name());
    ORT_ENFORCE(p_node != nullptr);
    int src_op_output_index = optimizer_utils::IndexOfNodeOutput(*p_node, *node_arg);
    src_op = OpTypePort(p_node->OpType(), src_op_output_index);

    ORT_ENFORCE(node_arg_to_bw_consumer_map.count(node_arg->Name()) == 1);
    for (const auto& bw_consumer : node_arg_to_bw_consumer_map[node_arg->Name()]) {
      const Node* p_bw_consumer = bw_consumer.first;
      int bw_consumer_input_index = bw_consumer.second;
      dest_ops[OpTypePort(p_bw_consumer->OpType(), bw_consumer_input_index)] += 1;
    }
  }

  std::string Normalize() const {
    std::string idx_str = "@" + alloc_type_str;
    // if (alloc_plan_per_value.alloc_kind == AllocKind::kReuse) {
    //   // const auto& alloc_plan = p_seq_exec_plan.allocation_plan;
    //   // ORT_ENFORCE(alloc_plan_per_value.reused_buffer >= 0 && static_cast<size_t>(alloc_plan_per_value.reused_buffer) < alloc_plan.size());
    //   idx_str += "@REUSE_IDX";
    // }

    std::ostringstream oss;
    oss << src_op.Normalize() << idx_str << " consumed by [";
    size_t i = 0;
    for (auto it = dest_ops.begin(); it != dest_ops.end(); ++it) {
      oss << "(" << it->first.Normalize() << ") X " << it->second << "";
      if (i != (dest_ops.size() - 1)) {
        oss << ",";
      }
      ++i;
    }
    oss << "]";
    return oss.str();
  }

  const TensorShapeAndType& GetShapeAndType() const {
    return shape_and_type;
  }

  bool ReusedBuffer() const {
    return alloc_plan_per_value.alloc_kind == AllocKind::kReuse;
  }

  void ResetAllocKindStr(std::string alloc_kind_str) {
    alloc_type_str = alloc_kind_str;
  }

  int OrtValueIdx() const {
    return ort_value_idx_;
  }

  int ReusedBufferIdx() const {
    return reused_buffer_idx_;
  }

  // TensorRepresentation(const std::string& name, const TensorShapeAndType& shape_and_type) : name(name), shape_and_type(shape_and_type) {}
 private:
  std::string name;
  int ort_value_idx_;
  int reused_buffer_idx_;

  std::string alloc_type_str;

  OpTypePort src_op;
  std::map<OpTypePort, int> dest_ops;
  TensorShapeAndType shape_and_type;
  AllocPlanPerValue alloc_plan_per_value;

  const SequentialExecutionPlan& p_seq_exec_plan;
};

int64_t PrepareForTransformation(const GraphViewer& graph_viewer,
                                 ActivationUsedMap& fw_op_output_arg_used_map,
                                 InlinedHashMap<NodeIndex, size_t>&
                                     node_index_to_its_order_in_topological_sort_map) {
  fw_op_output_arg_used_map.clear();

  //   GraphViewer graph_viewer(graph);
  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder(onnxruntime::ExecutionOrder::PRIORITY_BASED);

  // Find boundary ops between forward and backward pass, currently, it's limited to YieldOp.
  ptrdiff_t yield_op_order_in_topological_sort = -1;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr) { /* skip removed nodes*/
      continue;
    }

    if (p_node->OpType() == "YieldOp") {
      yield_op_order_in_topological_sort = static_cast<ptrdiff_t>(i);
    }

    node_index_to_its_order_in_topological_sort_map[p_node->Index()] = i;
  }

  // If boundary op found, create forward op output arg used map.
  if (yield_op_order_in_topological_sort >= 0) {
    for (size_t i = 0; i < node_ids.size(); ++i) {
      const Node* p_node = graph_viewer.GetNode(node_ids[i]);
      if (p_node == nullptr /* skip removed nodes*/) {
        continue;
      }

      const Node& node = *p_node;
      bool is_forward_op = IsForwardPassOperator(static_cast<ptrdiff_t>(i), yield_op_order_in_topological_sort);
      if (!is_forward_op) {
        continue;
      }

      for (auto& output_arg : node.OutputDefs()) {
        bool used_in_fw = false;
        bool used_in_bw = false;
        for (auto& consumer_node : graph_viewer.GetConsumerNodes(output_arg->Name())) {
          size_t consumer_node_index_in_topological_order =
              node_index_to_its_order_in_topological_sort_map.at(consumer_node->Index());
          if (IsForwardPassOperator(static_cast<ptrdiff_t>(consumer_node_index_in_topological_order),
                                    yield_op_order_in_topological_sort)) {
            used_in_fw = true;
          } else {
            used_in_bw = true;
          }
        }
        fw_op_output_arg_used_map.insert({{output_arg->Name(), std::make_pair(used_in_fw, used_in_bw)}});
      }
    }
  }

  // Return whether boundary op is found or not.
  return yield_op_order_in_topological_sort;
}

Status GetStashedActivationCandidates(const GraphViewer& graph_viewer,
                                      const InlinedHashMap<std::string, std::pair<bool, bool>>&
                                          fw_op_output_arg_used_map,
                                      InlinedHashMap<const Node*, InlinedVector<size_t>>&
                                          candidate_output_args_map,
                                      InlinedHashMap<std::string, InlinedVector<std::pair<const Node*, int>>>&
                                          node_arg_to_bw_consumer_map,
                                      int64_t boundary_op_order_in_topological_sort,
                                      const InlinedHashMap<NodeIndex, size_t>& node_index_to_its_order_in_topological_sort_map,
                                      const logging::Logger& logger) {
  for (auto& kv : fw_op_output_arg_used_map) {
    // used by bw, then it is a candidates.
    if (kv.second.second) {
      const Node* n = graph_viewer.GetProducerNode(kv.first);
      ORT_ENFORCE(n, "Activation should have a producer node");
      size_t k = 0;
      for (k = 0; k < n->OutputDefs().size(); ++k) {
        if (n->OutputDefs()[k]->Name().compare(kv.first) == 0) {
          break;
        }
      }

      candidate_output_args_map[n].push_back(k);
      LOGS(logger, VERBOSE) << "Find candidate output named [" << kv.first << "] of Node " << n->Name() << "("
                            << n->OpType() << ")";

      const NodeArg* node_arg = n->OutputDefs()[k];
      for (auto out_edge = n->OutputEdgesBegin(), end = n->OutputEdgesEnd(); out_edge != end; ++out_edge) {
        if (out_edge->GetSrcArgIndex() != static_cast<int>(k)) {
          continue;
        }
        const Node& next_node = out_edge->GetNode();
        // Skipp the forward consumers.
        if (node_index_to_its_order_in_topological_sort_map.at(next_node.Index()) <=
            static_cast<size_t>(boundary_op_order_in_topological_sort)) {
          continue;
        }
        auto& node_list = node_arg_to_bw_consumer_map[node_arg->Name()];
        node_list.push_back({&next_node, static_cast<int>(out_edge->GetDstArgIndex())});
      }
    }
  }

  return Status::OK();
}

struct ShapeAndTypeKeyedStat {
  ShapeAndTypeKeyedStat() {
  }

  void AddStat(const std::string& key, const TensorRepresentation& tensor_representation) {
    auto it = std::find(keys_names.begin(), keys_names.end(), key);
    size_t index = 0;
    if (it == keys_names.end()) {
      keys_names.push_back(key);
      unreused_resued_counts.push_back({0, 0});
      tensor_representations.push_back({});
      index = tensor_representations.size() - 1;
    } else {
      index = it - keys_names.begin();
    }

    // if (tensor_representation.ReusedBuffer()) {
    //   unreused_resued_counts[index].second += 1;
    // } else {
    //   unreused_resued_counts[index].first += 1;
    // }

    tensor_representations[index].push_back(tensor_representation);
  }

  std::string GroupTensorRepresentation(const std::vector<TensorRepresentation>& tensor_representations) const {
    std::unordered_map<std::string, int> tensor_representations_str_to_freq_map;
    for (auto& tensor_representation : tensor_representations) {
      std::string key = tensor_representation.Normalize();
      if (tensor_representations_str_to_freq_map.find(key) == tensor_representations_str_to_freq_map.end()) {
        tensor_representations_str_to_freq_map[key] = 0;
      }
      tensor_representations_str_to_freq_map[key]++;
    }

    std::ostringstream oss;
    size_t i = 0;
    for (auto& kv : tensor_representations_str_to_freq_map) {
      // if (i != 0) {
      //   oss << "\n";
      // }
      oss << "{" << kv.first << "} X " << kv.second << ",";
      i += 1;
    }

    return oss.str();
  }

  void Summarize(std::vector<std::vector<std::string>>& body) {
    // Loop all tensor representations and find those reused buffer that is self contained in those tensor.
    // We want to avoid compute the same tensor twice among the stashed activations.
    std::unordered_set<int> ortvalue_ids_for_stashed_activations;
    for (auto& tensor_representation_vector : tensor_representations) {
      for (auto& tensor_representation : tensor_representation_vector) {
        ortvalue_ids_for_stashed_activations.insert(tensor_representation.OrtValueIdx());
      }
    }

    for (size_t i = 0; i < tensor_representations.size(); ++i) {
      auto& tensor_representation_vector = tensor_representations[i];
      for (auto& tensor_representation : tensor_representation_vector) {
        if (tensor_representation.ReusedBuffer() && ortvalue_ids_for_stashed_activations.count(tensor_representation.ReusedBufferIdx()) > 0) {
          unreused_resued_counts[i].second += 1;
          tensor_representation.ResetAllocKindStr("REUSED_STASHED");
        } else {
          unreused_resued_counts[i].first += 1;
        }
      }
    }

    // std::ostringstream oss;
    body.clear();
    body.push_back({"Tensor shape and type string representation",
                    "Total count",
                    "Unreused",
                    // "Reused count",
                    "Break down by producer-consumer pattern"});

    for (size_t i = 0; i < keys_names.size(); ++i) {
      body.push_back({keys_names[i],
                      std::to_string(tensor_representations[i].size()),
                      std::to_string(unreused_resued_counts[i].first),
                      // std::to_string(unreused_resued_counts[i].second),
                      GroupTensorRepresentation(tensor_representations[i])});
      // oss << std::setw(20) << keys_names[i] << std::setw(10) << tensor_representations[i].size() << std::setw(200);
      // oss << GroupTensorRepresentation(tensor_representations[i]) << "\n";
    }
    // return oss.str();
  }

 private:
  std::vector<std::string> keys_names;
  std::vector<std::vector<TensorRepresentation>> tensor_representations;
  std::vector<std::pair<int, int>> unreused_resued_counts;  // The fist one is unreused, the second one is reused.
};                                                          // namespace training

Status SymbolizeMemoryPeak(const GraphViewer& graph_viewer,
                           const OrtValueNameIdxMap& ortvalue_name_to_idx_map,
                           const SequentialExecutionPlan& p_seq_exec_plan,
                           const logging::Logger& logger,
                           std::vector<std::vector<std::string>>& body,
                           std::unordered_map<std::string, bool>& loss_grad_stat) {
  LOGS(logger, WARNING) << "SymbolizeMemoryPeak";

  InlinedHashMap<std::string, std::pair<bool, bool>> fw_op_output_arg_used_map;
  InlinedHashMap<NodeIndex, size_t> node_index_to_its_order_in_topological_sort_map;
  int64_t boundary_op_order_in_topological_sort =
      PrepareForTransformation(graph_viewer, fw_op_output_arg_used_map,
                               node_index_to_its_order_in_topological_sort_map);
  if (boundary_op_order_in_topological_sort < 0) {
    LOGS(logger, WARNING) << "No boundary op found. Skip SymbolizeMemoryPeak.";
    return Status::OK();
  }

  InlinedHashMap<const Node*, InlinedVector<size_t>> candidate_output_args_map;
  InlinedHashMap<std::string, InlinedVector<std::pair<const Node*, int>>>
      node_arg_to_bw_consumer_map;
  ORT_RETURN_IF_ERROR(GetStashedActivationCandidates(graph_viewer, fw_op_output_arg_used_map,
                                                     candidate_output_args_map,
                                                     node_arg_to_bw_consumer_map,
                                                     boundary_op_order_in_topological_sort,
                                                     node_index_to_its_order_in_topological_sort_map,
                                                     logger));

  const auto& node_ids = graph_viewer.GetNodesInTopologicalOrder(onnxruntime::ExecutionOrder::PRIORITY_BASED);

  ShapeAndTypeKeyedStat stked_stat;
  loss_grad_stat.clear();
  // ShapeAndTypeKeyedStat grad_stat;
  for (int i = 0; i < static_cast<int>(node_ids.size()); ++i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr) {
      continue;
    }

    if (candidate_output_args_map.find(p_node) == candidate_output_args_map.end()) {
      if (p_node->OpType() == "SoftmaxCrossEntropyLossInternalGrad") {
        const NodeArg* node_arg = p_node->OutputDefs()[0];
        if (node_arg == nullptr) {
          continue;
        }

        ORT_ENFORCE(node_arg != nullptr && node_arg->Exists(), "node_arg cannot be null");
        auto shape_and_type = TensorShapeAndType(*node_arg->Shape(), *node_arg->TypeAsProto());

        auto name = node_arg->Name();
        ORT_ENFORCE(!node_arg->Name().empty());

        int ort_value_idx;
        ORT_ENFORCE(ortvalue_name_to_idx_map.GetIdx(name, ort_value_idx).IsOK());
        const auto& alloc_plan = p_seq_exec_plan.allocation_plan;
        ORT_ENFORCE(ort_value_idx >= 0 && static_cast<size_t>(ort_value_idx) < alloc_plan.size());
        loss_grad_stat[shape_and_type.Normalize()] = (alloc_plan[ort_value_idx].alloc_kind == AllocKind::kReuse);
      }
      continue;
    }

    for (auto output_arg_index : candidate_output_args_map[p_node]) {
      const NodeArg* p_output_arg = p_node->OutputDefs()[output_arg_index];
      // todo: handle unknown
      if (p_output_arg == nullptr || !p_output_arg->Exists() || p_output_arg->Shape() == nullptr || p_output_arg->Type() == nullptr) {
        LOGS(logger, WARNING) << "todo: " << p_output_arg->Name() << p_node->OpType() << p_node->Name();
        continue;
      }

      TensorRepresentation tensor_representation(graph_viewer, ortvalue_name_to_idx_map,
                                                 p_seq_exec_plan, node_arg_to_bw_consumer_map,
                                                 p_output_arg);

      std::string shape_and_type_key = tensor_representation.GetShapeAndType().Normalize();

      stked_stat.AddStat(shape_and_type_key, tensor_representation);
    }
  }

  stked_stat.Summarize(body);

  return Status::OK();
}  // namespace training

}  // namespace training
}  // namespace onnxruntime

namespace std {
template <>
struct hash<onnxruntime::training::OpTypePort> {
  size_t operator()(const onnxruntime::training::OpTypePort& op_type_port) const {
    return std::hash<std::string>()(op_type_port.Normalize());
  }
};
}  // namespace std
