// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "qdq_scales_fix.h"
#include "core/providers/openvino/ov_protobuf_utils.h"

#include <fstream>
#include <list>
#include <string>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include <ranges>
#include <algorithm>

namespace onnxruntime {
namespace openvino_ep {

namespace qdq_scales_fix {

namespace fs = std::filesystem;
using NodeRef = std::reference_wrapper<const Node>;
struct GraphNode;
float get_initializer_value(const Graph& graph, const std::string& initializer_name);

template <typename T, typename V>
bool contains(V&& begin, V&& end, const T& val) {
  for (V& iter = begin; iter != end; iter.operator++()) {
    if (iter->Name() == val) {
      return true;
    }
  }
  return false;
}

template <typename T, std::ranges::range R>
bool contains(const R& vec, const T& val) {
  for (auto iter = vec.begin(); iter != vec.end(); iter++) {
    if ((*iter)->Name() == val) {
      return true;
    }
  }
  return false;
}

bool contains(const std::vector<std::string>& container, const std::string& value) {
  return std::find(container.begin(), container.end(), value) != container.end();
}

struct GraphNode {
  GraphNode() = delete;

  template <typename N>
  GraphNode(const N& node, const std::string& op_type = {}) {
    node_name = node.Name();
    if constexpr (std::is_same_v<N, Node>) {
      node_ptr = &node;
      this->op_type = node.OpType();
      for (const auto iter : node.InputDefs()) {
        node_input_name.push_back(iter->Name());
      }
      for (const auto iter : node.OutputDefs()) {
        node_output_name.push_back(iter->Name());
      }
    } else {
      this->op_type = op_type;
      //** node_input_name = []
      //** node_output_name = []
    }

    if (op_type == "output") {
      down_to_output = true;
    }
  }

  bool operator==(const GraphNode&) const = default;

  void add_edge_to(GraphNode& dst_node) {
    to_node.push_back(&dst_node);
  }

  void add_edge_from(GraphNode& src_node) {
    from_node.push_back(&src_node);
  }

  std::vector<GraphNode*> apply_scale_to_graph(float scale_adj) {
    std::vector<GraphNode*> affected_dq;

    auto extend = [&affected_dq, scale_adj](const std::vector<GraphNode*>& new_nodes) {
      affected_dq.insert(affected_dq.end(), new_nodes.begin(), new_nodes.end());
    };

    if (op_type == "DequantizeLinear") {
      scale_factor *= scale_adj;
      affected_dq.push_back(this);
    } else if ((op_type == "Add") || (op_type == "QuantizeLinear")) {
      for (auto node : from_node) {
        extend(node->apply_scale_to_graph(scale_adj));
      }
    } else if (op_type == "Conv") {
      // just adjust w&b for conv&mul, stop propagate
      for (auto node : from_node) {
        if (node->op_type == "DequantizeLinear") {
          extend(node->apply_scale_to_graph(scale_adj));
        }
      }
    } else if ((op_type == "Mul") || (op_type == "MatMul")) {
      bool find_dq{false};
      for (auto node : from_node) {
        if (node->op_type == "DequantizeLinear" && !find_dq) {
          find_dq = true;
          extend(node->apply_scale_to_graph(scale_adj));
        }
      }
      if (!find_dq) {
        // cannot scale dq from here, choose input 0 to propagate
        extend(from_node.back()->from_node[0]->apply_scale_to_graph(scale_adj));
      }
    } else {
      ORT_THROW("Unknown case, node: %s", ToString().data());
    }

    return affected_dq;
  }

  std::vector<GraphNode*> down_propagate_scale() {
    std::vector<GraphNode*> affected_nodes;

    if (processed) {
      return affected_nodes;
    }

    if ((op_type == "InstanceNormalization") || (op_type == "Softmax")) {
      // pass
    } else if (op_type == "Add") {
      auto up_new_nodes = up_propagate_scale();
      affected_nodes.insert(affected_nodes.end(), up_new_nodes.begin(), up_new_nodes.end());

      for (auto node : to_node) {
        auto down_new_nodes = node->down_propagate_scale();
        affected_nodes.insert(affected_nodes.end(), down_new_nodes.begin(), down_new_nodes.end());
      }
    } else {
      affected_nodes.push_back(this);
      processed = true;

      for (auto node : to_node) {
        auto new_nodes = node->down_propagate_scale();
        affected_nodes.insert(affected_nodes.end(), new_nodes.begin(), new_nodes.end());
      }
    }
    return affected_nodes;
  }

  std::vector<GraphNode*> up_propagate_scale() {
    std::vector<GraphNode*> affected_nodes;

    if (processed) {
      return affected_nodes;
    }

    if ((op_type == "InstanceNormalization") || (op_type == "Softmax")) {
      ORT_THROW("Cannot propagate up through norm layer");
    } else if (op_type == "Conv") {
      affected_nodes.push_back(this);
      processed = true;

      for (auto node : from_node) {
        if (node->op_type == "DequantizeLinear") {
          affected_nodes.push_back(node);
        }
      }
    } else if ((op_type == "Mul") || (op_type == "MatMul")) {
      affected_nodes.push_back(this);
      processed = true;
      bool find_dq{false};

      for (auto node : from_node) {
        if ((node->op_type == "DequantizeLinear") && !find_dq) {
          find_dq = true;
          affected_nodes.push_back(node);
        }
      }
      if (!find_dq) {
        auto new_nodes = from_node.back()->from_node[0]->up_propagate_scale();
        affected_nodes.insert(affected_nodes.end(), new_nodes.begin(), new_nodes.end());
      }
    } else {
      affected_nodes.push_back(this);
      processed = true;

      for (auto node : from_node) {
        auto new_nodes = node->up_propagate_scale();
        affected_nodes.insert(affected_nodes.end(), new_nodes.begin(), new_nodes.end());
      }
    }

    return affected_nodes;
  }

  bool down_propagate_to_output() {
    if (down_to_output.has_value()) {
      return down_to_output.value();
    }

    bool local_down_to_output{false};
    if (op_type == "output") {
      local_down_to_output = true;
    } else if ((op_type == "InstanceNormalization") || (op_type == "Softmax")) {
      local_down_to_output = false;
    } else {
      for (auto node : to_node) {
        local_down_to_output = local_down_to_output || node->down_propagate_to_output();
      }
    }

    down_to_output = local_down_to_output;
    return local_down_to_output;
  }

  std::string ToString() const {
    // auto string = std::format("op={} name={} queued={} visited={} scale_factor={}",
    //                           op_type,
    //                           node_name,
    //                           queued,
    //                           visited,
    //                           scale_factor);
    auto print_node_vector = [](const std::vector<GraphNode*>& nodes) -> std::string {
      // auto comp = [](const GraphNode* left, const GraphNode* right) -> bool {
      //   return left->node_name < right->node_name;
      // };
      // std::sort(nodes.begin(), nodes.end(), comp);
      std::string ret = "[";
      for (size_t i = 0, size = nodes.size(); auto pnode : nodes) {
        if (pnode->node_name.size() == 0) continue;
        ret += pnode->node_name;
        if (++i < size) {
          ret += ", ";
        }
      }
      ret += "]";
      return ret;
    };
    std::string from_node_str = print_node_vector(from_node);
    std::string to_node_str = print_node_vector(to_node);

    auto print_string_vector = [](const std::vector<std::string>& nodes) -> std::string {
      // std::sort(nodes.begin(), nodes.end());
      std::string ret = "[";
      for (size_t i = 0, size = nodes.size(); const auto& node : nodes) {
        ret += node;
        if (++i < size) {
          ret += ", ";
        }
      }
      ret += "]";
      return ret;
    };
    std::string node_input_name_str = print_string_vector(node_input_name);
    std::string node_output_name_str = print_string_vector(node_output_name);

    auto print_bool = [](bool val) -> std::string {
      return (val) ? "True" : "False";
    };

    auto print_opt_bool = [print_bool](std::optional<bool> val) -> std::string {
      return (val.has_value()) ? print_bool(val.value()) : "None";
    };

    auto string = std::format("node_name={} op_type={} scale_factor={:.2f} visited={} queued={} down_to_output={} processed={} from_node={} to_node={} node_input_name={} node_output_name={}",
                              node_name,
                              op_type,
                              scale_factor,
                              visited,
                              print_bool(queued),
                              print_opt_bool(down_to_output),
                              print_bool(processed),
                              from_node_str,
                              to_node_str,
                              node_input_name_str,
                              node_output_name_str);
    return string;
  }

  const Node* node_ptr{nullptr};
  std::string node_name;
  std::string op_type;
  std::vector<std::string> node_input_name;
  std::vector<std::string> node_output_name;
  std::vector<GraphNode*> from_node;
  std::vector<GraphNode*> to_node;
  float scale_factor{1.f};
  int visited{0};
  bool queued{false};
  std::optional<bool> down_to_output;
  bool processed{false};
};

struct CustomGraph {
  CustomGraph() = delete;
  CustomGraph(Graph& graph) : original_graph{graph} {}

  void sort() {
    auto comp_node = [](const GraphNode& left, const GraphNode& right) -> bool {
      return left.node_name < right.node_name;
    };
    nodes.sort(comp_node);

    for (auto& node : nodes) {
      auto comp_pnode = [](const GraphNode* left, const GraphNode* right) -> bool {
        return left->node_name < right->node_name;
      };
      std::sort(node.from_node.begin(), node.from_node.end(), comp_pnode);
      std::sort(node.to_node.begin(), node.to_node.end(), comp_pnode);
    }
  }

  void add_node(const GraphNode& node) {
    nodes.push_back(node);
  }

  void add_edge(GraphNode& src, GraphNode& dst) {
    src.add_edge_to(dst);
    dst.add_edge_from(src);
  }

  auto get_start_nodes() {
    std::list<GraphNode*> start_nodes;

    for (auto& node : nodes) {
      if (node.from_node.empty()) {
        start_nodes.push_back(&node);
        node.queued = true;
      }
    }
    return start_nodes;
  }

  void initailize_search(float threshold = 1.f, bool scale_output = false) {
    remove_qdq(threshold, scale_output);
    for (auto& node : nodes) {
      node.visited = 0;
      node.queued = false;
    }
  }

  void init_propagate() {
    for (auto& node : nodes) {
      node.processed = false;
    }
  }

  void remove_qdq_pair(const GraphNode& node, std::list<GraphNode>& removed) {
    auto& q = node;
    InlinedVector<GraphNode*> dq_ptrs;

    for (auto& child : q.to_node) {
      if (child->node_ptr && child->node_ptr->OpType() == "DequantizeLinear") {
        dq_ptrs.push_back(child);
      }
    }

    if (dq_ptrs.empty()) {
      return;
    }

    for (std::size_t i = 1; i < dq_ptrs.size(); ++i) {
      if (dq_ptrs[i]->node_input_name[1] != dq_ptrs[0]->node_input_name[1] ||
          dq_ptrs[i]->node_input_name[2] != dq_ptrs[0]->node_input_name[2]) {
        return;
      }
    }

    auto& prev = *node.from_node[0];
    const auto& q_node = *q.node_ptr;

    bool is_prev_input = (prev.node_ptr == nullptr);
    std::string prev_output_name = is_prev_input ? prev.node_name : prev.node_output_name[0];

    InlinedVector<std::pair<const NodeArg*, const NodeArg*>> output_replacements;
    for (auto dq_ptr : dq_ptrs) {
      for (auto dst_node : dq_ptr->to_node) {
        for (auto& scr_node : dst_node->from_node) {
          if (*dq_ptr == *scr_node) {
            scr_node = &prev;
          }
        }

        auto it = std::find(dst_node->node_input_name.begin(), dst_node->node_input_name.end(), dq_ptr->node_output_name[0]);
        if (it != dst_node->node_input_name.end()) {
          *it = prev_output_name;
        }
      }
      for (auto& output : original_graph.GetOutputs()) {
        if (output->Name() == dq_ptr->node_output_name[0]) {
          const NodeArg* replacement_arg = nullptr;
          if (!is_prev_input) {
            replacement_arg = prev.node_ptr->OutputDefs()[0];
          } else {
            replacement_arg = original_graph.GetNodeArg(prev.node_name);
            ORT_ENFORCE(replacement_arg != nullptr, "Input not found: " + prev.node_name);
          }
          output_replacements.emplace_back(output, replacement_arg);
        }
      }
    }

    prev.to_node.erase(std::remove(prev.to_node.begin(), prev.to_node.end(), &q), prev.to_node.end());
    for (auto dq_ptr : dq_ptrs) {
      for (auto dst_node : dq_ptr->to_node) {
        auto it = std::find(prev.to_node.begin(), prev.to_node.end(), dst_node);
        if (it == prev.to_node.end()) {
          prev.to_node.push_back(dst_node);
        }
      }
    }
    auto q_iter = std::find(nodes.begin(), nodes.end(), q);
    if (q_iter != nodes.end()) {
      removed.splice(removed.end(), nodes, q_iter);
    }

    for (auto dq_ptr : dq_ptrs) {
      auto dq_iter = std::find(nodes.begin(), nodes.end(), *dq_ptr);
      if (dq_iter != nodes.end()) {
        removed.splice(removed.end(), nodes, dq_iter);
      }
    }

    auto remove_edge = [this](const Node& src, const Node& dst, int src_arg, int dst_arg) {
      original_graph.RemoveEdge(src.Index(), dst.Index(), src_arg, dst_arg);
    };

    auto in_edge = q_node.InputEdgesBegin();
    ORT_ENFORCE(in_edge != q_node.InputEdgesEnd(), "Q node must have an input edge");
    const int prev_output_index = in_edge->GetSrcArgIndex();

    if (in_edge != q_node.InputEdgesEnd()) {
      remove_edge(in_edge->GetNode(), q_node,
                  in_edge->GetSrcArgIndex(), in_edge->GetDstArgIndex());
    }
    for (auto dq_ptr : dq_ptrs) {
      auto& dq_node_ref = *dq_ptr->node_ptr;

      for (auto edge_it = dq_node_ref.InputEdgesBegin(); edge_it != dq_node_ref.InputEdgesEnd(); ++edge_it) {
        if (edge_it->GetNode().Index() == q_node.Index()) {
          remove_edge(edge_it->GetNode(), dq_node_ref, edge_it->GetSrcArgIndex(), edge_it->GetDstArgIndex());
          break;
        }
      }

      std::vector<std::tuple<NodeIndex, int, int>> output_edges;  // (dst_node_index, src_arg, dst_arg)
      for (auto out_edge_it = dq_node_ref.OutputEdgesBegin(); out_edge_it != dq_node_ref.OutputEdgesEnd(); ++out_edge_it) {
        output_edges.emplace_back(out_edge_it->GetNode().Index(),
                                  out_edge_it->GetSrcArgIndex(),
                                  out_edge_it->GetDstArgIndex());
      }

      for (const auto& edge : output_edges) {
        original_graph.RemoveEdge(dq_node_ref.Index(), std::get<0>(edge),
                                  std::get<1>(edge), std::get<2>(edge));
      }

      if (!is_prev_input) {
        for (const auto& edge : output_edges) {
          original_graph.AddEdge(prev.node_ptr->Index(),
                                 std::get<0>(edge),
                                 prev_output_index,
                                 std::get<2>(edge));
        }
      }
    }

    if (!output_replacements.empty()) {
      auto outputs = original_graph.GetOutputs();
      for (auto& output : outputs) {
        for (const auto& replacement : output_replacements) {
          if (output == replacement.first) {
            output = replacement.second;
            break;
          }
        }
      }
      original_graph.SetOutputs(outputs);
    }

    original_graph.RemoveNode(q_node.Index());
    for (auto dq_ptr : dq_ptrs) {
      original_graph.RemoveNode(dq_ptr->node_ptr->Index());
    }
  }

  std::list<GraphNode> remove_qdq(float threshold = 1.f, bool scale_output = false) {
    std::list<GraphNode> removed;
    std::vector<GraphNode*> nodes_copy;
    std::for_each(nodes.begin(), nodes.end(), [&nodes_copy](GraphNode& node) { nodes_copy.push_back(&node); });
    for (auto node : nodes_copy) {
      if (std::find(nodes.begin(), nodes.end(), *node) == nodes.end()) {
        continue;
      }

      if ((node->op_type == "QuantizeLinear") &&
          (node->to_node[0]->op_type == "DequantizeLinear")) {
        const auto& zero_point_name = node->node_input_name[2];
        const auto p_initializer = original_graph.GetConstantInitializer(zero_point_name, false);
        bool is_16_bit = p_initializer->has_data_type() &&
                         (p_initializer->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT16 ||
                          p_initializer->data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT16);
        if (!is_16_bit)
          continue;
        if (!scale_output && node->down_propagate_to_output()) {
          remove_qdq_pair(*node, removed);
          continue;
        }

        auto scale_name = node->node_input_name[1];  // Scale
        auto scale_value = get_initializer_value(original_graph, scale_name);
        if (scale_value / node->scale_factor < threshold) {
          remove_qdq_pair(*node, removed);
        }
      }
    }

    // Reconnect graph outputs if disconnected
    bool update_outputs{false};
    auto outputs = original_graph.GetOutputs();
    for (auto output : outputs) {
      bool found{false};
      for (auto node : original_graph.Nodes()) {
        if (contains(node->OutputNodesBegin(), node->OutputNodesEnd(), output->Name())) {
          found = true;
          break;
        }
      }

      if (!found) {
        // Connect the last valid node to the graph output
        for (auto node : std::ranges::reverse_view(original_graph.Nodes())) {
          if (!node->OutputDefs().empty()) {
            const auto& name = (*node->OutputDefs().begin())->Name();
            auto& node_arg = original_graph.GetOrCreateNodeArg(name, output->TypeAsProto());
            output = &node_arg;
            update_outputs = true;
          }
        }
      }
    }

    if (update_outputs) {
      original_graph.SetOutputs(outputs);
    }

    return removed;
  }

  void dump_custom_graph(fs::path path) {
    if (auto file = std::ofstream(path)) {
      std::vector<GraphNode*> node_ref;
      for (auto& node : nodes) {
        node_ref.emplace_back(&node);
      }

      for (const auto& node : node_ref) {
        std::string node_str = node->ToString();
        file << node_str << "\n";
      }
    }
  }

  std::list<GraphNode> nodes;
  std::list<GraphNode> removed_nodes;
  Graph& original_graph;
};

template <typename T>
T* get_mutable_initializer_data(const Graph& graph, const std::string& name) {
  auto initializer = graph.GetConstantInitializer(name, true);
  if (!initializer) return nullptr;

  if constexpr (std::is_same_v<T, float>) {
    if (initializer->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      return nullptr;
  }

  return reinterpret_cast<T*>(const_cast<char*>(initializer->raw_data().data()));
}

std::size_t get_initializer_size(const Graph& graph, const std::string& name) {
  auto initializer = graph.GetConstantInitializer(name, true);
  if (!initializer) return 0;

  std::size_t size = 1;
  if (!initializer->dims_size())
    return 0;
  for (int i = 0; i < initializer->dims_size(); ++i) {
    size *= initializer->dims()[i];
  }
  return size;
}

float get_initializer_value(const Graph& graph, const std::string& initializer_name) {
  const auto p_initializer = graph.GetConstantInitializer(initializer_name, false);

  if (p_initializer->has_raw_data()) {
    auto raw_data = get_mutable_initializer_data<float>(graph, initializer_name);
    auto size = get_initializer_size(graph, initializer_name);
    ORT_ENFORCE(size == 1, "Expected an initializer to be of size 1");
    return raw_data[0];
  } else
    return get_float_initializer_data(p_initializer);
}

void update_initializer_value(Graph& graph, const std::string& initializer_name, const float new_value) {
  const auto p_initializer = graph.GetConstantInitializer(initializer_name, false);

  if (p_initializer == nullptr) {
    return;
  }

  const auto& initializer = *p_initializer;

  // Verify 1D tensor
  ORT_ENFORCE(initializer.dims_size() == 1);
  ORT_ENFORCE(initializer.data_type() == onnx::TensorProto_DataType_FLOAT);

  // Create new tensor with updated value
  auto new_tensor = onnx::TensorProto::Create();
  new_tensor->copy_from(p_initializer);
  *(float*)new_tensor->mutable_raw_data()->data() = new_value;
  graph.RemoveInitializedTensor(initializer_name);
  graph.AddInitializedTensor(*new_tensor);
}

CustomGraph generate_graph_from_onnx(Graph& graph) {
  CustomGraph gen_graph{graph};

  for (auto pnode : graph.Nodes()) {
    if (pnode->NodeType() == Node::Type::Fused) continue;
    gen_graph.nodes.emplace_back(*pnode);
  }

  for (auto& src_node : gen_graph.nodes) {
    for (auto& dst_node : gen_graph.nodes) {
      if (src_node == dst_node) {
        continue;
      }

      for (auto& src_output : src_node.node_output_name) {
        if (contains(dst_node.node_input_name, src_output)) {
          gen_graph.add_edge(src_node, dst_node);
        }
      }
    }
  }

  for (auto& input_node : graph.GetInputs()) {
    auto& cur_input = gen_graph.nodes.emplace_back(*input_node, "input");
    for (auto& dst_node : gen_graph.nodes) {
      for (const auto& dst_output : dst_node.node_input_name) {
        if (dst_output == input_node->Name()) {
          gen_graph.add_edge(cur_input, dst_node);
        }
      }
    }
  }

  for (auto& output_node : graph.GetOutputs()) {
    auto& cur_output = gen_graph.nodes.emplace_back(*output_node, "output");
    for (auto& src_node : gen_graph.nodes) {
      for (const auto& dst_outputs : src_node.node_output_name) {
        if (dst_outputs == output_node->Name()) {
          gen_graph.add_edge(src_node, cur_output);
        }
      }
    }
  }

  gen_graph.sort();
  return gen_graph;
}

bool scale_graph(CustomGraph& gen_graph,
                 float threshold = 1.f,
                 float ratio = 10,
                 bool scale_output = false) {
  bool needs_second_run = false;
  gen_graph.initailize_search(threshold, scale_output);
  auto q = gen_graph.get_start_nodes();
  auto pred = [](const GraphNode* left, const GraphNode* right) -> bool {
    return left->node_name < right->node_name;
  };
  q.sort(pred);

  while (!q.empty()) {
    auto cur_node = q.front();
    q.pop_front();
    if (static_cast<std::size_t>(cur_node->visited) < cur_node->from_node.size()) {
      cur_node->queued = false;
    } else {
      if (cur_node->op_type == "QuantizeLinear" &&
          cur_node->to_node[0]->op_type == "DequantizeLinear") {
        needs_second_run = true;
        auto scale_name = *std::next(cur_node->node_input_name.begin());
        auto scale_value = get_initializer_value(gen_graph.original_graph, scale_name);

        // QDQ pair with scale over 1
        if (scale_value / cur_node->scale_factor > threshold) {
          gen_graph.init_propagate();
          // adjust previous op scale to threshold / 10
          auto scale_adj = scale_value / cur_node->scale_factor / threshold * ratio;

          // find related const dq to scale down
          auto affected_dq = cur_node->apply_scale_to_graph(scale_adj);
          std::vector<GraphNode*> affected_nodes;

          // then propage to graph to update scale
          for (auto& dq : affected_dq) {
            auto cur_affected = dq->down_propagate_scale();
            affected_nodes.insert(affected_nodes.end(), cur_affected.begin(), cur_affected.end());
          }

          for (auto& node : affected_nodes) {
            bool found = std::find(affected_dq.begin(), affected_dq.end(), node) != affected_dq.end();
            if (!found) {
              node->scale_factor *= scale_adj;
            }
          }

          auto removed_qdq = gen_graph.remove_qdq(threshold, scale_output);
          for (auto& qdq : removed_qdq) {
            try {
              q.remove(&qdq);
            } catch (...) {
            }
          }

          gen_graph.removed_nodes.splice(gen_graph.removed_nodes.end(), removed_qdq);

          cur_node = cur_node->to_node[0];
        }
      }

      for (auto dst : cur_node->to_node) {
        dst->visited += 1;
        if (!dst->queued) {
          dst->queued = true;
          q.push_back(dst);
        }
      }
    }
  }

  for (auto& node : gen_graph.nodes) {
    if (node.op_type == "DequantizeLinear" && node.scale_factor != 1.0f) {
      const auto& scale_name = node.node_input_name[1];

      auto scale_data = get_mutable_initializer_data<float>(gen_graph.original_graph, scale_name);
      if (scale_data) {
        const auto scale_size = get_initializer_size(gen_graph.original_graph, scale_name);
        if (!scale_size) {
          auto it = gen_graph.original_graph.GetConstantInitializer(scale_name, true);
          auto cur_scale = get_float_initializer_data(it);
          cur_scale /= node.scale_factor;
          set_float_initializer_data(it, cur_scale);
        } else {
          for (std::size_t i = 0; i < scale_size; ++i) {
            scale_data[i] /= node.scale_factor;
          }
        }
      }

      node.scale_factor = 1.0f;
    }
  }
  return needs_second_run;
}

Status copy_model(const GraphViewer& src_graph_viewer,
                  const logging::Logger& logger, std::unique_ptr<onnxruntime::Model>& model) {
  model = src_graph_viewer.CreateModel(logger);
  const auto& src_graph = src_graph_viewer.GetGraph();
  auto& dst_graph = model->MainGraph();

  const auto& inputs = src_graph.GetInputs();
  const auto& outputs = src_graph.GetOutputs();

  struct InputReplacement {
    NodeArg* graph_input;
    NodeArg* identity_output;
  };
  std::unordered_map<std::string, InputReplacement> input_replacement_map;

  struct OutputReplacement {
    NodeArg* intermediate_arg;
    NodeArg* original_output;
  };
  std::unordered_map<std::string, OutputReplacement> output_replacement_map;

  InlinedVector<const NodeArg*> dst_graph_inputs;
  dst_graph_inputs.reserve(inputs.size());
  for (auto& input : inputs) {
    const auto& input_name = input->Name();
    auto input_arg = src_graph.GetNodeArg(input_name);

    auto& dst_input_arg = dst_graph.GetOrCreateNodeArg(input_name, input_arg->TypeAsProto());
    dst_graph_inputs.push_back(&dst_input_arg);

    auto output_name = input_name + "_identity_output";
    auto& identity_output_arg = dst_graph.GetOrCreateNodeArg(output_name, input_arg->TypeAsProto());

    input_replacement_map[input_name] = {&dst_input_arg, &identity_output_arg};
  }

  InlinedVector<const NodeArg*> dst_graph_outputs;
  for (auto& output : outputs) {
    const auto& output_name = output->Name();
    auto output_arg = src_graph.GetNodeArg(output_name);

    std::string intermediate_name = "tmp_" + output_name;
    auto& intermediate_out = dst_graph.GetOrCreateNodeArg(intermediate_name, output_arg->TypeAsProto());

    auto& original_out = dst_graph.GetOrCreateNodeArg(output_name, output_arg->TypeAsProto());

    output_replacement_map[output_name] = {&intermediate_out, &original_out};
    dst_graph_outputs.push_back(&original_out);
  }

  dst_graph.SetInputs(dst_graph_inputs);
  dst_graph.SetOutputs(dst_graph_outputs);
  dst_graph.SetName(src_graph.Name());

  for (const auto& name : src_graph_viewer.GetOuterScopeNodeArgNames()) {
    auto node_arg = src_graph.GetNodeArg(name);
    ORT_RETURN_IF_NOT(node_arg != nullptr, "Outer scope node arg name '" + name + "'was added but does not exist. ");
    dst_graph.AddOuterScopeNodeArg(name);
  }

  for (auto& input : inputs) {
    const auto& input_name = input->Name();
    auto it = input_replacement_map.find(input_name);
    ORT_RETURN_IF_NOT(it != input_replacement_map.end(), "Missing replacement for input: " + input_name);

    InputReplacement& repl = it->second;
    InlinedVector<NodeArg*> input_args = {repl.graph_input};
    InlinedVector<NodeArg*> output_args = {repl.identity_output};

    std::string node_name = "IdentityInsertion_" + input_name;
    dst_graph.AddNode(node_name, "Identity", "Inserted identity node",
                      input_args, output_args,
                      nullptr, "");
  }

  for (auto pnode : src_graph.Nodes()) {
    if (pnode->NodeType() == Node::Type::Fused) continue;

    InlinedVector<NodeArg*> new_input_args;
    for (auto input_arg : pnode->InputDefs()) {
      if (!input_arg) {
        new_input_args.push_back(nullptr);
        continue;
      }

      auto it = input_replacement_map.find(input_arg->Name());
      if (it != input_replacement_map.end()) {
        new_input_args.push_back(it->second.identity_output);
      } else {
        auto& new_arg = dst_graph.GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
        new_input_args.push_back(&new_arg);
      }
    }
    InlinedVector<NodeArg*> new_output_args;
    for (auto output_arg : pnode->OutputDefs()) {
      if (output_arg == nullptr) {
        new_output_args.push_back(nullptr);
        continue;
      }

      auto it_output = output_replacement_map.find(output_arg->Name());
      if (it_output != output_replacement_map.end()) {
        new_output_args.push_back(it_output->second.intermediate_arg);
      } else {
        auto& new_arg = dst_graph.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
        new_output_args.push_back(&new_arg);
      }
    }

    dst_graph.AddNode(pnode->Name(), pnode->OpType(), pnode->Description(),
                      new_input_args, new_output_args,
                      &pnode->GetAttributes(), pnode->Domain());
  }

  for (auto& output : outputs) {
    const std::string& output_name = output->Name();
    auto it = output_replacement_map.find(output_name);
    if (it == output_replacement_map.end()) continue;

    OutputReplacement& repl = it->second;
    InlinedVector<NodeArg*> input_args = {repl.intermediate_arg};
    InlinedVector<NodeArg*> output_args = {repl.original_output};

    std::string node_name = "IdentityInsertion_" + output_name;
    dst_graph.AddNode(node_name, "Identity", "Inserted identitynode",
                      input_args, output_args, nullptr, "");
  }

  for (auto& [name, tensor_proto] : src_graph.GetAllInitializedTensors()) {
    dst_graph.AddInitializedTensor(*tensor_proto);
  }

  for (auto node_arg : src_graph.GetInputsIncludingInitializers()) {
    auto check_inputs = [node_arg](auto input_node_arg) {
      return input_node_arg->Name() == node_arg->Name();
    };
    if (std::find_if(dst_graph_inputs.begin(), dst_graph_inputs.end(), check_inputs) != dst_graph_inputs.end())
      continue;

    auto src_tensor_proto = src_graph.GetConstantInitializer(node_arg->Name(), true);
    if (src_tensor_proto) {
      auto dst_tensor_proto = onnx::TensorProto::Create();
      dst_tensor_proto->copy_from(src_tensor_proto);
      dst_graph.AddInitializedTensor(*dst_tensor_proto);
    }
  }

  ORT_RETURN_IF_ERROR(dst_graph.Resolve());
  return Status::OK();
}

Status Transform(const GraphViewer& src_graph_viewer,
                 const logging::Logger& logger,
                 /*out*/ std::unique_ptr<onnxruntime::Model>& model) {
  auto status = copy_model(src_graph_viewer, logger, model);
  auto g = generate_graph_from_onnx(model->MainGraph());

  float threshold{1.f};
  float ratio{10.f};
  bool scale_output{false};
  auto needs_second_run = scale_graph(g, threshold, ratio, scale_output);
  if (needs_second_run)
    scale_graph(g, threshold * 100, ratio, scale_output);
  return status;
}
}  // namespace qdq_scales_fix
}  // namespace openvino_ep
}  // namespace onnxruntime
