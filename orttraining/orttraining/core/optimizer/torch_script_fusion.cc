// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA

#include "orttraining/core/optimizer/torch_script_fusion.h"

#include "core/framework/compute_capability.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/providers/partitioning_utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

namespace {

using OpSetVersionList = std::initializer_list<OperatorSetVersion>;
using SizeTypeVec = InlinedVector<size_t>;
using StringVec = InlinedVector<std::string>;
using StringMap = InlinedHashMap<std::string, std::string>;
using NodeVec = InlinedVector<Node*>;
using NodeArgVec = InlinedVector<NodeArg*>;
using NodeArgSet = InlinedHashSet<NodeArg*>;
using NodeArgMap = InlinedHashMap<NodeArg*, std::string>;
using IsSupportedFunc = std::function<bool(const Graph&, const Node&)>;
using GetIRFunc = std::function<std::string(const Graph&, const Node&, const StringVec&, StringMap&, StringMap&)>;

int64_t HashScript(const std::string& script) {
  uint32_t hash = 0;
  for (char const& c : script) {
    hash = hash * 101 + c;
  }

  return static_cast<int64_t>(hash);
}

int OnnxDTypeToTorchDType(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return 6;
    case TensorProto_DataType_FLOAT16:
      return 5;
    case TensorProto_DataType_DOUBLE:
      return 7;
    case TensorProto_DataType_INT8:
      return 1;
    case TensorProto_DataType_INT16:
      return 2;
    case TensorProto_DataType_INT32:
      return 3;
    case TensorProto_DataType_INT64:
      return 4;
    case TensorProto_DataType_UINT8:
      return 0;
    case TensorProto_DataType_BOOL:
      return 11;
    default:
      return -1;
  }
}

std::string GetTensorType(const NodeArg* node_arg) {
  const TypeProto* type = node_arg->TypeAsProto();
  if (!type) return "Tensor";
  switch (type->tensor_type().elem_type()) {
    case TensorProto_DataType_FLOAT:
      return "Float()";
    case TensorProto_DataType_FLOAT16:
      return "Half()";
    case TensorProto_DataType_DOUBLE:
      return "Double()";
    case TensorProto_DataType_INT8:
      return "Char()";
    case TensorProto_DataType_INT16:
      return "Short()";
    case TensorProto_DataType_INT32:
      return "Int()";
    case TensorProto_DataType_INT64:
      return "Long()";
    case TensorProto_DataType_UINT8:
      return "Byte()";
    case TensorProto_DataType_BOOL:
      return "Bool()";
    default:
      return "Tensor";
  }
}

int64_t GetIntAttrOrDefault(const Node& node, const std::string& attr_name, int64_t default_value) {
  const auto& attrs = node.GetAttributes();
  if (attrs.find(attr_name) != attrs.end()) {
    auto& attr = attrs.at(attr_name);
    if (utils::HasInt(attr)) return attr.i();
  }
  return default_value;
}

int GetNodeArgDType(const NodeArg* node_arg) {
  const TypeProto* type = node_arg->TypeAsProto();
  if (!type) return TensorProto_DataType_UNDEFINED;
  return type->tensor_type().elem_type();
}

bool GetIntScalarConstant(const Graph& graph, const NodeArg* node_arg, int64_t& value) {
  if (!optimizer_utils::IsScalar(*node_arg)) return false;
  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node_arg->Name());
  if (!tensor_proto) return false;
  Initializer init_const{*tensor_proto, graph.ModelPath()};
  if (tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) return false;
  value = *(init_const.data<int64_t>());
  return true;
}

template <typename... Args>
std::string FormatString(const std::string& format, Args... args) {
  int size_int = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
  ORT_ENFORCE(size_int > 0, "Fail to format string.");
  size_t size = static_cast<size_t>(size_int);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

std::string GetIntConstantIR(StringMap& constants, int value) {
  std::string value_str = std::to_string(value);
  std::string name = "%cint" + value_str;
  std::replace(name.begin(), name.end(), '-', 'n');
  if (constants.find(name) == constants.end()) {
    constants.emplace(name, FormatString("%s: int = prim::Constant[value=%s]()", name.c_str(), value_str.c_str()));
  }
  return name;
}

std::string GetBoolConstantIR(StringMap& constants, bool value) {
  std::string name = value ? "%cbool1" : "%cbool0";
  std::string value_str = value ? "1" : "0";
  if (constants.find(name) == constants.end()) {
    constants.emplace(name, FormatString("%s: bool = prim::Constant[value=%s]()", name.c_str(), value_str.c_str()));
  }
  return name;
}

std::string GetNoneConstantIR(StringMap& constants) {
  std::string name = "%cnone";
  if (constants.find(name) == constants.end()) {
    constants.emplace(name, "%cnone: None = prim::Constant()");
  }
  return name;
}

struct OpInfo {
  OpInfo(const char* op_type, const OpSetVersionList& supported_versions, const char* domain, const bool is_no_op,
         const SizeTypeVec& input_mapping, IsSupportedFunc is_supported_func, GetIRFunc get_ir_func)
      : op_type_(op_type),
        supported_versions_(supported_versions),
        domain_(domain),
        is_no_op_(is_no_op),
        input_mapping_(input_mapping),
        is_supported_func_(is_supported_func),
        get_ir_func_(get_ir_func){};

  std::string op_type_;
  OpSetVersionList supported_versions_;
  std::string domain_;
  bool is_no_op_;
  SizeTypeVec input_mapping_;
  IsSupportedFunc is_supported_func_;
  GetIRFunc get_ir_func_;
};

const OpSetVersionList OpSetV1 = {1};
const OpSetVersionList OpSetV13_14 = {13, 14};
const OpSetVersionList OpSetV9 = {9};
const OpSetVersionList OpSetV13 = {13};
IsSupportedFunc default_is_supported = [](const Graph&, const Node&) { return true; };
const InlinedHashMap<std::string, OpInfo> kSupportedOps{
    {"Add", OpInfo("Add", OpSetV13_14, kOnnxDomain, false, {0, 1}, default_is_supported,
                   [](const Graph&, const Node&, const StringVec& inputs, StringMap& constants, StringMap&) {
                     return FormatString("aten::add(%s, %s, %s)", inputs[0].c_str(), inputs[1].c_str(),
                                         GetIntConstantIR(constants, 1).c_str());
                   })},
    {"Sub", OpInfo("Sub", OpSetV13_14, kOnnxDomain, false, {0, 1}, default_is_supported,
                   [](const Graph&, const Node&, const StringVec& inputs, StringMap& constants, StringMap&) {
                     return FormatString("aten::sub(%s, %s, %s)", inputs[0].c_str(), inputs[1].c_str(),
                                         GetIntConstantIR(constants, 1).c_str());
                   })},
    {"Mul", OpInfo("Mul", OpSetV13_14, kOnnxDomain, false, {0, 1}, default_is_supported,
                   [](const Graph&, const Node&, const StringVec& inputs, StringMap&, StringMap&) {
                     return FormatString("aten::mul(%s, %s)", inputs[0].c_str(), inputs[1].c_str());
                   })},
    {"Div", OpInfo("Div", OpSetV13_14, kOnnxDomain, false, {0, 1}, default_is_supported,
                   [](const Graph&, const Node&, const StringVec& inputs, StringMap&, StringMap&) {
                     return FormatString("aten::div(%s, %s)", inputs[0].c_str(), inputs[1].c_str());
                   })},
    // Where's 1st input is bool but PyTorch's DLPack doesn't support bool so extra aten::to will be introduced.
    // Profling result also shows that Where cannot be fused with other Ops for most of the time.
    // {"Where", OpInfo("Where", OpSetV9, kOnnxDomain, false, {0, 1, 2}, default_is_supported,
    //                  [](const Graph&, const Node&, const StringVec& inputs, StringMap&, StringMap&) {
    //                    return FormatString("aten::where(%s, %s, %s)", inputs[0].c_str(), inputs[1].c_str(),
    //                                        inputs[2].c_str());
    //                  })},
    {"Cast", OpInfo(
                 "Cast", OpSetV13, kOnnxDomain, false, {0},
                 [](const Graph&, const Node& node) {
                   return OnnxDTypeToTorchDType(static_cast<int>(GetIntAttrOrDefault(node, "to", 0))) != -1;
                 },
                 [](const Graph&, const Node& node, const StringVec& inputs, StringMap& constants, StringMap&) {
                   int torch_dtype = OnnxDTypeToTorchDType(static_cast<int>(GetIntAttrOrDefault(node, "to", 0)));
                   std::string bool0_name = GetBoolConstantIR(constants, false);
                   return FormatString("aten::to(%s, %s, %s, %s, %s)", inputs[0].c_str(),
                                       GetIntConstantIR(constants, torch_dtype).c_str(), bool0_name.c_str(),
                                       bool0_name.c_str(), GetNoneConstantIR(constants).c_str());
                 })},
    {"Reshape", OpInfo("Reshape", OpSetV13_14, kOnnxDomain, true, {0, 1}, default_is_supported,
                       [](const Graph&, const Node&, const StringVec& inputs, StringMap&, StringMap& cpu_input_types) {
                         cpu_input_types[inputs[1]] = "int[]";
                         return FormatString("aten::view(%s, %s)", inputs[0].c_str(), inputs[1].c_str());
                       })},
    {"Squeeze",
     OpInfo(
         "Squeeze", OpSetV13, kOnnxDomain, true, {0},
         [](const Graph& graph, const Node& node) {
           int64_t axis;
           return GetIntScalarConstant(graph, node.InputDefs()[1], axis);
         },
         [](const Graph& graph, const Node& node, const StringVec& inputs, StringMap& constants, StringMap&) {
           int64_t axis;
           GetIntScalarConstant(graph, node.InputDefs()[1], axis);
           return FormatString("aten::squeeze(%s, %s)", inputs[0].c_str(),
                               GetIntConstantIR(constants, static_cast<int>(axis)).c_str());
         })},
    {"Unsqueeze",
     OpInfo(
         "Unsqueeze", OpSetV13, kOnnxDomain, true, {0},
         [](const Graph& graph, const Node& node) {
           int64_t axis;
           return GetIntScalarConstant(graph, node.InputDefs()[1], axis);
         },
         [](const Graph& graph, const Node& node, const StringVec& inputs, StringMap& constants, StringMap&) {
           int64_t axis;
           GetIntScalarConstant(graph, node.InputDefs()[1], axis);
           return FormatString("aten::unsqueeze(%s, %s)", inputs[0].c_str(),
                               GetIntConstantIR(constants, static_cast<int>(axis)).c_str());
         })},
    {"Softmax", OpInfo(
                    "Softmax", OpSetV13, kOnnxDomain, false, {0},
                    [](const Graph&, const Node& node) {
                      return OnnxDTypeToTorchDType(GetNodeArgDType(node.OutputDefs()[0])) != -1;
                    },
                    [](const Graph&, const Node& node, const StringVec& inputs, StringMap& constants, StringMap&) {
                      int axis = static_cast<int>(GetIntAttrOrDefault(node, "axis", -1));
                      int torch_dtype = OnnxDTypeToTorchDType(GetNodeArgDType(node.OutputDefs()[0]));
                      return FormatString("aten::softmax(%s, %s, %s)", inputs[0].c_str(),
                                          GetIntConstantIR(constants, axis).c_str(),
                                          GetIntConstantIR(constants, torch_dtype).c_str());
                    })},
    {"SoftmaxGrad_13",
     OpInfo(
         "SoftmaxGrad_13", OpSetV1, kMSDomain, false, {0, 1},
         [](const Graph&, const Node& node) {
           return OnnxDTypeToTorchDType(GetNodeArgDType(node.InputDefs()[0])) != -1;
         },
         [](const Graph&, const Node& node, const StringVec& inputs, StringMap& constants, StringMap&) {
           int axis = static_cast<int>(GetIntAttrOrDefault(node, "axis", -1));
           int torch_dtype = OnnxDTypeToTorchDType(GetNodeArgDType(node.OutputDefs()[0]));
           return FormatString("aten::_softmax_backward_data(%s, %s, %s, %s)", inputs[0].c_str(), inputs[1].c_str(),
                               GetIntConstantIR(constants, axis).c_str(),
                               GetIntConstantIR(constants, torch_dtype).c_str());
         })},
};

NodeArgVec GetIR(const Graph& graph, Node& node, NodeArgMap& inputs, NodeArgMap& outputs, StringMap& constants,
                 StringMap& cpu_input_types, StringVec& irs) {
  const OpInfo& op_info = kSupportedOps.at(node.OpType());
  NodeArgVec new_inputs;
  StringVec input_names;
  for (auto& input_idx : op_info.input_mapping_) {
    auto& input = node.MutableInputDefs()[input_idx];
    if (inputs.find(input) != inputs.end()) {
      input_names.emplace_back(inputs[input]);
    } else if (outputs.find(input) != outputs.end()) {
      input_names.emplace_back(outputs[input]);
    } else {
      std::string input_name = std::to_string(inputs.size());
      if (input_name.size() == 1) {
        input_name = "0" + input_name;
      }
      input_name = "%i" + input_name;
      inputs.emplace(input, input_name);
      new_inputs.emplace_back(input);
      input_names.emplace_back(input_name);
    }
  }

  StringVec output_names;
  StringVec output_types;
  for (const auto& output : node.MutableOutputDefs()) {
    std::string output_name = std::to_string(outputs.size());
    if (output_name.size() == 1) {
      output_name = "0" + output_name;
    }
    output_name = "%t" + output_name;
    outputs.emplace(output, output_name);
    output_names.emplace_back(output_name);
    output_types.emplace_back(GetTensorType(output));
  }

  std::string ir_outputs = FormatString("%s: %s", output_names[0].c_str(), output_types[0].c_str());
  for (size_t i = 1; i < output_names.size(); ++i) {
    ir_outputs += FormatString(", %s: %s", output_names[i].c_str(), output_types[i].c_str());
  }

  irs.emplace_back(FormatString("%s = %s", ir_outputs.c_str(),
                                op_info.get_ir_func_(graph, node, input_names, constants, cpu_input_types).c_str()));
  return new_inputs;
}

std::string GetGraphIR(const StringVec& graph_input_names, const StringVec& graph_input_types,
                       const StringVec& graph_output_names, const StringMap& constants, const StringVec& irs) {
  std::string graph_inputs = FormatString("%s: %s", graph_input_names[0].c_str(), graph_input_types[0].c_str());
  for (size_t i = 1; i < graph_input_names.size(); ++i) {
    graph_inputs += FormatString(", %s: %s", graph_input_names[i].c_str(), graph_input_types[i].c_str());
  }
  std::string graph_outputs = graph_output_names[0];
  for (size_t i = 1; i < graph_output_names.size(); ++i) {
    graph_outputs += FormatString(", %s", graph_output_names[i].c_str());
  }
  std::string graph_ir = FormatString("graph(%s):\n", graph_inputs.c_str());
  StringVec constants_key;
  for (const auto& pair : constants) {
    constants_key.emplace_back(pair.first);
  }
  std::sort(constants_key.begin(), constants_key.end());
  for (const auto& key : constants_key) {
    graph_ir += FormatString("    %s\n", constants.at(key).c_str());
  }
  for (const auto& ir : irs) {
    graph_ir += FormatString("    %s\n", ir.c_str());
  }
  graph_ir += FormatString("    return (%s)\n", graph_outputs.c_str());
  return graph_ir;
}

struct Partition {
  NodeVec nodes;
  NodeArgSet outputs;
  NodeArgSet dependencies;
  size_t output_ref_count;

  void MergeFrom(const Partition& other) {
    nodes.insert(nodes.end(), other.nodes.begin(), other.nodes.end());
    outputs.insert(other.outputs.begin(), other.outputs.end());
    dependencies.insert(other.dependencies.begin(), other.dependencies.end());
    output_ref_count += other.output_ref_count;
  }

  bool IsValid() const {
    size_t count = 0;
    for (const auto& node : nodes) {
      if (!kSupportedOps.at(node->OpType()).is_no_op_) {
        ++count;
        if (count >= 2) return true;
      }
    }
    return false;
  }
};

}  // namespace

bool TorchScriptFusion::IsSupportedNode(const Graph& graph, const Node& node) const {
  const auto& op_type = node.OpType();
  if (!graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
      kSupportedOps.find(op_type) == kSupportedOps.end()) {
    return false;
  }

  // PyTorch's DLPack doesn't support bool for now.
  for (auto& input : node.InputDefs()) {
    const TypeProto* type = input->TypeAsProto();
    if (!type || type->tensor_type().elem_type() == TensorProto_DataType_BOOL) {
      return false;
    }
  }
  for (auto& output : node.OutputDefs()) {
    const TypeProto* type = output->TypeAsProto();
    if (!type || type->tensor_type().elem_type() == TensorProto_DataType_BOOL) {
      return false;
    }
  }

  const auto& op_info = kSupportedOps.at(op_type);
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, op_info.op_type_, op_info.supported_versions_,
                                                        op_info.domain_) &&
         op_info.is_supported_func_(graph, node);
}

Status TorchScriptFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  size_t global_id = 0;
  InlinedHashMap<size_t, Partition> partitions;
  InlinedHashMap<size_t, Partition> partitions_to_fuse;
  InlinedHashMap<NodeArg*, size_t> active_outputs;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (!p_node) continue;
    auto& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    bool is_supported = IsSupportedNode(graph, node);
    SizeTypeVec partitions_to_merge;
    for (auto& pair : partitions) {
      auto& partition = pair.second;
      bool connect_to_output = false;
      bool connect_to_dependency = false;
      for (auto& input : node.MutableInputDefs()) {
        if (partition.outputs.find(input) != partition.outputs.end()) {
          partition.output_ref_count--;
          connect_to_output = true;
        }
        if (partition.dependencies.find(input) != partition.dependencies.end()) {
          connect_to_dependency = true;
        }
      }
      if (is_supported && connect_to_output && !connect_to_dependency) {
        partitions_to_merge.emplace_back(pair.first);
      } else if (connect_to_output || connect_to_dependency) {
        for (auto& output : node.MutableOutputDefs()) {
          partition.dependencies.emplace(output);
        }
      }
    }

    if (!partitions_to_merge.empty()) {
      std::sort(partitions_to_merge.begin(), partitions_to_merge.end());
      Partition& dst = partitions.at(partitions_to_merge[0]);
      for (size_t i = partitions_to_merge.size() - 1; i > 0; --i) {
        dst.MergeFrom(partitions.at(partitions_to_merge[i]));
        partitions.erase(partitions_to_merge[i]);
      }

      dst.nodes.emplace_back(&node);
      for (auto& output : node.MutableOutputDefs()) {
        dst.outputs.emplace(output);
      }
      dst.output_ref_count += node.GetOutputEdgesCount();
    } else if (is_supported) {
      Partition partition;
      partition.nodes.emplace_back(&node);
      for (auto& node_def : node.MutableOutputDefs()) {
        partition.outputs.emplace(node_def);
      }
      partition.output_ref_count = node.GetOutputEdgesCount();
      partitions.emplace(global_id++, partition);
    }

    SizeTypeVec partitions_to_erase;
    for (auto& pair : partitions) {
      if (pair.second.output_ref_count == 0) {
        if (pair.second.IsValid()) {
          pair.second.outputs.clear();
          pair.second.dependencies.clear();
          partitions_to_fuse.emplace(pair);
        }
        partitions_to_erase.emplace_back(pair.first);
      }
    }

    for (auto& id : partitions_to_erase) {
      partitions.erase(id);
    }

    for (auto& input : node.MutableInputDefs()) {
      if (active_outputs.find(input) != active_outputs.end()) {
        active_outputs.at(input)--;
        if (active_outputs.at(input) == 0) {
          active_outputs.erase(input);
          for (auto& pair : partitions) {
            pair.second.dependencies.erase(input);
          }
        }
      }
    }

    for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
      NodeArg* output = node.MutableOutputDefs()[it->GetSrcArgIndex()];
      if (active_outputs.find(output) == active_outputs.end()) {
        active_outputs.emplace(output, 1);
      } else {
        active_outputs.at(output)++;
      }
    }
  }

  SizeTypeVec partition_ids;
  for (auto& pair : partitions_to_fuse) {
    partition_ids.emplace_back(pair.first);
  }
  std::sort(partition_ids.begin(), partition_ids.end());

  for (auto& id : partition_ids) {
    auto& partition = partitions_to_fuse.at(id);
    std::cout << "[PARTITION] node number: " << partition.nodes.size() << std::endl;
    for (auto& node : partition.nodes) {
      std::cout << "    " << node->Name() << " " << node->OpType() << std::endl;
    }

    NodeArgVec input_args;
    NodeArgVec output_args;
    NodeArgMap input_names;
    NodeArgMap output_names;
    InlinedHashMap<NodeArg*, size_t> output_ref_counts;
    StringMap constants;
    StringMap cpu_input_types;
    StringVec irs;
    for (auto& p_node : partition.nodes) {
      auto new_inputs = GetIR(graph, *p_node, input_names, output_names, constants, cpu_input_types, irs);
      input_args.insert(input_args.end(), new_inputs.begin(), new_inputs.end());

      for (auto& input : p_node->MutableInputDefs()) {
        if (output_ref_counts.find(input) != output_ref_counts.end()) {
          output_ref_counts.at(input)--;
          if (output_ref_counts.at(input) == 0) {
            output_ref_counts.erase(input);
          }
        }
      }

      for (auto it = p_node->OutputEdgesBegin(), end = p_node->OutputEdgesEnd(); it != end; ++it) {
        NodeArg* output = p_node->MutableOutputDefs()[it->GetSrcArgIndex()];
        if (output_ref_counts.find(output) == output_ref_counts.end()) {
          output_ref_counts.emplace(output, 1);
        } else {
          output_ref_counts.at(output)++;
        }
      }
    }

    StringVec graph_input_names;
    StringVec graph_input_types;
    std::vector<int64_t> cpu_inputs;
    for (size_t i = 0; i < input_args.size(); ++i) {
      std::string input_name = input_names[input_args[i]];
      graph_input_names.emplace_back(input_name);
      if (cpu_input_types.find(input_name) != cpu_input_types.end()) {
        graph_input_types.emplace_back(cpu_input_types.at(input_name));
        cpu_inputs.emplace_back(static_cast<int64_t>(i));
      } else {
        graph_input_types.emplace_back(GetTensorType(input_args[i]));
      }
    }

    StringVec graph_output_names;
    InlinedHashMap<std::string, NodeArg*> output_name_to_node_args;
    for (auto& pair : output_ref_counts) {
      std::string output_name = output_names[pair.first];
      graph_output_names.emplace_back(output_name);
      output_name_to_node_args.emplace(output_name, pair.first);
    }
    std::sort(graph_output_names.begin(), graph_output_names.end());
    for (const auto& output_name : graph_output_names) {
      output_args.emplace_back(output_name_to_node_args.at(output_name));
    }

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("TorchScript"), "TorchScript",
                                     "Fused nodes for TorchScript", input_args, output_args, {}, kMSDomain);
    std::string graph_ir = GetGraphIR(graph_input_names, graph_input_types, graph_output_names, constants, irs);
    std::cout << "[GRAPH] key: " << HashScript(graph_ir) << ", name: " << fused_node.Name() << std::endl
              << graph_ir << std::endl;
    fused_node.AddAttribute("key", HashScript(graph_ir));
    fused_node.AddAttribute("script", graph_ir);
    if (!cpu_inputs.empty()) {
      fused_node.AddAttribute("cpu_inputs", cpu_inputs);
    }
    fused_node.SetExecutionProviderType(partition.nodes[0]->GetExecutionProviderType());

    for (auto& p_node : partition.nodes) {
      graph_utils::RemoveNodeOutputEdges(graph, *p_node);
      graph.RemoveNode(p_node->Index());
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // USE_CUDA
