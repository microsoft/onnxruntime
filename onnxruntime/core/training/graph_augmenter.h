// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <string>
#include <vector>
#include "core/graph/basic_types.h"
#include "core/common/status.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace training {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::TypeProto;

struct ArgDef {
  ArgDef(std::string name, const TypeProto* type = nullptr) : name(name), type_proto(type) {}
  std::string name;
  const TypeProto* type_proto;

  bool operator==(const ArgDef& other) const {
    return name == other.name;
  }
};

struct OpDef {
  OpDef(const std::string& type, const std::string& domain = kOnnxDomain)
      : type(type),
        domain(domain){};

  std::string type;
  std::string domain;
};

struct NodeDef {
  NodeDef(const OpDef& op_def,
          const std::vector<ArgDef>& input_args,
          const std::vector<ArgDef>& output_args,
          const NodeAttributes& attributes = NodeAttributes(),
          const std::string& name = "") : op_type(op_def.type),
                                          domain(op_def.domain),
                                          input_args(input_args),
                                          output_args(output_args),
                                          attributes(attributes),
                                          name(name){};

  NodeDef(const std::string& op_type,
          const std::vector<ArgDef>& input_args,
          const std::vector<ArgDef>& output_args,
          const NodeAttributes& attributes = NodeAttributes(),
          const std::string& name = "") : op_type(op_type),
                                          input_args(input_args),
                                          output_args(output_args),
                                          attributes(attributes),
                                          name(name){};

  NodeDef(const std::string& op_type,
          const std::vector<ArgDef>& input_args,
          const std::vector<ArgDef>& output_args,
          const std::vector<AttributeProto>& attribute_protos,
          const std::string& name = "") : op_type(op_type),
                                          input_args(input_args),
                                          output_args(output_args),
                                          name(name) {
    for (const AttributeProto& a : attribute_protos) {
      attributes.insert({a.name(), a});
    }
  }

  std::string op_type;
  std::string domain = kOnnxDomain;
  std::vector<ArgDef> input_args;
  std::vector<ArgDef> output_args;
  NodeAttributes attributes;
  std::string name;
};

/** GraphAugmenter is a stateless class to add new elements into a Graph.
    The elements to be added could be:
    1. Nodes
    2. Outputs
       Note: during Graph::Resolve(), input and output will be infered from the nodes, in which:
             1. A node arg becomes a graph input if it is not used by any node's output.
             2. A node arg becomes a graph output if it is not used by any node's input.
             So we don't have to worry about input, but sometimes need to explicitly 
             set an intermediate node arg as graph output.
    3. Initializers    
*/
class GraphAugmenter {
 public:
  class GraphDefs {
   public:
    void AddNodeDefs(const std::vector<NodeDef>& node_defs) {
      for (auto node_def : node_defs) {
        // Copy constant node value to graph_initializers_
        if (node_def.op_type == kConstant) {
          TensorProto initializer = node_def.attributes.at("value").t();
          initializer.set_name(node_def.output_args[0].name);
          graph_initializers_.push_back(initializer);
        } else {
          node_defs_.push_back(node_def);
        }
      }
    }

    const std::vector<NodeDef>& NodeDefs() const {
      return node_defs_;
    }

    std::vector<NodeDef>& NodeDefs() {
      return node_defs_;
    }

    void AddGraphOutputs(const std::vector<std::string>& names) {
      graph_output_names_.insert(graph_output_names_.end(), names.begin(), names.end());
    }

    const std::vector<std::string>& GraphOutputs() const {
      return graph_output_names_;
    }

    void AddInitializers(const std::vector<TensorProto>& tensors) {
      graph_initializers_.insert(graph_initializers_.end(), tensors.begin(), tensors.end());
    }

    const std::vector<TensorProto>& Initializers() const {
      return graph_initializers_;
    }

   private:
    std::vector<NodeDef> node_defs_;
    std::vector<std::string> graph_output_names_;
    std::vector<TensorProto> graph_initializers_;
  };

  // Augment the graph with new_graph_elements which defines new nodes, outputs, initializers.
  static common::Status AugmentGraph(Graph& graph, const GraphDefs& graph_element_defs);
};
}  // namespace training
}  // namespace onnxruntime
