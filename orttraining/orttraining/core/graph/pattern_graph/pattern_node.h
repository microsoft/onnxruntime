// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/contrib_ops/onnx_function_util.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace training {

/**
 * @brief Pattern graph input data types.
 * Class representing allowed data types for inputs of the pattern graph.
 */
struct PGraphInputTypes {
  friend class PGraphInput;

 public:
  /**
    @brief Predefined categories for common used data type set.
  */
  enum class TypesCategory {
    AllIntegerTensorTypes,
    AllFloatTensorTypes,
  };

 public:
  /* Construct allowed data types from predefined categories. */
  PGraphInputTypes(TypesCategory category);
  /* Construct allowed data types from explicit list of data types. */
  PGraphInputTypes(const std::vector<ONNX_NAMESPACE::TensorProto_DataType>& allowed_types)
      : allowed_types_(allowed_types) {}

 private:
  /**
   * We need a valid data type for the input when building the pattern graph.
   * Here we choose the first data type in the allowed list. For user of this class,
   * default type is meaningless, and will not be used as strict type checking,
   * instead, if target input/output arg's data type in allowed_types_, then it is a match.
   */
  ONNX_NAMESPACE::TensorProto_DataType GetDefaultType() const {
    ORT_ENFORCE(allowed_types_.size(), "Empty type list in PGraphInputTypes.");
    return allowed_types_.at(0);
  }

  void Init(const std::vector<ONNX_NAMESPACE::TensorProto_DataType>& types) {
    allowed_types_ = types;
  }

  std::vector<ONNX_NAMESPACE::TensorProto_DataType> allowed_types_;
};

struct PGraphInputShape {
  friend class PGraphInput;

 public:
  PGraphInputShape(const std::vector<std::vector<std::string>>& allowed_shapes)
      : allowed_symbolic_shapes_(allowed_shapes) {
  }

  bool CanBeAnyShape() const {
    return allowed_symbolic_shapes_.empty();
  }

 private:
  std::vector<std::vector<std::string>> allowed_symbolic_shapes_;
};

/**
 * @brief Pattern graph input description.
 * Class representing a graph input to a pattern graph. Two kinds of graph inputs are supported:
 * > Graph inputs, which is generated from other subgraphs.
 * > Constant node, which consume no else, but generate a constant value.
 *
 * During node matching, by default, all user given fields will be checked, compared with
 * the target graph.
 */
struct PGraphInput {
  friend class PatternGraph;

 public:
  /*
   * is_dangling: if it's set to false, this arg would not be required  to match an arg in target graph
   * is_constant: it indicates if the arg is a constant.
   * Since we reuse Augumenter to implement the pattern graph, we cannot get real constant but only use a flag.
   */
  PGraphInput(const std::string& output_arg_name,
              const PGraphInputTypes& type,
              const PGraphInputShape& shape,
              bool is_dangling = true,
              bool is_constant = true)
      : output_arg_name_(output_arg_name),
        is_constant_(is_constant),
        is_dangling_(is_dangling),
        allowed_types_(type.allowed_types_) {
    SetTensorProto(type.GetDefaultType());
    if (shape.CanBeAnyShape()) {
      t_proto_.add_dims(1);
    } else {
      // for (auto dim : shape.allowed_symbolic_shapes_[0]) {
      //   t_proto_.add_dims(dim);
      //   set_dim_param
      // }
    }
  }

  PGraphInput(const std::string& output_arg_name,
              const PGraphInputTypes& type,
              int rank = 1,
              bool is_dangling = true,
              bool is_constant = true)
      : output_arg_name_(output_arg_name),
        is_constant_(is_constant),
        is_dangling_(is_dangling),
        allowed_types_(type.allowed_types_) {
    SetTensorProto(type.GetDefaultType());
    while (rank--) {
      t_proto_.add_dims(1);
    }
  }

  bool MatchesDataType(const Graph& graph, const NodeArg& input_arg) const;
  bool MatchesShape(const Graph& graph, const NodeArg& input_arg) const;

  std::string GetArgName() const {
    return output_arg_name_;
  }

  bool IsConstant() const {
    return is_constant_;
  }

  bool IsDangling() const {
    return is_dangling_;
  }

 private:
  /**
   * @brief Get the NodeDef object, which will be used by GraphAugmenter.
   *
   * @return NodeDef
   */
  NodeDef GetNodeDef() const {
    /**
     * We used a trick here:
     * For graph inputs, no matter where it is specified as a Constant Node or not in the constructor,
     * we create Constant node here. This is a simpler way to define such data type and shape,
     * then the underlying GraphArgument can take it as graph inputs, and build graphs successfully.
     *
     * Be noted, though we give the concrete data for those Constant node, but we won't use it during
     * node matching.
     */
    return NodeDef(
        "Constant",
        {},
        {ArgDef(output_arg_name_, nullptr)},
        {ONNX_NAMESPACE::MakeAttribute("value", t_proto_)});
  }

  /**
   * We need to set the value to help to build the graph but the value would not be used in matching.
   * So we assign 0 to the tensor value here.
   */
  void SetTensorProto(ONNX_NAMESPACE::TensorProto_DataType type);

 private:
  std::string output_arg_name_;
  bool is_constant_;
  bool is_dangling_;
  TensorProto t_proto_;
  std::vector<ONNX_NAMESPACE::TensorProto_DataType> allowed_types_;
};

/**
 * @brief Pattern graph node description.
 * Class representing a graph node to a pattern graph.
 *
 * During node matching, by default, all user given fields will be checked, compared with
 * the target node.
 */
struct PGraphNode {
  friend class DefaultNodeCompareFunc;
  friend class PatternGraph;

 public:
  /**
   * Create a node using mostly string-based description, among them:
   * @param op_type ONNX operator name.
   * @param input_args_names input arg names (generated by either PGraphInput or other PGraphNode as output_arg_name),
   *    which this operator takes as inputs.
   * @param output_args_names output arg names, which this operator generates as outputs.
   * @param node_name [optional] a unique string (in the same pattern graph) representing name of the node.
   *    If not specified, will automatically assigned.
   * @param domain_version_maps [optional] a list of allowed opset domains/versions.
   *    If not specified, node compare will ignore domain/version check.
   * @param attributes [optional] attributes that will be used to compare during node compare.
   * @param output_edges_count [optional] the output edge out that will be used in node compare.
   *    if it remains default, it will decide the output_edges_count by the generated onnx graph.
   *    If it is set to negative, then no check for output edges count will be taken in the node.
   *    If it is set to positive, then we will check the corresponding node in target graph that if it has that number output edges.
   *    (todo): we need consider the output edge count constraint on each output of the node. Now we
   *        only compare the total edge count.
   */
  PGraphNode(const std::string& op_type,
             const std::vector<std::string>& input_args_names,
             const std::vector<std::string>& output_args_names,
             const std::string& node_name = "",
             const std::unordered_map<std::string, std::vector<int>>& domain_version_maps = {},
             const std::vector<AttributeProto>& attributes = {},
             int output_edges_count = 0,
             bool ignore_input_arg_orders = false)
      : op_type_(op_type),
        input_args_names_(input_args_names),
        output_args_names_(output_args_names),
        node_name_(node_name),
        output_edges_count_(output_edges_count),
        domain_version_maps_(domain_version_maps),
        attributes(attributes),
        ignore_order_(ignore_input_arg_orders) {
    if (node_name.empty()) {
      CreateNodeName();
    }
  }

  const std::string& GetNodeName() const {
    return node_name_;
  }

  bool NameEquals(const std::string& name) const {
    return node_name_.compare(name) == 0;
  }

  bool MatchesOpType(const std::string& op_type) const {
    return op_type_.compare(op_type) == 0;
  }

  bool MatchesDomainVersion(const std::string& domain, const int version) const;

  bool IgnoreNodeArgOrder() const {
    return ignore_order_;
  }

  const std::unordered_map<std::string, std::vector<int>>& GetDomainVersionMap() const { return domain_version_maps_; }

  /**
   * @brief Get the NodeDef object, which will be used by GraphAugmenter.
   *
   * @return NodeDef
   */
  NodeDef GetNodeDef() const;

 private:
  void CreateNodeName();

  std::string op_type_;
  std::vector<std::string> input_args_names_;
  std::vector<std::string> output_args_names_;
  std::string node_name_;
  int output_edges_count_;
  std::unordered_map<std::string, std::vector<int>> domain_version_maps_;
  std::vector<AttributeProto> attributes;
  bool ignore_order_;
};

}  // namespace training
}  // namespace onnxruntime
