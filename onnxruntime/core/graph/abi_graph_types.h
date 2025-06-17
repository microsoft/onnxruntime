// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"

#define DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(external_type, internal_type, internal_api) \
  external_type* ToExternal() { return static_cast<external_type*>(this); }                        \
  const external_type* ToExternal() const { return static_cast<const external_type*>(this); }      \
  static internal_type* ToInternal(external_type* e) {                                             \
    return e->graph_ir_api == (internal_api) ? static_cast<internal_type*>(e) : nullptr;           \
  }                                                                                                \
  static const internal_type* ToInternal(const external_type* e) {                                 \
    return e->graph_ir_api == (internal_api) ? static_cast<const internal_type*>(e) : nullptr;     \
  }

// The public ORT graph IR types (e.g., OrtGraph, OrtNode, etc.) have different implementations for the
// ModelEditor API and EP API. This enum allows a user of the base class (e.g., OrtGraph) to determine
// the API for which the derived class was created.
enum class OrtGraphIrApi {
  kInvalid = 0,
  kModelEditorApi,
  kEpApi,
};

/// <summary>
/// Public type that represents an ONNX value info.
/// </summary>
struct OrtValueInfo {
  explicit OrtValueInfo(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtValueInfo() = default;

  /// <summary>
  /// Returns the value's name.
  /// </summary>
  /// <returns>The value's name.</returns>
  virtual const std::string& GetName() const = 0;

  /// <summary>
  /// Return's an object describing the value's type and shape.
  /// </summary>
  /// <returns>OrtTypeInfo with the type and shape.</returns>
  virtual const OrtTypeInfo* GetTypeInfo() const = 0;

  struct ProducerInfo {
    ProducerInfo() = default;
    ProducerInfo(const OrtNode* node, size_t output_index) : node(node), output_index(output_index) {}
    const OrtNode* node = nullptr;
    size_t output_index = 0;
  };

  /// <summary>
  /// Returns the node (and output index) that produced the value.
  /// </summary>
  /// <param name="producer_info">Output parameter set to the node and the output index that produced the value.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetProducerInfo(ProducerInfo& producer_info) const = 0;

  struct ConsumerInfo {
    ConsumerInfo() = default;
    ConsumerInfo(const OrtNode* node, int64_t input_index) : node(node), input_index(input_index) {}
    const OrtNode* node = nullptr;
    int64_t input_index = 0;  // Negative if it is an implicit input to a node that contains a subgraph (e.g., If).
  };

  /// <summary>
  /// Returns information on the nodes that consume the value. Includes each consumer node's input index,
  /// which could be -1 for an implicit input to the node (e.g., If or Loop node).
  /// </summary>
  /// <param name="consumer_infos">Output parameter set to the array of ConsumerInfo objects that describe the
  ///                              use of this value (consumer node and input index).</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetConsumerInfos(std::vector<ConsumerInfo>& consumer_infos) const = 0;

  /// <summary>
  /// Returns the number of consumers for this value. In this context, a consumer is a tuple of the node and the input
  /// index that uses the value.
  /// </summary>
  /// <param name="num_consumers">Output parameter set to the number of consumers.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNumConsumerInfos(size_t& num_consumers) const = 0;

  /// <summary>
  /// Returns the associated initializer value if this value represents an initializer (constant or non-constant).
  /// </summary>
  /// <param name="value">Output parameter set to the initializer value or nullptr if this value is not
  ///                     an initializer.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetInitializerValue(const OrtValue*& value) const = 0;

  /// <summary>
  /// Determine if the value is a required graph input.
  /// </summary>
  /// <param name="is_required_graph_input">Output parameter set to true if the value is a required graph input.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status IsRequiredGraphInput(bool& is_required_graph_input) const = 0;

  /// <summary>
  /// Determine if the value is an optional graph input.
  /// </summary>
  /// <param name="is_optional_graph_input">Output parameter set to true if the value is an optional graph
  ///                                       input.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status IsOptionalGraphInput(bool& is_optional_graph_input) const = 0;

  /// <summary>
  /// Determine if a the value is a graph output.
  /// </summary>
  /// <param name="is_graph_output">Output parameter set to true if the value is a graph output.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status IsGraphOutput(bool& is_graph_output) const = 0;

  /// <summary>
  /// Determine if the value is a constant initializer.
  /// </summary>
  /// <param name="is_const_initializer">Output parameter set to true if the value is a constant
  ///                                    initializer.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status IsConstantInitializer(bool& is_const_initializer) const = 0;

  /// <summary>
  /// Determine if the value is defined in an outer scope (i.e., a parent graph).
  /// </summary>
  /// <param name="is_outer_scope">Output parameter set to true if the value is defined in an outer scope.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status IsFromOuterScope(bool& is_outer_scope) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};

/// <summary>
/// Public type that represents an ONNX attribute. Currently, an OrtOpAttr is interchangeable with AttributeProto.
/// </summary>
struct OrtOpAttr {
  ONNX_NAMESPACE::AttributeProto attr_proto;
};

/// <summary>
/// Public type that represents an ONNX node.
/// </summary>
struct OrtNode {
  explicit OrtNode(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtNode() = default;

  /// <summary>
  /// Returns the node's ID, which is unique in it's graph.
  /// </summary>
  /// <returns>The node's ID.</returns>
  virtual size_t GetId() const = 0;

  /// <summary>
  /// Returns the node's name.
  /// </summary>
  /// <returns>The node's name</returns>
  virtual const std::string& GetName() const = 0;

  /// <summary>
  /// Returns the node's operator type (e.g., "Conv").
  /// </summary>
  /// <returns>The node's operator type.</returns>
  virtual const std::string& GetOpType() const = 0;

  /// <summary>
  /// Returns the node's domain name.
  /// </summary>
  /// <returns>The node's domain name.</returns>
  virtual const std::string& GetDomain() const = 0;

  /// <summary>
  /// Gets the opset version in which the node's operator type was first defined.
  /// </summary>
  /// <param name="since_version">Output parameter set to the node's operator "since version".</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetSinceVersion(int& since_version) const = 0;

  /// <summary>
  /// Gets the node's inputs as an array of OrtValueInfo elements wrapped in an OrtConstPointerArray.
  /// </summary>
  /// <param name="inputs">Output parameter set to the node's inputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetInputs(const OrtConstPointerArray*& inputs) const = 0;

  /// <summary>
  /// Gets the node's outputs as an array of OrtValueInfo elements wrapped in an OrtConstPointerArray.
  /// </summary>
  /// <param name="outputs">Output parameter set to the node's outputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetOutputs(const OrtConstPointerArray*& outputs) const = 0;

  /// <summary>
  /// Gets the node's implicit inputs as an array of OrtValueInfo elements wrapped in an OrtConstPointerArray.
  /// Applies to a node that contains a subgraph (e.g., If or Loop). An implicit input is a value consumed by an
  /// internal subgraph node that is not defined in the subgraph.
  /// </summary>
  /// <param name="implicit_inputs">Output parameter set to the node's implicit inputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetImplicitInputs(const OrtConstPointerArray*& implicit_inputs) const = 0;

  /// <summary>
  /// Gets the node's number of attributes.
  /// </summary>
  /// <param name="num_attrs">Output parameter set to number of attributes contained by the node.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNumAttributes(size_t& num_attrs) const = 0;

  /// <summary>
  /// Gets the node's attributes as an array of OrtValueInfo elements wrapped in an OrtConstPointerArray.
  /// </summary>
  /// <param name="attrs">Output parameter set to the node's attributes.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetAttributes(const OrtConstPointerArray*& attrs) const = 0;

  /// <summary>
  /// Gets the node's number of subgraphs.
  /// </summary>
  /// <param name="num_subgraphs">Output parameter set to number of subgraphs contained by the node.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNumSubgraphs(size_t& num_subgraphs) const = 0;

  /// <summary>
  /// Gets the node's subgraphs (e.g., subgraphs contained by an If or Loop node).
  /// </summary>
  /// <param name="subgraphs">Output parameter set to the node's subgraphs as OrtGraph instances.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetSubgraphs(onnxruntime::InlinedVector<const OrtGraph*>& subgraphs) const = 0;

  /// <summary>
  /// Gets the node's parent graph, which is the graph that contains this node.
  /// </summary>
  /// <param name="parent_graph">Output parameter set to the node's parent graph.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetParentGraph(const OrtGraph*& parent_graph) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};

/// <summary>
/// Public type that represents an ONNX graph.
/// </summary>
struct OrtGraph {
  explicit OrtGraph(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtGraph() = default;

  /// <summary>
  /// Returns the graph's name.
  /// </summary>
  /// <returns>The graph's name.</returns>
  virtual const std::string& GetName() const = 0;

  /// <summary>
  /// Returns the model's ONNX IR version. Important in checking for optional graph inputs
  /// (aka non-constant initializers), which were introduced in ONNX IR version 4.
  /// </summary>
  /// <returns>The model's ONNX IR version.</returns>
  virtual int64_t GetOnnxIRVersion() const = 0;

  /// <summary>
  /// Gets the graph's inputs (including initializers) as OrtValueInfo instances wrapped in an OrtConstPointerArray.
  /// </summary>
  /// <param name="inputs">Output parameter set to the graph's inputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetInputs(const OrtConstPointerArray*& inputs) const = 0;

  /// <summary>
  /// Gets the graph's outputs as OrtValueInfo instances wrapped in an OrtConstPointerArray.
  /// </summary>
  /// <param name="inputs">Output parameter set to the graph's outputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetOutputs(const OrtConstPointerArray*& outputs) const = 0;

  /// <summary>
  /// Gets the graph's initializers (both constant and non-constant) as OrtValueInfo instances wrapped in an
  /// OrtConstPointerArray.
  /// </summary>
  /// <param name="initializers">Output parameter set to the graph's initializers.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetInitializers(const OrtConstPointerArray*& initializers) const = 0;

  /// <summary>
  /// Gets the graph's nodes as OrtNode instances wrapped in an OrtConstPointerArray. The nodes are sorted in
  /// a default "reverse DFS" topological order.
  /// </summary>
  /// <param name="nodes">Output parameter set to the graph's nodes.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNodes(const OrtConstPointerArray*& nodes) const = 0;

  /// <summary>
  /// Gets the graph's parent node, if any. The parent_node is nullptr if this is not a nested subgraph.
  /// </summary>
  /// <param name="parent_node">Output parameter set to the parent node.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetParentNode(const OrtNode*& parent_node) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};
