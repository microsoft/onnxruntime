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
  /// Returns the number of node inputs.
  /// </summary>
  /// <returns>The number of node inputs.</returns>
  virtual size_t GetNumInputs() const = 0;

  /// <summary>
  /// Gets the node's inputs as OrtValueInfo instances.
  /// </summary>
  /// <param name="inputs">Buffer into which to copy the inputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetInputs(gsl::span<const OrtValueInfo*> inputs) const = 0;

  /// <summary>
  /// Returns the number of node outputs.
  /// </summary>
  /// <returns>The number of node outputs.</returns>
  virtual size_t GetNumOutputs() const = 0;

  /// <summary>
  /// Gets the node's outputs as OrtValueInfo instances.
  /// </summary>
  /// <param name="outputs">Buffer into which to copy the outputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetOutputs(gsl::span<const OrtValueInfo*> outputs) const = 0;

  /// <summary>
  /// Returns the number of node implicit inputs.
  /// Applies to a node that contains a subgraph (e.g., If or Loop). An implicit input is a value consumed by an
  /// internal subgraph node that is not defined in the subgraph.
  /// </summary>
  /// <param name="num_implicit_inputs">Output parameter set to the number of implicit inputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNumImplicitInputs(size_t& num_implicit_inputs) const = 0;

  /// <summary>
  /// Gets the node's implicit inputs.
  /// Applies to a node that contains a subgraph (e.g., If or Loop). An implicit input is a value consumed by an
  /// internal subgraph node that is not defined in the subgraph.
  /// </summary>
  /// <param name="implicit_inputs">Buffer into which to copy the implicit inputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetImplicitInputs(gsl::span<const OrtValueInfo*> implicit_inputs) const = 0;

  /// <summary>
  /// Returns the number of node attributes.
  /// </summary>
  /// <returns>The number of node attributes.</returns>
  virtual size_t GetNumAttributes() const = 0;

  /// <summary>
  /// Gets the node's attributes.
  /// </summary>
  /// <param name="attrs">Buffer into which to copy the attributes.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetAttributes(gsl::span<const OrtOpAttr*> attrs) const = 0;

  /// <summary>
  /// Gets the number of node subgraphs.
  /// </summary>
  /// <param name="num_subgraphs">Output parameter set to the number of subgraphs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNumSubgraphs(size_t& num_subgraphs) const = 0;

  /// <summary>
  /// Gets the node's subgraphs (e.g., subgraphs contained by an If or Loop node).
  /// </summary>
  /// <param name="subgraphs">Buffer into which to copy the subgraphs.</param>
  /// <param name="opt_attribute_names">Optional buffer into which to copy the attribute name for each subgraph.
  /// If set, must point to a buffer with the same number of elements as `subgraphs`.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetSubgraphs(gsl::span<const OrtGraph*> subgraphs,
                                           const char** opt_attribute_names) const = 0;

  /// <summary>
  /// Gets the node's parent graph, which is the graph that contains this node.
  /// </summary>
  /// <param name="parent_graph">Output parameter set to the node's parent graph.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetGraph(const OrtGraph*& parent_graph) const = 0;

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
  /// Returns the model's path, which could be empty if unknown.
  /// </summary>
  /// <returns>The model path.</returns>
  virtual const ORTCHAR_T* GetModelPath() const = 0;

  /// <summary>
  /// Returns the model's ONNX IR version. Important in checking for optional graph inputs
  /// (aka non-constant initializers), which were introduced in ONNX IR version 4.
  /// </summary>
  /// <returns>The model's ONNX IR version.</returns>
  virtual int64_t GetOnnxIRVersion() const = 0;

  /// <summary>
  /// Gets the number of operator sets (domain, opset version) the graph's model relies on.
  /// </summary>
  /// <param name="num_operator_sets">Output parameter set to the number of operator sets.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNumOperatorSets(size_t& num_operator_sets) const = 0;

  /// <summary>
  /// Gets the operator sets the graph's model relies on. An operator set is uniquely identified by a
  /// (domain, opset version) pair.
  /// </summary>
  /// <param name="domains">Buffer into which to copy the domains.</param>
  /// <param name="opset_versions">Buffer into which to copy the opset version for each domain.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetOperatorSets(gsl::span<const char*> domains,
                                              gsl::span<int64_t> opset_versions) const = 0;

  /// <summary>
  /// Returns the number of graph inputs, including initializers that appear in the list of graph inputs.
  /// </summary>
  /// <returns>The number of graph inputs.</returns>
  virtual size_t GetNumInputs() const = 0;

  /// <summary>
  /// Gets the graph's inputs (including initializers) as OrtValueInfo instances.
  /// </summary>
  /// <param name="inputs">Buffer into which to copy the inputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetInputs(gsl::span<const OrtValueInfo*> inputs) const = 0;

  /// <summary>
  /// Returns the number of graph outputs.
  /// </summary>
  /// <returns>The number of graph outputs.</returns>
  virtual size_t GetNumOutputs() const = 0;

  /// <summary>
  /// Gets the graph's outputs as OrtValueInfo instances.
  /// </summary>
  /// <param name="outputs">Buffer into which to copy the outputs.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetOutputs(gsl::span<const OrtValueInfo*> outputs) const = 0;

  /// <summary>
  /// Returns the number of graph initializers (both constant and non-constant).
  /// </summary>
  /// <returns>The number of graph initializers.</returns>
  virtual size_t GetNumInitializers() const = 0;

  /// <summary>
  /// Gets the graph's initializers (both constant and non-constant) as OrtValueInfo instances.
  /// </summary>
  /// <param name="initializers">The buffer into which to copy the initializers.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetInitializers(gsl::span<const OrtValueInfo*> initializers) const = 0;

  /// <summary>
  /// Returns the number of graph nodes.
  /// </summary>
  /// <returns>The number of graph nodes.</returns>
  virtual size_t GetNumNodes() const = 0;

  /// <summary>
  /// Gets the graph's nodes. The nodes are sorted in a default "reverse DFS" topological order.
  /// </summary>
  /// <param name="nodes">Buffer into which to copy the nodes.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetNodes(gsl::span<const OrtNode*> nodes) const = 0;

  /// <summary>
  /// Gets the graph's parent node, if any. The parent_node is nullptr if this is not a nested subgraph.
  /// </summary>
  /// <param name="parent_node">Output parameter set to the parent node.</param>
  /// <returns>A status indicating success or an error.</returns>
  virtual onnxruntime::Status GetParentNode(const OrtNode*& parent_node) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};
