// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#ifdef INTREE_EP
#include "onnx/onnx_pb.h"
#endif

namespace onnxruntime {
/// <summary>
/// Enum of DataTypes using standard ONNX values. Casting to/from int32_t is encouraged.
/// </summary>
enum class DataType : int32_t {
  UNDEFINED = 0,
  FLOAT = 1,   // float
  UINT8 = 2,   // uint8_t
  INT8 = 3,    // int8_t
  UINT16 = 4,  // uint16_t
  INT16 = 5,   // int16_t
  INT32 = 6,   // int32_t
  INT64 = 7,   // int64_t
  STRING = 8,  // string
  BOOL = 9,    // bool
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
  FLOAT8E4M3FN = 17,
  FLOAT8E4M3FNUZ = 18,
  FLOAT8E5M2 = 19,
  FLOAT8E5M2FNUZ = 20,
};

namespace interface {
class GraphViewRef;
/// <summary>
/// An interface for a constant tensor value used by initializers
/// </summary>
class TensorRef {
 public:
  /// <returns>The shape of the tensor. Values are nonnegative.</returns>
  virtual std::vector<int64_t> Shape() const = 0;

  virtual size_t NumElements() const = 0;

  /// <returns>The dtype of the tensor.</returns>
  virtual DataType DType() const = 0;

  /// <summary>
  /// Retrieves copy of raw data bytes from the tensor. Used for reading initializers specifying axes/pads/scales.
  /// </summary>
  /// <returns>Flattened tensor data in bytes</returns>
  virtual std::vector<uint8_t> Data() const = 0;

  virtual ~TensorRef(){};
};

class ValueInfoViewRef {
 public:
   /// <returns>The name of the value in the graph</returns>
  virtual std::string_view Name() const = 0;

  /// <returns>
  /// The inferred/declared tensor shape of the value. nullopt if rank is unknown, otherwise a vector with entries
  /// representing the dimensions of the value. Use -1 for unknown dimensions.
  /// </returns>
  virtual std::optional<std::vector<int64_t>> Shape() const = 0;

  /// <returns>The inferred/declared dtype of the value. UNDEFINED (0) if dtype is unknown.</returns>
  virtual DataType DType() const = 0;

  virtual ~ValueInfoViewRef(){};
};

class NodeViewRef {
 public:
  virtual size_t Index() const = 0;

  virtual std::string_view Name() const = 0;

  /// <returns>Op computed by the node</returns>
  virtual std::string_view OpType() const = 0;

  /// <returns>Domain containing the op. Empty string if node has no domain set.</returns>
  virtual std::string_view Domain() const = 0;

  /// <returns>Names of input values. Empty string may be included for optional inputs.</returns>
  virtual std::vector<std::string_view> Inputs() const = 0;

  /// <returns>Names of output values. Empty string may be included for optional outputs.</returns>
  virtual std::vector<std::string_view> Outputs() const = 0;

  /// <param name="name">Name of the attribute to return</param>
  /// <returns>
  /// The attribute value, or nullopt if the attribute is not present on the node, or is not of type int.
  /// </returns>
  virtual std::optional<int64_t> GetAttributeInt(std::string_view name) const = 0;

  /// <param name="name">Name of the attribute to return</param>
  /// <returns>
  /// The attribute value, or nullopt if the attribute is not present on the node, or is not of type string.
  /// </returns>
  virtual std::optional<std::string> GetAttributeString(std::string_view name) const = 0;

  /// <param name="name">Name of the attribute to return</param>
  /// <returns>
  /// The attribute value, or nullopt if the attribute is not present on the node, or is not of type int[].
  /// </returns>
  virtual std::optional<std::vector<int64_t>> GetAttributeInts(std::string_view name) const = 0;

  virtual std::optional<std::vector<float>> GetAttributeFloats(std::string_view name) const = 0;

  /// <summary>
  /// Convenience method. Returns whether node is of the specified op type and domain
  /// </summary>
  /// <param name="op_type">Op type</param>
  /// <param name="domain">Domain. Empty string and "onnx.ai" are treated as equal.</param>
  /// <returns></returns>
  virtual bool IsOp(std::string_view op_type, std::string_view domain = "") const {
    if (OpType() != op_type) {
      return false;
    }
    std::string_view node_domain = Domain();
    return node_domain == domain ||
           ((domain == "" || domain == "ai.onnx") && (node_domain == "" || node_domain == "ai.onnx"));
  }

  /// <summary>
  /// Convenience method. Returns value of int attribute with name, or given default if unset.
  /// </summary>
  /// <param name="name">Attribute name</param>
  /// <param name="default_value">Default value</param>
  /// <returns>Attribute value or default value</returns>
  virtual int64_t GetAttributeIntDefault(std::string_view name, int64_t default_value) const {
    return GetAttributeInt(name).value_or(default_value);
  }

  virtual void ForEachDef(std::function<void(const ValueInfoViewRef&, bool is_input)> func, bool include_missing_optional_defs) const = 0;

  /// <summary>
  /// Returns the schema since version for the op_type of this node. Value of -1 means it is not set.
  /// </summary>
  /// <returns>since version or default value -1</returns>
  virtual int SinceVersion() const = 0;

  // TODO: Shall we add this API to support subgraph access? looks it contradicts with the comment of GraphRef: No ability to access subgraphs is provided
  virtual std::vector<std::unique_ptr<GraphViewRef>> GetSubgraphs() const = 0;

  virtual ~NodeViewRef(){};
};

class GraphViewRef {
 public:
  virtual std::string_view Name() const = 0;

  virtual std::string_view ModelPath() const = 0;

  /// <param name="domain">Domain name to find in model opset_import</param>
  /// <returns>Opset of domain declared in model, or nullopt if domain is not present</returns>
  virtual std::optional<int64_t> Opset(std::string_view domain) const = 0;

  /// <returns>Topologically-sorted list of nodes in the graph</returns>
  virtual std::vector<std::unique_ptr<NodeViewRef>> NodeViews() const = 0;

  /// <summary>
  /// Checks whether the value name refers to a constant initializer and if so, returns a Tensor corresponding to it.
  /// Constants from parent graphs may be included.
  /// </summary>
  /// <param name="name">Value name. Must be nonempty.</param>
  /// <returns>Tensor corresponding to the constant initializer or nullptr</returns>
  virtual std::unique_ptr<TensorRef> GetConstant(std::string_view name) const = 0;

  virtual std::unique_ptr<NodeViewRef> GetNode(size_t node_index) const = 0;

  virtual std::vector<std::string_view> GetInputsIncludingInitializers() const = 0;

  virtual std::vector<std::string_view> GetInputs() const = 0;

  virtual std::vector<std::string_view> GetOutputs() const = 0;

  virtual bool HasInitializerName(std::string_view name) const = 0;

  virtual bool IsConstantInitializer(std::string_view name, bool check_outer_scope) const = 0;

  virtual std::vector<size_t> GetNodesInTopologicalOrder() const = 0;

  /// <summary>
  /// Returns a ValueInfo instance for querying info about the value with the given name. Behavior is undefined if
  /// the name does not refer to a value in the graph.
  /// <param name="name">Value name. Must be nonempty.</param>
  /// <returns>A ValueInfo instance corresponding to the value with the given name</returns>
  virtual std::unique_ptr<ValueInfoViewRef> GetValueInfoView(std::string_view name) const = 0;

  virtual std::unique_ptr<NodeViewRef> GetNodeViewProducingOutput(std::string_view name) const = 0;

  virtual std::vector<std::unique_ptr<NodeViewRef>> GetNodeViewsConsumingOutput(std::string_view name) const = 0;

  virtual bool IsSubGraph() const = 0;

#ifdef INTREE_EP
  virtual onnx::ModelProto ToModelProto() const = 0;
#endif
  virtual ~GraphViewRef(){};
};
}

}
