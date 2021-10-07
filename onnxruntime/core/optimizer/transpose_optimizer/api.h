#pragma once

#include <vector>
#include <string_view>
#include <memory>
#include <optional>

namespace onnx_layout_transformation {
namespace api {

/* General notes:
 *   - All pointers are non-null unless specified otherwise
 *   - If two function calls return identical components of a model (i.e. same node) 
 *     they MAY use two different instances of the class (Node) to do so.
 *     This allows for lazy creation of API class instances.
 */

// A constant tensor value used by initializers and attributes
class Tensor {
 public:
  // TODO: how to deal with optional/seq/map tensors
  virtual std::vector<int64_t> Shape() const = 0;
  virtual std::vector<int64_t> DataInt64() const = 0;
  virtual ~Tensor(){};
};

// A named value declared in a graph. Either a graph input, graph initializer, or node output
// Must be able to provide up-to-date information on the value of that name unless that value is removed from the graph.
class ValueInfo {
 public:
  virtual const std::string_view Name() const = 0;
  // nullptr if rank is unknown, -1 for unknown dims
  virtual std::optional<std::vector<int64_t>> Shape() const = 0;

  /**** Editing ****/
  // nullptr if rank is unknown, -1 for unknown dims
  virtual void SetShape(const std::vector<int64_t>* shape) = 0;
  // TODO: what to do if shape is None?
  virtual void PermuteDims(const std::vector<int64_t>& perm) = 0;
  virtual void UnsqueezeDims(const std::vector<int64_t>& axes) = 0;
  virtual ~ValueInfo(){};
};

// A node in a graph. Information should remain up-to-date even if node is modified, unless it is deleted.
// Methods will never be called on deleted instances.
class Node {
 public:
  virtual const std::string_view Name() const = 0;
  virtual const std::string_view OpType() const = 0;
  virtual const std::string_view Domain() const = 0;
  virtual std::vector<std::string_view> Inputs() const = 0;
  virtual std::vector<std::string_view> Outputs() const = 0;
  virtual std::optional<int64_t> GetAttributeInt(const std::string_view name) const = 0;
  virtual std::optional<std::vector<int64_t>> GetAttributeInts(const std::string_view name) const = 0;

  /**** Editing ****/
  // Add int attribute with name and value. Overwrite existing value if present.
  virtual void SetAttributeInt(const std::string_view name, int64_t value) = 0;
  virtual void SetAttributeInts(const std::string_view name, const std::vector<int64_t>& value) = 0;
  // Copy all attributes from an existing node.
  virtual void CopyAttributes(const Node& node) = 0;
  // Remove attribute with name if present.
  virtual void ClearAttribute(const std::string_view name) = 0;
  //virtual void SetInputs(const std::vector<std::string_view>& inputs) = 0;
  virtual void SetInput(size_t i, const std::string_view name) = 0;
  virtual void AddInput(const std::string_view name) = 0;
  virtual ~Node(){};

 public:
  virtual bool IsOp(const std::string_view op_type, const std::string_view domain = "") const {
    return OpType() == op_type && Domain() == domain;
  }
  virtual int64_t GetAttributeIntDefault(const std::string_view name, int64_t default_value) const {
    std::optional<int64_t> value = GetAttributeInt(name);
    if (value == std::nullopt) {
      return default_value;
    }
    return *value;
  }
};

struct ValueConsumers {
  // List of nodes in the current graph with value as input
  std::vector<std::unique_ptr<Node>> nodes;
  // True iff all consumers are nodes in the current graph (not graph outputs/nodes in subgraphs)
  bool comprehensive;
};

class Graph {
 public:
  virtual std::optional<int64_t> Opset(const std::string_view domain = "") const = 0;
  // Return a topologically-sorted list of nodes in the graph
  virtual std::vector<std::unique_ptr<Node>> Nodes() const = 0;
  virtual std::vector<std::string_view> Inputs() const = 0;
  virtual std::vector<std::string_view> Outputs() const = 0;
  // Return nullptr if the value is not a constant (initializer with no corresponding input)
  virtual std::unique_ptr<Tensor> GetConstant(const std::string_view name) const = 0;
  virtual std::unique_ptr<ValueInfo> GetValueInfo(const std::string_view name) const = 0;
  virtual std::unique_ptr<ValueConsumers> GetValueConsumers(const std::string_view name) const = 0;
  // Return nullptr if name is from an input/initializer
  virtual std::unique_ptr<Node> GetNodeByOutput(const std::string_view name) const = 0;

  /**** Editing ****/
  // Transpose an initializer "in place". Input will always be valid name of an initializer. Update shape for ValueInfo.
  virtual void TransposeInitializer(const std::string_view name, const std::vector<int64_t> perm) = 0;
  // Like TransposeInitializer. Product of dims will always match number of elements. Should be fast since
  // data buffer is unchanged.
  virtual void ReshapeInitializer(const std::string_view name, const std::vector<int64_t>& shape) = 0;
  virtual std::unique_ptr<Node> AddNode(const std::string_view op_type, const std::vector<std::string_view>& inputs,
                                        size_t num_outputs = 1, const std::string_view domain = "") = 0;
  // Deletes a node from the graph. Node instances will never be accessed after deletion.
  virtual void RemoveNode(Node& node) = 0;
  virtual void RemoveInitializer(const std::string_view name) = 0;
  // Create an int64 initializer with the specified shape and values. Return the name.
  virtual const std::string_view AddInitializerInt64(const std::vector<int64_t>& shape, const std::vector<int64_t>& values) = 0;
  // "Moves" an output from one node to another, (effectively transfering the output name, shape, type,
  // and all consumers, even those in subgraphs). Source node should be given a new output name. The destination
  // node's output is guaranteed to have no consumers before the call and can be deleted once replaced.
  virtual void MoveOutput(Node& src_node, size_t src_idx, Node& dst_node, size_t dst_idx) = 0;
  // Copy value info from one output to another. Retains data that cannot be encoded in ValueInfo class.
  // Is it always node to node?
  virtual void CopyValueInfo(const std::string_view src_name, const std::string_view dst_name) = 0;

 public:
  virtual bool HasValueConsumers(const std::string_view name) const {
    auto consumers = GetValueConsumers(name);
    bool unused = consumers->comprehensive && consumers->nodes.size() == 0;
    return !unused;
  }
};

}  // namespace api
}  // namespace onnx_layout_transformation

namespace onnx_layout_transformation {

struct LayoutHandlerResult {
  bool should_transpose;
  size_t rank;
  std::optional<std::string_view> new_op_type;
  std::optional<std::string_view> new_domain;
};

typedef LayoutHandlerResult LayoutHandler(api::Graph& graph, api::Node& node);

// Push/remove transposes from the graph. Returns true if the graph was modified.
bool Optimize(api::Graph& graph, bool allow_extended_ops);

bool ChannelLastToChannelFirst(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& handler_map, bool allow_extended_ops);
bool ChannelFirstToChannelLast(api::Graph& graph, std::unordered_map<std::string_view, LayoutHandler*>& handler_map, bool allow_extended_ops);

}  // namespace onnx_layout_transformation
