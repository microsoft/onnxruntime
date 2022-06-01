// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <string>
#include <map>
#include <limits>
#include "dnnl.hpp"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlNode;

class DnnlNodeArg {
 public:
  DnnlNodeArg(DnnlNode* node, size_t index, bool is_output)
      : node_(node), index_(index), is_output_(is_output){};
  DnnlNodeArg() = default;
  DnnlNode* GetNode() { return node_; };
  size_t GetIndex() { return index_; };
  bool IsOutput() { return is_output_; };
  bool Exists() { return node_ != nullptr; };
  bool operator==(const DnnlNodeArg& e) const {
    return node_ == e.node_ && index_ == e.index_ && is_output_ == e.is_output_;
  };

 private:
  DnnlNode* node_ = nullptr;
  size_t index_ = std::numeric_limits<size_t>::max();
  bool is_output_ = false;
};

class DnnlTensor {
 public:
  DnnlTensor(const NodeArg* arg);
  DnnlTensor(std::string name);
  DnnlTensor() = default;
  std::string Name() const;
  dnnl::memory::dims Dim() const;
  dnnl::memory::data_type Type() const;
  dnnl::memory::format_tag Format();
  //check whether the tensor is dynamic, e.g. contains unspecified dimension
  bool IsDynamic();
  //check whether the tensor exsits for optional input output
  bool Exists();
  std::vector<DnnlNodeArg>& GetConsumers() { return consumers_; };
  DnnlNodeArg& GetProducer() { return producer_; };
  void SetProducer(const DnnlNodeArg& arg);
  void ResetProducer();
  void AddConsumer(const DnnlNodeArg& arg);
  void RemoveConsumer(const DnnlNodeArg& arg);

 private:

  const ONNX_NAMESPACE::TensorShapeProto* GetShape() const;

  std::string tensor_name_;
  ONNX_NAMESPACE::DataType arg_type_;
  std::unique_ptr<ONNX_NAMESPACE::TypeProto> arg_type_proto_;
  //a tensor can have no producer (input.initializer) or no consumer (output for subgraph)
  DnnlNodeArg producer_;
  std::vector<DnnlNodeArg> consumers_;
};

class DnnlNode {
 public:
  DnnlNode(const Node* node);
  DnnlNode() = default;
  std::string& Name();
  size_t& Index();
  std::string& OpType();
  DnnlTensor& Input(int index);
  size_t InputCount();
  DnnlTensor& Output(int index);
  size_t OutputCount();
  NodeAttributes& Attributes();
  std::vector<DnnlTensor*>& Inputs();
  std::vector<DnnlTensor*>& Outputs();
  int SinceVersion();
  void AppendPostOp(std::string op);
  const std::vector<std::string>& GetPostOps();

 private:
  int since_version_;
  std::vector<DnnlTensor*> inputs_;
  std::vector<DnnlTensor*> outputs_;
  static DnnlTensor empty_tensor_;
  std::string name_;  // node can have empty/duplicate name, rely on index instead
  std::string op_type_;
  size_t index_ = std::numeric_limits<size_t>::max();
  std::unique_ptr<NodeAttributes> attr_ = NodeAttributes::Create();
  std::vector<std::string> postops_;
};

class DnnlSubgraph {
 public:
  DnnlSubgraph(const GraphViewer& graph_viewer);
  std::vector<DnnlNode*> GetDnnlNodes();
  DnnlNode* GetDnnlNode(size_t node_index);
  DnnlTensor* GetDnnlTensor(const std::string& tensor_name);
  size_t GetMaxNodeIndex();
  std::vector<size_t> GetDnnlNodesInTopologicalOrder();
  std::vector<DnnlTensor*> GetDnnlInputs();
  std::vector<DnnlTensor*> GetDnnlOutputs();
  std::vector<DnnlTensor*> GetDnnlInitializers();
  // build the subgraph IR
  void Build(const GraphViewer& graph_viewer);
  //check whether the subgraph is dynamic
  void TopoSort();
  bool IsDynamic();
  void AddNode(std::unique_ptr<DnnlNode> new_node);
  void RemoveNode(size_t node_index);
  void AddTensor(std::unique_ptr<DnnlTensor> new_tensor);
  void RemoveTensor(const std::string& tensor_name);

 private:
  //graph owns all nodes
  std::vector<std::unique_ptr<DnnlNode>> dnnl_nodes_;
  std::vector<size_t> nodes_in_topological_order_;
  //graph owns all tensors
  std::unordered_map<std::string, std::unique_ptr<DnnlTensor>> dnnl_tensors_;
  std::vector<DnnlTensor*> inputs_;
  std::vector<DnnlTensor*> outputs_; //output should never get deleted from graph transformation
  std::vector<DnnlTensor*> initializers_;
  bool is_dynamic_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
