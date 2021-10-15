// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <string>
#include <map>
#include "dnnl.hpp"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlTensor {
 public:
  DnnlTensor(const NodeArg* arg);
  DnnlTensor(std::string name);
  std::string Name() const;
  dnnl::memory::dims Dim();
  dnnl::memory::data_type Type();
  dnnl::memory::format_tag Format();
  //check whether the tensor is dynamic, e.g. contains unspecified dimension
  bool IsDynamic();
  //check whether the tensor exsits for optional input output
  bool Exists();

 private:
  std::string tensor_name_;
  const NodeArg* arg_;
  bool is_dynamic_;
};

class DnnlNode {
 public:
  DnnlNode(const Node* node);
  std::string Name();
  std::string OpType();
  DnnlTensor Input(int index);
  size_t InputCount();
  DnnlTensor Output(int index);
  size_t OutputCount();
  const NodeAttributes& Attributes();

 private:
  const Node* onnx_node_;
  std::vector<DnnlTensor> inputs_;
  std::vector<DnnlTensor> outputs_;
};

class DnnlSubgraph {
 public:
  DnnlSubgraph(const GraphViewer& graph_viewer);
  std::vector<DnnlNode> GetDnnlNodes();
  std::vector<DnnlTensor> GetDnnlInputs();
  std::vector<DnnlTensor> GetDnnlOutputs();
  std::vector<DnnlTensor> GetDnnlInitializers();
  // build the subgraph IR
  void Build();
  //check whether the subgraph is dynamic
  bool IsDynamic();

 private:
  std::vector<DnnlNode> dnnl_nodes_;
  std::vector<DnnlTensor> inputs_;
  std::vector<DnnlTensor> outputs_;
  std::vector<DnnlTensor> initializers_;
  const GraphViewer& graph_viewer_;
  bool is_dynamic_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
