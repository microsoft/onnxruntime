// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/transpose_optimizer/api.h"
#include "core/graph/graph.h"
#include "core/framework/execution_provider.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_layout_transformation;

namespace onnxruntime {

class OrtValueInfo : public api::ValueInfo {
 private:
  onnxruntime::Graph& graph_;
  std::string name_;

 public:
  OrtValueInfo(onnxruntime::Graph& graph, std::string name) : graph_(graph), name_(name){};
  virtual const std::string_view Name() const override;
  virtual std::optional<std::vector<int64_t>> Shape() const override;

  virtual void SetShape(const std::vector<int64_t>* shape) override;
  virtual void PermuteDims(const std::vector<int64_t>& perm) override;
  virtual void UnsqueezeDims(const std::vector<int64_t>& axes) override;
};

class OrtTensor : public api::Tensor {
 private:
  const onnx::TensorProto& tensor_proto_;
  const Graph& graph_;
  AllocatorPtr cpu_allocator_;
 public:
  OrtTensor(const onnx::TensorProto& tensor_proto, const Graph& graph, AllocatorPtr cpu_allocator) : tensor_proto_(tensor_proto), graph_(graph), cpu_allocator_(cpu_allocator){};
  const onnx::TensorProto& TensorProto() {
    return tensor_proto_;
  }
  virtual std::vector<int64_t> Shape() const;
  virtual std::vector<int64_t> DataInt64() const;
};

class OrtGraph;

class OrtNode : public api::Node {
 private:
  onnxruntime::Node& node_;
  Graph& graph_;
 public:
  OrtNode(onnxruntime::Node& node, Graph& graph) : node_(node), graph_(graph){};
  onnxruntime::Node& Node() {
    return node_;
  }
  virtual const std::string_view Name() const;
  virtual const std::string_view OpType() const;
  virtual const std::string_view Domain() const;
  virtual std::vector<std::string_view> Inputs() const;
  virtual std::vector<std::string_view> Outputs() const;
  virtual std::optional<int64_t> GetAttributeInt(const std::string_view name) const;
  virtual std::optional<std::vector<int64_t>> GetAttributeInts(const std::string_view name) const;
  virtual void SetAttributeInt(const std::string_view name, int64_t value);
  virtual void SetAttributeInts(const std::string_view name, const std::vector<int64_t>& value);
  virtual void CopyAttributes(const api::Node& node);
  virtual void ClearAttribute(const std::string_view name);
  virtual void SetInput(size_t i, const std::string_view name);
  virtual void AddInput(const std::string_view name);
};

class OrtGraph : public api::Graph {
 private:
  onnxruntime::Graph& graph_;
  std::vector<std::string_view> inputs_;
  std::vector<std::string_view> outputs_;
  AllocatorPtr cpu_allocator_;
  const logging::Logger& logger_;
 public:
  OrtGraph(onnxruntime::Graph& graph, AllocatorPtr cpu_allocator, const logging::Logger& logger);
  onnxruntime::Graph& Graph() {
    return graph_;
  }
  virtual std::optional<int64_t> Opset(const std::string_view domain = "") const;
  virtual std::vector<std::unique_ptr<api::Node>> Nodes() const;
  virtual std::vector<std::string_view> Inputs() const;
  virtual std::vector<std::string_view> Outputs() const;
  virtual std::unique_ptr<api::Tensor> GetConstant(const std::string_view name) const;
  virtual std::unique_ptr<api::ValueInfo> GetValueInfo(const std::string_view name) const;
  virtual std::unique_ptr<api::ValueConsumers> GetValueConsumers(const std::string_view name) const;
  virtual std::unique_ptr<api::Node> GetNodeByOutput(const std::string_view name) const;
  virtual void TransposeInitializer(const std::string_view name, const std::vector<int64_t> perm);
  virtual void ReshapeInitializer(const std::string_view name, const std::vector<int64_t>& shape);
  virtual std::unique_ptr<api::Node> AddNode(const std::string_view op_type, const std::vector<std::string_view>& inputs,
                                             size_t num_outputs = 1, const std::string_view domain = "");
  virtual void RemoveNode(api::Node& node);
  virtual void RemoveInitializer(const std::string_view name);
  virtual const std::string_view AddInitializerInt64(const std::vector<int64_t>& shape, const std::vector<int64_t>& values);
  virtual void MoveOutput(api::Node& src_node, size_t src_idx, api::Node& dst_node, size_t dst_idx);
  virtual void CopyValueInfo(const std::string_view src_name, const std::string_view dst_name);
};

}  // namespace onnxruntime