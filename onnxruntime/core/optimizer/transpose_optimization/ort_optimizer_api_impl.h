// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The ONNX Runtime specific implementation of the generic transpose optimizer API.
#pragma once


#include <deque>
#include <iterator>
#include <optional>

#include "core/optimizer/transpose_optimization/optimizer_api.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/graph/graph_view_ref.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnx_transpose_optimization;

namespace onnxruntime {
class GraphViewer;

class ApiValueInfoView final : public ValueInfoViewRef {
 private:
  const NodeArg& node_arg_;

 public:
  explicit ApiValueInfoView(const NodeArg& node_arg) : node_arg_(node_arg) {}
  std::string_view Name() const override;
  std::optional<std::vector<int64_t>> Shape() const override;
  onnxruntime::DataType DType() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiValueInfoView);
};

class ApiValueInfo final : public api::ValueInfoRef {
 private:
  NodeArg& node_arg_;

 public:
  explicit ApiValueInfo(NodeArg& node_arg) : node_arg_(node_arg) {}
  std::string_view Name() const override;
  std::optional<std::vector<int64_t>> Shape() const override;
  onnxruntime::DataType DType() const override;

  void SetShape(const std::vector<int64_t>* shape) override;
  void PermuteDims(const std::vector<int64_t>& perm) override;
  void UnsqueezeDims(const std::vector<int64_t>& axes) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiValueInfo);
};

class ApiTensor final : public TensorRef {
 private:
  const onnx::TensorProto& tensor_proto_;
  const Path& model_path_;
  AllocatorPtr cpu_allocator_;

 public:
  explicit ApiTensor(const onnx::TensorProto& tensor_proto, const Path& model_path, AllocatorPtr cpu_allocator)
      : tensor_proto_(tensor_proto), model_path_(model_path), cpu_allocator_(std::move(cpu_allocator)) {}

  const onnx::TensorProto& TensorProto() {
    return tensor_proto_;
  }

  std::vector<int64_t> Shape() const override;
  size_t NumElements() const override;
  onnxruntime::DataType DType() const override;
  std::vector<uint8_t> Data() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiTensor);
};

class ApiNodeView final : public NodeViewRef {
 private:
  const onnxruntime::Node& node_;
 public:
  explicit ApiNodeView(const onnxruntime::Node& node) : node_(node) {}
  std::string_view OpType() const override {
    return node_.OpType();
  }
  std::string_view Domain() const override {
    return node_.Domain();
  }
  std::vector<std::string_view> Inputs() const override;
  std::vector<std::string_view> Outputs() const override;
  std::optional<int64_t> GetAttributeInt(std::string_view name) const override;
  std::optional<std::string> GetAttributeString(std::string_view name) const override;
  std::optional<std::vector<int64_t>> GetAttributeInts(std::string_view name) const override;
  int SinceVersion() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiNodeView);
};

class ApiNode final : public api::NodeRef {
 private:
  onnxruntime::Node& node_;
  Graph& graph_;

 public:
  explicit ApiNode(onnxruntime::Node& node, Graph& graph) : node_(node), graph_(graph) {}

  onnxruntime::Node& Node() {
    return node_;
  }

  std::string_view OpType() const override {
    return node_.OpType();
  }
  std::string_view Domain() const override {
    return node_.Domain();
  }
  std::vector<std::string_view> Inputs() const override;
  std::vector<std::string_view> Outputs() const override;
  std::optional<int64_t> GetAttributeInt(std::string_view name) const override;
  std::optional<std::string> GetAttributeString(std::string_view name) const override;
  std::optional<std::vector<int64_t>> GetAttributeInts(std::string_view name) const override;
  void SetAttributeInt(std::string_view name, int64_t value) override;
  void SetAttributeInts(std::string_view name, const std::vector<int64_t>& value) override;
  void CopyAttributes(const api::NodeRef& node) override;
  void ClearAttribute(std::string_view name) override;
  void SetInput(size_t i, std::string_view name) override;
  std::string_view GetExecutionProviderType() const override;
  virtual int SinceVersion() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiNode);
};

class ApiGraphView final : public GraphViewRef {
 private:
  const onnxruntime::GraphViewer& graph_;
  AllocatorPtr cpu_allocator_;

 public:
  explicit ApiGraphView(const onnxruntime::GraphViewer& graph, AllocatorPtr cpu_allocator)
      : graph_(graph), cpu_allocator_(std::move(cpu_allocator)) {}

  std::optional<int64_t> Opset(std::string_view domain = "") const override;
  std::vector<std::unique_ptr<NodeViewRef>> Nodes() const override;
  std::unique_ptr<onnxruntime::TensorRef> GetConstant(std::string_view name) const override;
  std::unique_ptr<onnxruntime::ValueInfoViewRef> GetValueInfo(std::string_view name) const override;
#ifdef INTREE_EP
  onnx::ModelProto ToModelProto() override;
#endif
 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiGraphView);
};

class ApiGraph final : public api::GraphRef {
 private:
  onnxruntime::Graph& graph_;
  AllocatorPtr cpu_allocator_;
  const char* new_node_ep_;

 public:
  explicit ApiGraph(onnxruntime::Graph& graph, AllocatorPtr cpu_allocator, const char* new_node_ep)
      : graph_(graph), cpu_allocator_(std::move(cpu_allocator)), new_node_ep_(new_node_ep) {}

  onnxruntime::Graph& Graph() {
    return graph_;
  }

  std::optional<int64_t> Opset(std::string_view domain = "") const override;
  std::vector<std::unique_ptr<api::NodeRef>> Nodes() const override;
  std::unique_ptr<onnxruntime::TensorRef> GetConstant(std::string_view name) const override;
  std::unique_ptr<onnxruntime::TensorRef> GetLocalConstant(std::string_view name) const override;
  std::unique_ptr<api::ValueInfoRef> GetValueInfo(std::string_view name) const override;
  std::unique_ptr<api::ValueConsumers> GetValueConsumers(std::string_view name) const override;
  std::unique_ptr<api::NodeRef> GetNodeProducingOutput(std::string_view name) const override;
  void TransposeInitializer(std::string_view name, const std::vector<int64_t>& perm) override;
  void ReshapeInitializer(std::string_view name, const std::vector<int64_t>& shape) override;
  std::unique_ptr<api::NodeRef> AddNode(std::string_view op_type, const std::vector<std::string_view>& inputs,
                                        size_t num_outputs = 1, std::string_view domain = "") override;

  std::unique_ptr<api::NodeRef> CopyNode(const api::NodeRef& source_node, std::string_view op_type,
                                         std::string_view domain = "",
                                         std::optional<int> since_version = std::nullopt) override;
  void RemoveNode(api::NodeRef& node) override;
  void RemoveInitializer(std::string_view name) override;
  std::string_view AddInitializer(onnxruntime::DataType dtype, const std::vector<int64_t>& shape,
                                  const std::vector<uint8_t>& data) override;
  void MoveOutput(api::NodeRef& src_node, size_t src_idx, api::NodeRef& dst_node, size_t dst_idx) override;
  void CopyValueInfo(std::string_view src_name, std::string_view dst_name) override;
  bool HasValueConsumers(std::string_view name) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiGraph);
};
}
