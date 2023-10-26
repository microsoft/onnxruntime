// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The ONNX Runtime specific implementation of the generic transpose optimizer API.
#pragma once

#ifdef _WIN32
#pragma warning(disable : 4250)
#endif

#include <deque>
#include <iterator>
#include <optional>
#include "core/optimizer/transpose_optimization/optimizer_api.h"
#include "core/graph/graph_view_api_impl.h"
#include "core/graph/graph_viewer.h"

using namespace onnx_transpose_optimization;

namespace onnxruntime {
class ApiValueInfo final : virtual public api::ValueInfoRef, public ApiValueInfoView {
 private:
  NodeArg& node_arg_;
 public:
  explicit ApiValueInfo(NodeArg& node_arg) : ApiValueInfoView(node_arg), node_arg_(node_arg) {}

  void SetShape(const std::vector<int64_t>* shape) override;
  void PermuteDims(const std::vector<int64_t>& perm) override;
  void UnsqueezeDims(const std::vector<int64_t>& axes) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiValueInfo);
};

class ApiNode final : virtual public api::NodeRef, public ApiNodeView {
 private:
  onnxruntime::Node& node_;
  Graph& graph_;

 public:
  explicit ApiNode(onnxruntime::Node& node, Graph& graph) : ApiNodeView(node), node_(node), graph_(graph) {}
  onnxruntime::Node& Node() {
    return node_;
  }
  void SetAttributeInt(std::string_view name, int64_t value) override;
  void SetAttributeInts(std::string_view name, const std::vector<int64_t>& value) override;
  void CopyAttributes(const api::NodeRef& node) override;
  void ClearAttribute(std::string_view name) override;
  void SetInput(size_t i, std::string_view name) override;
  std::string_view GetExecutionProviderType() const override;
  int64_t Id() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiNode);
};

class ApiGraph final : virtual public api::GraphRef, public ApiGraphView {
 private:
  onnxruntime::Graph& graph_;
  const char* new_node_ep_;

 public:
  explicit ApiGraph(onnxruntime::Graph& graph, AllocatorPtr cpu_allocator, const char* new_node_ep) : ApiGraphView(graph, std::move(cpu_allocator)), graph_(graph), new_node_ep_(new_node_ep) {}
  onnxruntime::Graph& Graph() {
    return graph_;
  }

  std::vector<std::unique_ptr<api::NodeRef>> Nodes() const override;
  std::unique_ptr<interface::TensorRef> GetLocalConstant(std::string_view name) const override;
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
