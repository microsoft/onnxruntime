// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "interface/graph/graph.h"
#include "core/framework/allocator.h"
#include "core/graph/graph.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;
class ModelProto;
}

namespace onnxruntime {
class NodeArg;
class Path;

const onnx::TensorShapeProto* GetNodeArgShape(const NodeArg* node_arg);
std::unique_ptr<interface::TensorRef> CreateApiTensor(const onnx::TensorProto* tensor, const Path& path, AllocatorPtr cpu_allocator);

class ApiValueInfoView : virtual public interface::ValueInfoViewRef {
 private:
  const NodeArg& node_arg_;

 public:
  explicit ApiValueInfoView(const NodeArg& node_arg) : node_arg_(node_arg) {}
  std::string_view Name() const override;
  std::optional<std::vector<int64_t>> Shape() const override;
  DataType DType() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiValueInfoView);
};

class ApiTensor final : public interface::TensorRef {
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
  DataType DType() const override;
  std::vector<uint8_t> Data() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiTensor);
};

class ApiNodeView : virtual public interface::NodeViewRef {
 private:
  const Node& node_;
 public:
  explicit ApiNodeView(const Node& node) : node_(node) {}
  size_t Index() const override { return node_.Index(); }
  std::string_view Name() const override { return node_.Name(); }
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
  std::optional<std::vector<float>> GetAttributeFloats(std::string_view name) const override;
  void ForEachDef(std::function<void(const interface::ValueInfoViewRef&, bool is_input)> func, bool include_missing_optional_defs) const override;
  int SinceVersion() const override;
  std::vector<std::unique_ptr<interface::GraphViewRef>> GetSubgraphs() const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiNodeView);
};

class ApiGraphView : virtual public interface::GraphViewRef {
 private:
  const Graph& graph_;
  const IndexedSubGraph* isg_;
 protected:
  AllocatorPtr cpu_allocator_;
 public:
  explicit ApiGraphView(const Graph& graph, AllocatorPtr cpu_allocator, const IndexedSubGraph* isg = nullptr) : graph_(graph), isg_(isg), cpu_allocator_(std::move(cpu_allocator)) {}

  std::string_view Name() const override;
  std::string_view ModelPath() const override;
  std::optional<int64_t> Opset(std::string_view domain = "") const override;
  std::vector<std::unique_ptr<interface::NodeViewRef>> NodeViews() const override;
  std::unique_ptr<interface::TensorRef> GetConstant(std::string_view name) const override;
  std::unique_ptr<interface::NodeViewRef> GetNode(size_t node_index) const override;
  std::vector<std::string_view> GetInputsIncludingInitializers() const override;
  std::vector<std::string_view> GetInputs() const override;
  std::vector<std::string_view> GetOutputs() const override;
  bool HasInitializerName(std::string_view name) const override;
  bool IsConstantInitializer(std::string_view name, bool check_outer_scope) const override;
  std::vector<size_t> GetNodesInTopologicalOrder() const override;
  std::unique_ptr<interface::ValueInfoViewRef> GetValueInfoView(std::string_view name) const override;
  std::unique_ptr<interface::NodeViewRef> GetNodeViewProducingOutput(std::string_view name) const override;
  std::vector<std::unique_ptr<interface::NodeViewRef>> GetNodeViewsConsumingOutput(std::string_view name) const override;
  bool IsSubGraph() const override { return graph_.IsSubgraph(); }
#ifdef INTREE_EP
  onnx::ModelProto ToModelProto() const override;
#endif
 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiGraphView);
};

}
