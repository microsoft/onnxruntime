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

class ApiGraphView : virtual public interface::GraphViewRef {
 private:
  const Graph& graph_;
 protected:
  AllocatorPtr cpu_allocator_;
 public:
  explicit ApiGraphView(const Graph& graph, AllocatorPtr cpu_allocator) : graph_(graph), cpu_allocator_(std::move(cpu_allocator)) {}

  std::optional<int64_t> Opset(std::string_view domain = "") const override;
  std::vector<std::unique_ptr<interface::NodeViewRef>> NodeViews() const override;
  std::unique_ptr<interface::TensorRef> GetConstant(std::string_view name) const override;
  std::unique_ptr<interface::ValueInfoViewRef> GetValueInfoView(std::string_view name) const override;
#ifdef INTREE_EP
  onnx::ModelProto ToModelProto() override;
#endif
 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ApiGraphView);
};

}
