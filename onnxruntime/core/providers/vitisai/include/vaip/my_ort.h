// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "./export.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_VITISAI
namespace onnxruntime {
class Model;
class Graph;
class Node;
class NodeArg;
class Path;
struct IndexedSubGraph;
struct ProviderHost;
using NodeIndex = size_t;
using ModelMetaData = std::unordered_map<std::string, std::string>;
}  // namespace onnxruntime
namespace ONNX_NAMESPACE {
class AttributeProto;
class TensorProto;
class GraphProto;
}  // namespace ONNX_NAMESPACE
#else
namespace onnxruntime {
struct Model;
struct Graph;
struct Node;
struct NodeArg;
struct Path;
struct IndexedSubGraph;
struct ProviderHost;
using NodeIndex = size_t;
using ModelMetaData = std::unordered_map<std::string, std::string>;
}  // namespace onnxruntime
namespace ONNX_NAMESPACE {
struct AttributeProto;
struct TensorProto;
struct GraphProto;
}  // namespace ONNX_NAMESPACE
#endif

namespace vaip_core {
using namespace ONNX_NAMESPACE;
using namespace onnxruntime;
struct ModelDeleter {
  VAIP_DLL_SPEC void operator()(Model* tp) const;
};
using ModelPtr = std::unique_ptr<Model, ModelDeleter>;
struct AttributeProtoDeleter {
  VAIP_DLL_SPEC void operator()(AttributeProto* p) const;
};
using AttributeProtoPtr =
    std::unique_ptr<AttributeProto, AttributeProtoDeleter>;

struct TensorProtoDeleter {
  VAIP_DLL_SPEC void operator()(TensorProto* tp) const;
};
using TensorProtoPtr = std::unique_ptr<TensorProto, TensorProtoDeleter>;

/// I cannot forward declare a using directive, because
/// std::unorderd_map required AttributeProto must be defiend.
class NodeAttributes;
struct NodeAttributesDeleter {
  VAIP_DLL_SPEC void operator()(NodeAttributes* p) const;
};
using NodeAttributesPtr =
    std::unique_ptr<NodeAttributes, NodeAttributesDeleter>;
/// get node's input
/// when Node* is nullptr, it is a tensor in the initializer.
/// node_arg is always non-null.
struct NodeInput {
  /// 1. node == nullptr, node_arg == nullptr ， pattern is not
  /// matched.
  ///
  /// 2. node != nullptr, node_arg != nullptr , input is another
  /// node's output
  ///
  /// 3. node == nullptr, node_arg != nullptr , input is a graph input
  /// or a constant initializer. node_arg_is_constant() is used to
  /// test if it is a constant intializer.
  ///
  /// 4. node != nullptr, node_arg == nullptr, never happen. invalid state.

  const Node* node;
  const NodeArg* node_arg;
  bool is_matched() const { return node_arg != nullptr; }
};

using InitializedTensorSet =
    std::unordered_map<std::string, const TensorProto*>;
}  // namespace vaip_core
