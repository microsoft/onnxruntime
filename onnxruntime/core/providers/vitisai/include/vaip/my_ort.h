// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "./_sanity_check.h"
#include "./export.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
struct Model;
struct Graph;
struct GraphViewer;
struct Node;
struct NodeArg;
struct ProviderHost;
struct NodeAttributes;
}  // namespace onnxruntime
namespace ONNX_NAMESPACE {
struct AttributeProto;
struct TensorProto;
#ifndef USE_VITISAI
enum TensorProto_DataType : int {
  TensorProto_DataType_UNDEFINED = 0,
  TensorProto_DataType_FLOAT = 1,
  TensorProto_DataType_UINT8 = 2,
  TensorProto_DataType_INT8 = 3,
  TensorProto_DataType_UINT16 = 4,
  TensorProto_DataType_INT16 = 5,
  TensorProto_DataType_INT32 = 6,
  TensorProto_DataType_INT64 = 7,
  TensorProto_DataType_STRING = 8,
  TensorProto_DataType_BOOL = 9,
  TensorProto_DataType_FLOAT16 = 10,
  TensorProto_DataType_DOUBLE = 11,
  TensorProto_DataType_UINT32 = 12,
  TensorProto_DataType_UINT64 = 13,
  TensorProto_DataType_COMPLEX64 = 14,
  TensorProto_DataType_COMPLEX128 = 15,
  TensorProto_DataType_BFLOAT16 = 16
};
enum AttributeProto_AttributeType : int {
  AttributeProto_AttributeType_UNDEFINED = 0,
  AttributeProto_AttributeType_FLOAT = 1,
  AttributeProto_AttributeType_INT = 2,
  AttributeProto_AttributeType_STRING = 3,
  AttributeProto_AttributeType_TENSOR = 4,
  AttributeProto_AttributeType_GRAPH = 5,
  AttributeProto_AttributeType_SPARSE_TENSOR = 11,
  AttributeProto_AttributeType_TYPE_PROTO = 13,
  AttributeProto_AttributeType_FLOATS = 6,
  AttributeProto_AttributeType_INTS = 7,
  AttributeProto_AttributeType_STRINGS = 8,
  AttributeProto_AttributeType_TENSORS = 9,
  AttributeProto_AttributeType_GRAPHS = 10,
  AttributeProto_AttributeType_SPARSE_TENSORS = 12,
  AttributeProto_AttributeType_TYPE_PROTOS = 14
};
#endif

}  // namespace ONNX_NAMESPACE

namespace vaip_core {
class GraphHolder;
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::TensorProto;
using onnxruntime::Graph;
using onnxruntime::GraphViewer;
using onnxruntime::Model;
using onnxruntime::Node;
using onnxruntime::NodeArg;
using onnxruntime::NodeAttributes;
struct ModelDeleter {
  VAIP_DLL_SPEC void operator()(Model* tp) const;
};
using ModelPtr = std::unique_ptr<Model, ModelDeleter>;
struct AttributeProtoDeleter {
  VAIP_DLL_SPEC void operator()(AttributeProto* p) const;
};
using AttributeProtoPtr = std::unique_ptr<AttributeProto, AttributeProtoDeleter>;

struct TensorProtoDeleter {
  VAIP_DLL_SPEC void operator()(TensorProto* tp) const;
};
using TensorProtoPtr = std::unique_ptr<TensorProto, TensorProtoDeleter>;

struct NodeAttributesDeleter {
  VAIP_DLL_SPEC void operator()(NodeAttributes* p) const;
};
using NodeAttributesPtr = std::unique_ptr<NodeAttributes, NodeAttributesDeleter>;
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

using ModelMetaData = std::unordered_map<std::string, std::string>;
}  // namespace vaip_core
