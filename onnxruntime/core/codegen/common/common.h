// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/inlined_containers.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_nodes.h"
#include "core/graph/graph_viewer.h"

#ifndef NDEBUG
#define ORT_ENFORCE_DEBUG(...) ORT_ENFORCE(__VA_ARGS__)
#else
#define ORT_ENFORCE_DEBUG(...)
#endif  // !NDEBUG

// DYN_PROMOTE is a simplified llvm::dyn_cast, which does not need RTTI
// DYN_PROMOTE is faster than dynamic_cast and also has smaller binary size
// Please use DYN_PROMOTE in a critical path.
#define DYN_PROMOTE(BASE)                                     \
  template <typename ToType>                                  \
  inline const ToType* Promote(const BASE* base) {            \
    if (ToType::IsType(base))                                 \
      return static_cast<const ToType*>(base);                \
    return nullptr;                                           \
  }                                                           \
                                                              \
  template <typename ToType>                                  \
  inline ToType* Promote(BASE* base) {                        \
    if (ToType::IsType(base))                                 \
      return static_cast<ToType*>(base);                      \
    return nullptr;                                           \
  }                                                           \
                                                              \
  template <typename ToType>                                  \
  inline ToType* Promote(const std::unique_ptr<BASE>& base) { \
    if (ToType::IsType(base.get()))                           \
      return static_cast<ToType*>(base);                      \
    return nullptr;                                           \
  }                                                           \
                                                              \
  template <typename ToType>                                  \
  inline ToType* Promote(const std::shared_ptr<BASE>& base) { \
    if (ToType::IsType(base.get()))                           \
      return static_cast<ToType*>(base);                      \
    return nullptr;                                           \
  }

// DYN_PROMOTE_BASE is a macro inserted in the base class to support DYN_PROMOTE
// TYPE_ID is required for DYN_PROMOTE and TYPE_ID is a enum class
// TYPE_ID_VAR is a corresponding variable name for in the base class
#define DYN_PROMOTE_BASE(BASE, TYPE_ID, TYPE_ID_VAR) \
  inline const TYPE_ID TypeID() const {              \
    return TYPE_ID_VAR;                              \
  }                                                  \
                                                     \
  static inline bool IsType(const BASE*) {           \
    return true;                                     \
  }

// DYN_PROMOTE_DERIVED is a macro inserted in a derived class to support DYN_PROMOTE
// TYPE_ID is required for DYN_PROMOTE and TYPE_ID is a enum class
// TYPE_ID_VALUE is corresponding TYPE_ID::value of a derived class.
#define DYN_PROMOTE_DERIVED(BASE, TYPE_ID, TYPE_ID_VALUE) \
  static inline bool IsType(const BASE* base) {           \
    ORT_ENFORCE_DEBUG(nullptr != base);                   \
    return base->TypeID() == TYPE_ID::TYPE_ID_VALUE;      \
  }

// DYNAMIC_PROMOTE is a dynamic_cast needing RTTI
// DYNAMIC_PROMOTE is usually slower than than DYN_PROMOTE.
// Please use DYNAMIC_PROMOTE in a non-critical path.
#define DYNAMIC_PROMOTE(BASE)                            \
  template <typename X>                                  \
  inline const X* Promote(const BASE* base) {            \
    auto derived = dynamic_cast<const X*>(base);         \
    ORT_ENFORCE(nullptr != derived);                     \
    return derived;                                      \
  }                                                      \
                                                         \
  template <typename X>                                  \
  inline X* Promote(BASE* base) {                        \
    auto derived = dynamic_cast<X*>(base);               \
    ORT_ENFORCE(nullptr != derived);                     \
    return derived;                                      \
  }                                                      \
                                                         \
  template <typename X>                                  \
  inline X* Promote(const std::unique_ptr<BASE>& base) { \
    auto derived = dynamic_cast<X*>(base.get());         \
    ORT_ENFORCE(nullptr != derived);                     \
    return derived;                                      \
  }                                                      \
                                                         \
  template <typename X>                                  \
  inline X* Promote(const std::shared_ptr<BASE>& base) { \
    auto derived = dynamic_cast<X*>(base.get());         \
    ORT_ENFORCE(nullptr != derived);                     \
    return derived;                                      \
  }

namespace onnxruntime {

// Nodekey is used as a key for maps
using NodeKey = std::string;

NodeKey GetKey(const onnxruntime::Node* node);
NodeKey GetKey(const onnxruntime::Node& node);
NodeKey GetKey(const onnxruntime::NodeArg* def);

bool IsRecurrentNode(const onnxruntime::Node& node);

bool IsAliasNode(const onnxruntime::Node& node);

// Helper function that creates ComputeCapability for subgraphs
std::unique_ptr<ComputeCapability> ToCapacity(const onnxruntime::GraphViewer& graph,
                                              int fused_count,
                                              std::unique_ptr<IndexedSubGraph>& subgraph);

bool IsFusedNode(const Node& node);

bool HasLoop(const Node& node);

const Graph* GetSubgraph(const Node& node);

std::string NormalizeCppName(const std::string& name);

std::string NormalizeNodeArgName(const NodeArg* def);

// Return the corresponding input node for the NodeArg of the given node
const onnxruntime::Node* GetInputNode(const Node& node, const NodeArg* def);

int64_t ShapeRank(const NodeArg* def);

bool ShapeHasValue(const NodeArg* def, int i);

bool ShapeHasSymbol(const NodeArg* def, int i);

int64_t ShapeValue(const NodeArg* def, int i);

const std::string& ShapeSymbol(const NodeArg* def, int i);

ONNX_NAMESPACE::TensorProto_DataType TensorProtoDataType(const NodeArg* def);

// Convert ConstGraphNodes to internal NodePtrs without check lifetime.
// Please use it only locally when GraphNodes still exist
InlinedVector<const Node*> ConvertGraphNodesToNodePtrs(const ConstGraphNodes& graph_nodes);

enum : int {
  Dimension_Unknown = -1,
};

}  // namespace onnxruntime
