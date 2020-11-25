// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {
class Graph;
class Node;
}  // namespace onnxruntime

namespace onnxruntime {

// Function representation class.
class Function {
 public:
  virtual ~Function() {}
  virtual const ONNX_NAMESPACE::OpSchema& OpSchema() const = 0;

  virtual const onnxruntime::Graph& Body() const = 0;

  virtual const IndexedSubGraph& GetIndexedSubGraph() const = 0;
};

std::unique_ptr<Function> MakeFunction(const onnxruntime::Graph& graph,
                                       std::unique_ptr<IndexedSubGraph> customized_func);
}  // namespace onnxruntime
