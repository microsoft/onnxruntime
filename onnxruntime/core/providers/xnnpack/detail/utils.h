// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/op_kernel.h"
#include "xnnpack.h"

namespace onnxruntime {
struct IndexedSubGraph::MetaDef;
class GraphViewer;

namespace xnnpack {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

struct XnnpackOperatorDeleter {
  void operator()(struct xnn_operator* p) const {
    if (p != nullptr) {
      // Ignore returned value because it fails only when xnnpack wasn't initialized
      xnn_delete_operator(p);
    }
  }
};

using XnnpackOperator = std::unique_ptr<struct xnn_operator, XnnpackOperatorDeleter>;

std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const Node& conv, const Node& activation,
                                                         const GraphViewer& graph);
}  // namespace xnnpack
}  // namespace onnxruntime
