// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/initializer.h"
#include "core/optimizer/propagate_cast_ops.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
Status PropagateCastOps::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  (void) graph;
  (void) modified;
  (void) graph_level;
  (void) logger;
  return Status::OK();
}

} // namespace onnxruntime