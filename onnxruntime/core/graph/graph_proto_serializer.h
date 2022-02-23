// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_viewer.h"

namespace onnxruntime {

class GraphProtoSerializer {
 public:
  GraphProtoSerializer(const GraphViewer* graph_view) : graph_viewer_(graph_view) {}

  ONNX_NAMESPACE::GraphProto ToProto(bool include_initializer);

 private:
  const GraphViewer* graph_viewer_;
};
}  // namespace onnxruntime
