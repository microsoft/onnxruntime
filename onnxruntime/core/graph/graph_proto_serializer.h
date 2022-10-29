// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_viewer.h"

namespace onnxruntime {

void GraphViewerToProto(const GraphViewer& graph_view, ONNX_NAMESPACE::GraphProto& graph_proto, bool include_initializer, bool include_outer_scope_args);
}  // namespace onnxruntime
