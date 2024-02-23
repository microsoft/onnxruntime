// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

namespace onnxruntime {
class Graph;
class IExecutionProvider;

// Layout transformation related functions.
namespace layout_transformation {
// DebugGraphFn can be used to debug the graph modifications made during layout transformation.
// See kDebugLayoutTransformation in /include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h for
// more details.
using DebugGraphFn = std::function<void(const Graph&)>;

// TransformLayoutFunction is used by GraphPartitioner when transforming a graph from NCHW to NHWC if the EP has a
// preferred layout of NHWC.
//
// It is optionally provided to graph partitioning by InferenceSession (core/session),
// and used in layout transformation (core/optimizers/transpose_optimizer)
using TransformLayoutFunction = std::function<Status(Graph& graph, bool& modified, IExecutionProvider& current_ep,
                                                     const DebugGraphFn& debug_graph_fn)>;
}  // namespace layout_transformation
}  // namespace onnxruntime
