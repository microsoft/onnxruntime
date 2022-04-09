#pragma once
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "onnx/onnx_pb.h"
#include "core/graph/graph.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {
namespace function_utils {

std::unique_ptr<ONNX_NAMESPACE::OpSchema> CreateSchema(const Graph& graph,
                                                       const IndexedSubGraph& nodes_to_fuse);

}

}  // namespace onnxruntime
