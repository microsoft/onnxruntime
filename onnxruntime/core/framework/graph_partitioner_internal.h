// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Internal header for tests of graph_partitioner.cc. Not part of any public API.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {

Status ApplyFusedNodeMemTypes(KernelDefBuilder& builder,
                              const NodeComputeInfo& info,
                              const IndexedSubGraph::MetaDef& metadef);

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
