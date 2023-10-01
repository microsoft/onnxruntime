// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "GraphDescBuilder.h"
#include "DmlRuntimeGraphFusionTransformer.h"

namespace Dml
{
    onnxruntime::OpKernel* CreateRuntimeFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        const onnxruntime::Graph& graph,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        std::unordered_map<std::string, GraphNodeProperties>&& partitionNodePropsMap
    );
} // namespace Dml
