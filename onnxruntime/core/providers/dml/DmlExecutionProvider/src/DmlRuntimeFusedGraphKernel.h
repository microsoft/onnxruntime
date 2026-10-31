// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <filesystem>
#include "core/framework/op_kernel.h"
#include "GraphDescBuilder.h"
#include "DmlRuntimeGraphFusionTransformer.h"

namespace Dml
{
    // `modelPath` is taken by value: the kernel outlives the caller that supplies it (the creation
    // callback registered by DmlGraphFusionHelper::RegisterDynamicKernel runs after that frame has
    // returned), so the kernel must own its own copy rather than alias the caller's.
    onnxruntime::OpKernel* CreateRuntimeFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        std::shared_ptr<const onnxruntime::IndexedSubGraph> indexedSubGraph,
        std::filesystem::path modelPath,
        std::vector<std::shared_ptr<onnxruntime::Node>>&& subgraphNodes,
        std::vector<const onnxruntime::NodeArg*>&& subgraphInputs,
        std::vector<const onnxruntime::NodeArg*>&& subgraphOutputs,
        std::vector<std::shared_ptr<onnxruntime::NodeArg>>&& intermediateNodeArgs,
        std::unordered_map<std::string, GraphNodeProperties>&& partitionNodePropsMap,
        std::vector<ONNX_NAMESPACE::TensorProto>&& ownedInitializers
    );
} // namespace Dml
