// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "GraphDescBuilder.h"
#include "DmlGraphFusionTransformer.h"

namespace Dml
{
    onnxruntime::OpKernel* CreateFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        Windows::AI::MachineLearning::Adapter::EdgeShapes& outputShapes,
        std::vector<ComPtr<ID3D12Resource>>& nonOwnedGraphInputsFromInitializers,
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& initializeResourceRefs,
        std::vector<uint8_t>&& isInputsUploadedByDmlEP,
        const ExecutionProviderImpl* providerImpl,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        std::unordered_map<std::string, GraphNodeProperties>& partitionNodePropsMap
    );
} // namespace Dml
