// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "GraphDescBuilder.h"
#include "DmlGraphFusionTransformer.h"

namespace Dml
{
    onnxruntime::OpKernel* CreateFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator,
        Windows::AI::MachineLearning::Adapter::EdgeShapes& outputShapes,
        bool reuseCommandList,
        std::vector<ComPtr<ID3D12Resource>>& nonOwnedGraphInputsFromInitializers,
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& initializeResourceRefs,
        std::vector<uint8_t>& inputsConstant,
        std::vector<bool>& inputsUsed,
        ComPtr<ID3D12Resource> persistentResource,
        ComPtr<IUnknown> persistentResourceAllocatorUnk,
        std::optional<DML_BUFFER_BINDING> persistentResourceBinding
    );
} // namespace Dml
