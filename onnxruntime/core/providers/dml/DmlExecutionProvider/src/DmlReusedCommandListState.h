// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <DirectML.h>

namespace Dml
{
    struct GraphInputInfo
    {
        const onnxruntime::NodeArg* inputArg;
        std::optional<uint32_t> globalInputIndex;
        std::shared_ptr<onnxruntime::Tensor> ownedInputTensor;
    };

    struct GraphOutputInfo
    {
        const onnxruntime::NodeArg* outputArg;
        std::optional<uint32_t> globalOutputIndex;
        std::shared_ptr<onnxruntime::Tensor> ownedOutputTensor;
    };

    struct GraphInfo
    {
        std::vector<GraphInputInfo> inputs;
        std::vector<GraphOutputInfo> outputs;
        std::vector<onnxruntime::Node*> nodes;
        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledOp;
        std::optional<DML_BUFFER_BINDING> persistentResourceBinding;
        Microsoft::WRL::ComPtr<ID3D12Resource> persistentResource;
        Microsoft::WRL::ComPtr<IUnknown> persistentResourceAllocatorUnknown;
    };

    struct DmlReusedCompiledOpInfo
    {
        const GraphInfo* graphInfo;
        Microsoft::WRL::ComPtr<ID3D12Resource> temporaryResource;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
        Microsoft::WRL::ComPtr<IDMLBindingTable> bindingTable;
        uint64_t tempBindingAllocId = 0;
        uint64_t temporaryResourceSize;

        // Bindings from previous executions of a re-used command list
        std::vector<uint64_t> inputBindingAllocIds;
        std::vector<uint64_t> outputBindingAllocIds;
    };

    struct DmlReusedCommandListState
    {
        // Re-usable command list, supporting descriptor heap, and DML binding table to update that heap.
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> graphicsCommandList;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator;
        mutable std::vector<DmlReusedCompiledOpInfo> compiledOpsInfo;

        // Fence tracking the status of the command list's last execution, and whether its descriptor heap
        // can safely be updated.
        mutable Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        mutable uint64_t completionValue = 0;
    };
}
