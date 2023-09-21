// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "MLOperatorAuthorImpl.h"
#include "FusedGraphKernel.h"
#include "DmlGraphFusionHelper.h"

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml
{
    class FusedGraphKernel : public onnxruntime::OpKernel
    {
    public:
        FusedGraphKernel() = delete;

        FusedGraphKernel(
            const onnxruntime::OpKernelInfo& kernelInfo,
            ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator,
            Windows::AI::MachineLearning::Adapter::EdgeShapes& outputShapes,
            bool reuseCommandList,
            std::vector<ComPtr<ID3D12Resource>>& nonOwnedGraphInputsFromInitializers,
            std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& initializeResourceRefs,
            std::vector<DML_BUFFER_BINDING> initInputBindings,
            std::vector<uint8_t>&& isInputsUploadedByDmlEP,
            std::vector<bool>&& inputsUsed) :
        OpKernel(kernelInfo),
        m_compiledExecutionPlanOperator(compiledExecutionPlanOperator),
        m_inputsUsed(std::move(inputsUsed)),
        m_outputShapes(outputShapes),
        m_isInputsUploadedByDmlEP(std::move(isInputsUploadedByDmlEP)),
        m_nonOwnedGraphInputsFromInitializers(nonOwnedGraphInputsFromInitializers)
        {
            // Get the execution provider interfaces
            m_executionHandle = kernelInfo.GetExecutionProvider()->GetExecutionHandle();
            if (m_executionHandle)
            {
                // We assume the execution object inherits IUnknown as its first base
                ComPtr<IUnknown> providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(m_executionHandle));

                // Get the WinML-specific execution provider interface from the execution object.
                ORT_THROW_IF_FAILED(providerExecutionObject.As(&m_provider));
                ORT_THROW_IF_FAILED(providerExecutionObject.As(&m_winmlProvider));
            }

            TranslateAndCompileGraph(
                kernelInfo,
                initializeResourceRefs,
                initInputBindings,
                reuseCommandList);
        }

        void TranslateAndCompileGraph(
            const onnxruntime::OpKernelInfo& kernelInfo,
            std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& initializeResourceRefs,
            std::vector<DML_BUFFER_BINDING> initInputBindings,
            bool reuseCommandList
        )
        {
            // Allocate a persistent resource and initialize the operator
            UINT64 persistentResourceSize = m_compiledExecutionPlanOperator->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_provider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Disabled,
                    m_persistentResource.GetAddressOf(),
                    m_persistentResourceAllocatorUnk.GetAddressOf()));

                m_persistentResourceBinding = DML_BUFFER_BINDING { m_persistentResource.Get(), 0, persistentResourceSize };
            }

            ORT_THROW_IF_FAILED(m_provider->InitializeOperator(
                m_compiledExecutionPlanOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(initInputBindings)));

            // Queue references to objects which must be kept alive until resulting GPU work completes
            m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());
            m_winmlProvider->QueueReference(m_persistentResourceAllocatorUnk.Get());

            std::for_each(
                initializeResourceRefs.begin(),
                initializeResourceRefs.end(),
                [&](ComPtr<ID3D12Resource>& resource){ m_winmlProvider->QueueReference(WRAP_GRAPHICS_UNKNOWN(resource).Get()); }
            );

            if (reuseCommandList)
            {
                BuildReusableCommandList();
            }
        }

        onnxruntime::Status Compute(onnxruntime::OpKernelContext* kernelContext) const override
        {
            // Only re-use the cached command list if its prior execution is complete on the GPU.
            // This requirement can be avoided by mantaining ring buffers.
            if (!m_graphicsCommandList ||
                (m_fence != nullptr && m_fence->GetCompletedValue() < m_completionValue))
            {
                // Wrap tensors as required by Dml::IExecutionProvider::ExecuteOperator
                OpKernelContextWrapper contextWrapper(
                    kernelContext,
                    Info().GetExecutionProvider(),
                    true,
                    nullptr);

                ORT_THROW_IF_FAILED(m_provider->AddUAVBarrier());

                // Get input resources for execution, excluding those which were specified as owned by DML and provided
                // at initialization instead.
                std::vector<ComPtr<IMLOperatorTensor>> inputTensors(kernelContext->InputCount());
                std::vector<ID3D12Resource*> inputPtrs(kernelContext->InputCount());

                for (int i = 0; i < kernelContext->InputCount(); ++i)
                {
                    if (!m_inputsUsed[i])
                    {
                        continue;
                    }

                    if (m_nonOwnedGraphInputsFromInitializers[i])
                    {
                        inputPtrs[i] = m_nonOwnedGraphInputsFromInitializers[i].Get();
                    }
                    else if (!m_isInputsUploadedByDmlEP[i])
                    {
                        ORT_THROW_IF_FAILED(contextWrapper.GetInputTensor(i, inputTensors[i].GetAddressOf()));
                        inputPtrs[i] = m_provider->DecodeResource(MLOperatorTensor(inputTensors[i].Get()).GetDataInterface().Get());
                    }
                }

                auto aux = contextWrapper.GetOutputTensors(m_outputShapes);
                ExecuteOperator(
                    m_compiledExecutionPlanOperator.Get(),
                    m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                    inputPtrs,
                    aux);

                ORT_THROW_IF_FAILED(m_provider->AddUAVBarrier());

                // Queue references to objects which must be kept alive until resulting GPU work completes
                m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());
                m_winmlProvider->QueueReference(m_persistentResourceAllocatorUnk.Get());
            }
            else
            {
                ExecuteReusableCommandList(kernelContext);
            }

            return onnxruntime::Status::OK();
        }

        void ExecuteOperator(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<ID3D12Resource*> inputTensors,
            gsl::span<IMLOperatorTensor*> outputTensors) const
        {
            auto FillBindingsFromTensors = [this](auto& bufferBindings, auto& bindingDescs,  gsl::span<IMLOperatorTensor*>& tensors)
            {
                for (IMLOperatorTensor* tensor : tensors)
                {
                    if (tensor)
                    {
                        assert(tensor->IsDataInterface());
                        ID3D12Resource* resource = m_provider->DecodeResource(MLOperatorTensor(tensor).GetDataInterface().Get());
                        D3D12_RESOURCE_DESC resourceDesc = resource->GetDesc();
                        bufferBindings.push_back({ resource, 0, resourceDesc.Width });
                        bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
                    }
                    else
                    {
                        bufferBindings.push_back({ nullptr, 0, 0 });
                        bindingDescs.push_back({ DML_BINDING_TYPE_NONE, nullptr });
                    }
                }
            };

            auto FillBindingsFromBuffers = [](auto& bufferBindings, auto& bindingDescs,  gsl::span<ID3D12Resource*>& resources)
            {
                for (ID3D12Resource* resource : resources)
                {
                    if (resource)
                    {
                        D3D12_RESOURCE_DESC resourceDesc = resource->GetDesc();
                        bufferBindings.push_back({ resource, 0, resourceDesc.Width });
                        bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
                    }
                    else
                    {
                        bufferBindings.push_back({ nullptr, 0, 0 });
                        bindingDescs.push_back({ DML_BINDING_TYPE_NONE, nullptr });
                    }
                }
            };

            std::vector<DML_BUFFER_BINDING> inputBufferBindings;
            inputBufferBindings.reserve(inputTensors.size());
            std::vector<DML_BINDING_DESC> inputBindings;
            inputBindings.reserve(inputTensors.size());
            FillBindingsFromBuffers(inputBufferBindings, inputBindings, inputTensors);

            std::vector<DML_BUFFER_BINDING> outputBufferBindings;
            outputBufferBindings.reserve(outputTensors.size());
            std::vector<DML_BINDING_DESC> outputBindings;
            outputBindings.reserve(outputTensors.size());
            FillBindingsFromTensors(outputBufferBindings, outputBindings, outputTensors);

            ORT_THROW_IF_FAILED(m_provider->ExecuteOperator(
                op,
                persistentResourceBinding,
                inputBindings,
                outputBindings));
            }

    private:
        void BuildReusableCommandList()
        {
            ComPtr<IDMLDevice> device;
            ORT_THROW_IF_FAILED(m_provider->GetDmlDevice(device.GetAddressOf()));

            DML_BINDING_PROPERTIES execBindingProps = m_compiledExecutionPlanOperator->GetBindingProperties();

            D3D12_DESCRIPTOR_HEAP_DESC desc = {};
            desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            desc.NumDescriptors = execBindingProps.RequiredDescriptorCount;
            desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

            ComPtr<ID3D12Device> d3dDevice;
            ORT_THROW_IF_FAILED(m_provider->GetD3DDevice(d3dDevice.GetAddressOf()));

            ORT_THROW_IF_FAILED(d3dDevice->CreateDescriptorHeap(&desc, IID_GRAPHICS_PPV_ARGS(m_heap.ReleaseAndGetAddressOf())));

            // Create a binding table for execution.
            DML_BINDING_TABLE_DESC bindingTableDesc = {};
            bindingTableDesc.Dispatchable = m_compiledExecutionPlanOperator.Get();
            bindingTableDesc.CPUDescriptorHandle = m_heap->GetCPUDescriptorHandleForHeapStart();
            bindingTableDesc.GPUDescriptorHandle = m_heap->GetGPUDescriptorHandleForHeapStart();
            bindingTableDesc.SizeInDescriptors = execBindingProps.RequiredDescriptorCount;

            ORT_THROW_IF_FAILED(device->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&m_bindingTable)));

            ORT_THROW_IF_FAILED(d3dDevice->CreateCommandAllocator(
                m_provider->GetCommandListTypeForQueue(),
                IID_GRAPHICS_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

            ORT_THROW_IF_FAILED(d3dDevice->CreateCommandList(
                0,
                m_provider->GetCommandListTypeForQueue(),
                m_commandAllocator.Get(),
                nullptr,
                IID_GRAPHICS_PPV_ARGS(m_graphicsCommandList.ReleaseAndGetAddressOf())));

            if (m_persistentResource)
            {
                DML_BINDING_DESC persistentResourceBindingDesc =
                    { DML_BINDING_TYPE_BUFFER, m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr };
                m_bindingTable->BindPersistentResource(&persistentResourceBindingDesc);
            }

            ID3D12DescriptorHeap* descriptorHeaps[] = { m_heap.Get() };
            m_graphicsCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);

            ComPtr<IDMLCommandRecorder> recorder;
            ORT_THROW_IF_FAILED(device->CreateCommandRecorder(IID_PPV_ARGS(recorder.GetAddressOf())));

            recorder->RecordDispatch(m_graphicsCommandList.Get(), m_compiledExecutionPlanOperator.Get(), m_bindingTable.Get());

            ORT_THROW_IF_FAILED(m_graphicsCommandList->Close());
        }

        void ExecuteReusableCommandList(onnxruntime::OpKernelContext* kernelContext) const
        {
            DML_BINDING_PROPERTIES execBindingProps = m_compiledExecutionPlanOperator->GetBindingProperties();

            std::vector<DML_BUFFER_BINDING> inputBindings(kernelContext->InputCount());
            std::vector<DML_BINDING_DESC> inputBindingDescs(kernelContext->InputCount());

            OpKernelContextWrapper contextWrapper(
                kernelContext,
                Info().GetExecutionProvider(),
                true,
                nullptr);

            // Populate input bindings, excluding those which were specified as owned by DML and provided
            // at initialization instead.
            m_inputBindingAllocIds.resize(inputBindings.size());
            bool inputBindingsChanged = false;

            for (uint32_t i = 0; i < inputBindings.size(); ++i)
            {
                if (!m_isInputsUploadedByDmlEP[i] && m_inputsUsed[i])
                {
                    if (m_nonOwnedGraphInputsFromInitializers[i])
                    {
                        inputBindings[i].Buffer = m_nonOwnedGraphInputsFromInitializers[i].Get();
                        inputBindings[i].SizeInBytes = m_nonOwnedGraphInputsFromInitializers[i]->GetDesc().Width;
                        inputBindingDescs[i] = {DML_BINDING_TYPE_BUFFER, &inputBindings[i]};
                    }
                    else
                    {
                        assert(kernelContext->InputType(gsl::narrow_cast<int>(i))->IsTensorType());
                        const onnxruntime::Tensor* tensor = kernelContext->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(i));

                        uint64_t allocId;
                        DmlGraphFusionHelper::UnwrapTensor(m_winmlProvider.Get(), tensor, &inputBindings[i].Buffer, &allocId);
                        inputBindingsChanged = inputBindingsChanged || (!allocId || m_inputBindingAllocIds[i] != allocId);
                        inputBindings[i].Buffer->Release(); // Avoid holding an additional reference
                        inputBindings[i].SizeInBytes = DmlGraphFusionHelper::AlignToPow2<size_t>(tensor->SizeInBytes(), 4);
                        inputBindingDescs[i] = {DML_BINDING_TYPE_BUFFER, &inputBindings[i]};
                        m_inputBindingAllocIds[i] = allocId;
                    }
                }
            }

            if (inputBindingsChanged)
            {
                m_bindingTable->BindInputs(gsl::narrow_cast<uint32_t>(inputBindingDescs.size()), inputBindingDescs.data());
            }

            // Populate Output bindings
            std::vector<DML_BUFFER_BINDING> outputBindings(kernelContext->OutputCount());
            std::vector<DML_BINDING_DESC> outputBindingDescs(kernelContext->OutputCount());

            m_outputBindingAllocIds.resize(outputBindings.size());
            bool outputBindingsChanged = false;

            for (uint32_t i = 0; i < outputBindings.size(); ++i)
            {
                std::vector<int64_t> outputDims;
                outputDims.reserve(m_outputShapes.GetShape(i).size());
                for (uint32_t dimSize : m_outputShapes.GetShape(i))
                {
                    outputDims.push_back(dimSize);
                }

                onnxruntime::Tensor* tensor = kernelContext->Output(
                    static_cast<int>(i),
                    onnxruntime::TensorShape::FromExistingBuffer(outputDims)
                    );

                uint64_t allocId;
                DmlGraphFusionHelper::UnwrapTensor(m_winmlProvider.Get(), tensor, &outputBindings[i].Buffer, &allocId);
                outputBindingsChanged = outputBindingsChanged || (!allocId || m_outputBindingAllocIds[i] != allocId);
                outputBindings[i].Buffer->Release(); // Avoid holding an additional reference
                outputBindings[i].SizeInBytes = DmlGraphFusionHelper::AlignToPow2<size_t>(tensor->SizeInBytes(), 4);
                outputBindingDescs[i] = {DML_BINDING_TYPE_BUFFER, &outputBindings[i]};
                m_outputBindingAllocIds[i] = allocId;
            }

            if (outputBindingsChanged)
            {
                m_bindingTable->BindOutputs(gsl::narrow_cast<uint32_t>(outputBindingDescs.size()), outputBindingDescs.data());
            }

            if (execBindingProps.TemporaryResourceSize > 0)
            {
                // Allocate temporary data which will automatically be freed when the GPU work
                // which is scheduled up to the point that this method returns has completed.
                ComPtr<IUnknown> tempAlloc;
                uint64_t tempAllocId = 0;
                ORT_THROW_IF_FAILED(contextWrapper.AllocateTemporaryData(static_cast<size_t>(execBindingProps.TemporaryResourceSize), tempAlloc.GetAddressOf(), &tempAllocId));

                ComPtr<IUnknown> tempResourceUnk;
                m_winmlProvider->GetABIDataInterface(false, tempAlloc.Get(), &tempResourceUnk);

                // Bind the temporary resource.
                ComPtr<ID3D12Resource> tempResource;
                ORT_THROW_IF_FAILED(tempResourceUnk->QueryInterface(tempResource.GetAddressOf()));
                DML_BUFFER_BINDING tempBufferBinding = {tempResource.Get(), 0, execBindingProps.TemporaryResourceSize};
                DML_BINDING_DESC tempBindingDesc = { DML_BINDING_TYPE_BUFFER, &tempBufferBinding };

                if (!tempAllocId || m_tempBindingAllocId != tempAllocId)
                {
                    m_bindingTable->BindTemporaryResource(&tempBindingDesc);
                }

                m_tempBindingAllocId = tempAllocId;
            }

            // Execute the command list and if it succeeds, update the fence value at which this command may be
            // re-used.
            ComPtr<ID3D12Fence> fence;
            uint64_t completionValue;
            HRESULT hr = m_provider->ExecuteCommandList(m_graphicsCommandList.Get(), fence.GetAddressOf(), &completionValue);

            if (hr == DXGI_ERROR_DEVICE_REMOVED)
            {
                ComPtr<ID3D12Device> device;
                ORT_THROW_IF_FAILED(m_provider->GetD3DDevice(&device));
                ORT_THROW_IF_FAILED(device->GetDeviceRemovedReason());
            }

            ORT_THROW_IF_FAILED(hr);
            m_fence = fence;
            m_completionValue = completionValue;

            // Queue references to objects which must be kept alive until resulting GPU work completes
            m_winmlProvider->QueueReference(WRAP_GRAPHICS_UNKNOWN(m_graphicsCommandList).Get());
            m_winmlProvider->QueueReference(WRAP_GRAPHICS_UNKNOWN(m_heap).Get());
            m_winmlProvider->QueueReference(m_bindingTable.Get());
            m_winmlProvider->QueueReference(m_persistentResourceAllocatorUnk.Get());
        }

        ComPtr<IDMLCompiledOperator> m_compiledExecutionPlanOperator;
        std::vector<bool> m_inputsUsed;
        const void* m_executionHandle = nullptr;
        ComPtr<IWinmlExecutionProvider> m_winmlProvider;
        ComPtr<Dml::IExecutionProvider> m_provider;
        Windows::AI::MachineLearning::Adapter::EdgeShapes& m_outputShapes;

        // Re-usable command list, supporting descriptor heap, and DML binding table to update that heap.
        ComPtr<ID3D12GraphicsCommandList> m_graphicsCommandList;
        ComPtr<ID3D12CommandAllocator> m_commandAllocator;
        ComPtr<ID3D12DescriptorHeap> m_heap;
        ComPtr<IDMLBindingTable> m_bindingTable;
        std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;
        ComPtr<ID3D12Resource> m_persistentResource;
        ComPtr<IUnknown> m_persistentResourceAllocatorUnk; // Controls when the persistent resource is returned to the allocator

        // Bindings from previous executions of a re-used command list
        mutable std::vector<uint64_t> m_inputBindingAllocIds;
        mutable std::vector<uint64_t> m_outputBindingAllocIds;
        mutable uint64_t m_tempBindingAllocId = 0;

        // Fence tracking the status of the command list's last execution, and whether its descriptor heap
        // can safely be updated.
        mutable ComPtr<ID3D12Fence> m_fence;
        mutable uint64_t m_completionValue = 0;

        std::vector<uint8_t> m_isInputsUploadedByDmlEP;
        std::vector<ComPtr<ID3D12Resource>> m_nonOwnedGraphInputsFromInitializers;
    };

    onnxruntime::OpKernel* CreateFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator,
        Windows::AI::MachineLearning::Adapter::EdgeShapes& outputShapes,
        bool reuseCommandList,
        std::vector<ComPtr<ID3D12Resource>>& nonOwnedGraphInputsFromInitializers,
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& initializeResourceRefs,
        std::vector<DML_BUFFER_BINDING> initInputBindings,
        std::vector<uint8_t>&& isInputsUploadedByDmlEP,
        std::vector<bool>&& inputsUsed
        )
    {
        return new FusedGraphKernel(
            info,
            compiledExecutionPlanOperator,
            outputShapes,
            reuseCommandList,
            nonOwnedGraphInputsFromInitializers,
            initializeResourceRefs,
            initInputBindings,
            std::move(isInputsUploadedByDmlEP),
            std::move(inputsUsed)
        );
    }
} // namespace Dml
