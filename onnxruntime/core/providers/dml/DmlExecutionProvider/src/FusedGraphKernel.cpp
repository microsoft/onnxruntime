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
                    m_persistentResourceAllocatorUnknown.GetAddressOf()));

                m_persistentResourceBinding = DML_BUFFER_BINDING { m_persistentResource.Get(), 0, persistentResourceSize };
            }

            ORT_THROW_IF_FAILED(m_provider->InitializeOperator(
                kernelInfo.GetAllocator(OrtMemTypeDefault).get(),
                m_compiledExecutionPlanOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(initInputBindings)));

            // Queue references to objects which must be kept alive until resulting GPU work completes
            m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());
            m_winmlProvider->QueueReference(m_persistentResourceAllocatorUnknown.Get());

            std::for_each(
                initializeResourceRefs.begin(),
                initializeResourceRefs.end(),
                [&](ComPtr<ID3D12Resource>& resource){ m_winmlProvider->QueueReference(WRAP_GRAPHICS_UNKNOWN(resource).Get()); }
            );

            if (reuseCommandList)
            {
                auto reusableCommandList = DmlGraphFusionHelper::BuildReusableCommandList(
                    m_provider.Get(),
                    m_compiledExecutionPlanOperator.Get(),
                    m_persistentResource.Get(),
                    m_persistentResourceBinding);

                m_reusedCommandLists.push_back(std::move(reusableCommandList));
            }
        }

        onnxruntime::Status Compute(onnxruntime::OpKernelContext* kernelContext) const override
        {
            // Only re-use the cached command list if its prior execution is complete on the GPU.
            // This requirement can be avoided by mantaining ring buffers.
            if (m_reusedCommandLists.empty())
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
                    Info().GetAllocator(OrtMemTypeDefault).get(),
                    m_compiledExecutionPlanOperator.Get(),
                    m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                    inputPtrs,
                    aux);

                ORT_THROW_IF_FAILED(m_provider->AddUAVBarrier());

                // Queue references to objects which must be kept alive until resulting GPU work completes
                m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());
                m_winmlProvider->QueueReference(m_persistentResourceAllocatorUnknown.Get());
            }
            else
            {
                if (m_reusedCommandLists.front()->fence &&
                    m_reusedCommandLists.front()->fence->GetCompletedValue() < m_reusedCommandLists.front()->completionValue)
                {
                    auto reusableCommandList = DmlGraphFusionHelper::BuildReusableCommandList(
                        m_provider.Get(),
                        m_compiledExecutionPlanOperator.Get(),
                        m_persistentResource.Get(),
                        m_persistentResourceBinding);

                    m_reusedCommandLists.push_front(std::move(reusableCommandList));
                }

                // We don't need to keep a reference on the temporary resource once we have recorded into the command list, so the
                // memory can be reused by the allocator
                constexpr bool keepTemporaryResourceAlive = false;

                DmlGraphFusionHelper::ExecuteReusableCommandList(
                    kernelContext,
                    *m_reusedCommandLists.front(),
                    m_compiledExecutionPlanOperator.Get(),
                    Info(),
                    m_isInputsUploadedByDmlEP,
                    m_inputsUsed,
                    m_nonOwnedGraphInputsFromInitializers,
                    m_outputShapes,
                    m_winmlProvider.Get(),
                    m_provider.Get(),
                    m_persistentResourceAllocatorUnknown.Get(),
                    keepTemporaryResourceAlive);

                m_reusedCommandLists.push_back(std::move(m_reusedCommandLists.front()));
                m_reusedCommandLists.pop_front();
            }

            return onnxruntime::Status::OK();
        }

        void ExecuteOperator(
            onnxruntime::IAllocator* allocator,
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
                allocator,
                op,
                persistentResourceBinding,
                inputBindings,
                outputBindings));
        }

    private:
        ComPtr<IDMLCompiledOperator> m_compiledExecutionPlanOperator;
        std::vector<bool> m_inputsUsed;
        const void* m_executionHandle = nullptr;
        ComPtr<IWinmlExecutionProvider> m_winmlProvider;
        ComPtr<Dml::IExecutionProvider> m_provider;
        Windows::AI::MachineLearning::Adapter::EdgeShapes& m_outputShapes;

        mutable std::deque<std::unique_ptr<DmlReusedCommandListState>> m_reusedCommandLists;

        std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;
        ComPtr<ID3D12Resource> m_persistentResource;
        ComPtr<IUnknown> m_persistentResourceAllocatorUnknown; // Controls when the persistent resource is returned to the allocator

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
