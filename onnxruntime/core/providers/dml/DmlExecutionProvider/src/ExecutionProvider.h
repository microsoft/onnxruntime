// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "GraphTransformer.h"
#include "core/providers/dml/DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "core/providers/dml/DmlExecutionProvider/src/IExecutionProvider.h"

#include <wrl/client.h>
#include <wrl/implements.h>

namespace WRL {
template <typename... TInterfaces>
using Base = Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    TInterfaces...>;
}

namespace Dml
{
    using Microsoft::WRL::ComPtr;
    class PooledUploadHeap;
    class ReadbackHeap;
    class ExecutionContext;
    class BucketizedBufferAllocator;
    class CPUAllocator;
    class ExecutionProvider;

    class ExecutionProviderImpl : public WRL::Base<Dml::IExecutionProvider,
                                  Windows::AI::MachineLearning::Adapter::IWinmlExecutionProvider>
    {
    public:
        ExecutionProviderImpl(
            IDMLDevice* dmlDevice,
            ID3D12Device* d3d12Device,
            ID3D12CommandQueue* queue,
            bool enableMetacommands,
            bool enableDynamicGraphFusion);

        void ReleaseCompletedReferences();

    public: // implements Dml::IExecutionProvider
        STDMETHOD(GetD3DDevice)(_COM_Outptr_ ID3D12Device** d3dDevice) const noexcept final;

        STDMETHOD(GetDmlDevice)(_COM_Outptr_ IDMLDevice** dmlDevice) const noexcept final;

        STDMETHOD(ExecuteCommandList)(
            ID3D12GraphicsCommandList* commandList,
            _Outptr_ ID3D12Fence** fence,
            _Out_ uint64_t* completionValue
            ) const noexcept final;

        STDMETHOD(AddUAVBarrier)() const noexcept final;

        STDMETHOD(InitializeOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<const DML_BUFFER_BINDING> inputBindings
            ) const noexcept final;

        STDMETHOD(ExecuteOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<IMLOperatorTensor*> inputTensors,
            gsl::span<IMLOperatorTensor*> outputTensors
            ) const noexcept final;

        STDMETHOD(ExecuteOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<DML_BINDING_DESC> inputTensors,
            gsl::span<DML_BINDING_DESC> outputTensors
            ) const noexcept final;

        STDMETHOD(CopyTensor)(IMLOperatorTensor* dst, IMLOperatorTensor* src) const noexcept final;
        STDMETHOD(CopyTensors)(gsl::span<IMLOperatorTensor*> dst, gsl::span<IMLOperatorTensor*> src) const noexcept final;

        STDMETHOD(FillTensorWithPattern)(
            IMLOperatorTensor* dst,
            gsl::span<const std::byte> rawValue
            ) const noexcept final;

        STDMETHOD(UploadToResource)(ID3D12Resource* dstData, const void* srcData, uint64_t srcDataSize) const noexcept final;

        std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
        GetCapability(
            const onnxruntime::GraphViewer& graph,
            const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup
            ) const;

        uint32_t GetSupportedDeviceDataTypeMask() const;

        onnxruntime::common::Status CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst) const;
        onnxruntime::common::Status CopyTensors(const std::vector<onnxruntime::IDataTransfer::SrcDstPair>& src_dst_pairs) const;

        // IWinmlExecutionProvider methods
        void QueueReference(IUnknown* object) override;

        void GetShadowCopyIfRequired(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** dataCopy) const override;

        void GetABIDataInterface(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** abiData) const override;

       uint64_t TryGetPooledAllocationId(
            IUnknown* data,
            bool isInternalOperator) override;

        void GetABIExecutionInterfaceAndInvalidateState(
            bool isInternalOperator,
            IUnknown** abiExecutionObject) const override;

        bool TransitionsRequiredForOperator(
            bool isInternalOperator
        ) override;

        void TransitionResourcesForOperator(
            bool isBeforeOp,
            uint32_t resourceCount,
            IUnknown** resources
        ) override;

        STDMETHOD_(D3D12_COMMAND_LIST_TYPE, GetCommandListTypeForQueue)() const override;
        STDMETHOD_(void, Flush)() const override;

        // Waits for flushed work, discards unflushed work, and discards associated references to
        // prevent circular references.  Must be the last call on the object before destruction.
        void Close() override;

        void WaitForOutstandingWork();

        // Allocate a resource from pools.  Releasing pooledResource returns it to the pool.
        STDMETHOD(AllocatePooledResource)(
            size_t size,
            AllocatorRoundingMode roundingMode,
            ID3D12Resource **d3dResource,
            IUnknown* *pooledResource
        ) const noexcept final;

        STDMETHOD_(ID3D12Resource*, DecodeResource)(void* allocation) const noexcept final;

        std::shared_ptr<onnxruntime::KernelRegistry> GetKernelRegistry() const
        {
            return m_kernelRegistry;
        }

        STDMETHOD_(bool, IsMcdmDevice)() const noexcept final;
        STDMETHOD_(bool, CustomHeapsSupported)() const noexcept final;

        STDMETHOD_(bool, MetacommandsEnabled)() const noexcept final;
        bool DynamicGraphFusionEnabled() const noexcept;
        std::shared_ptr<onnxruntime::IAllocator> GetGpuAllocator();
        std::shared_ptr<onnxruntime::IAllocator> GetCpuInputAllocator();

        std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap>
        GetInternalRegistrationInfoMap() const;

        void IncreasePartitionKernelPrefixVal() const
        {
            m_partitionKernelPrefixVal++;
        }

        uint64_t GetPartitionKernelPrefixVal() const
        {
            return m_partitionKernelPrefixVal;
        }

        onnxruntime::common::Status OnSessionInitializationEnd();
        std::vector<onnxruntime::AllocatorPtr> CreatePreferredAllocators();

    private:
        void Initialize(ID3D12CommandQueue* queue, ExecutionProvider& executionProvider);

        bool IsNodeSupportedByDml(
            const onnxruntime::Node& node,
            const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup,
            uint32_t supportedDeviceDataTypeMask // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        ) const;

        void FlushUploadsIfReady() const;

        ComPtr<ID3D12Device> m_d3d12Device;
        ComPtr<IDMLDevice> m_dmlDevice;
        bool m_isMcdmDevice = false;
        bool m_areCustomHeapsSupported = false;
        bool m_areMetacommandsEnabled = true;
        bool m_dynamicGraphFusionEnabled = false;
        bool m_native16BitShaderOpsSupported = false;
        bool m_sessionInitialized = false;
        std::shared_ptr<ExecutionContext> m_context;
        std::unique_ptr<PooledUploadHeap> m_uploadHeap;
        std::unique_ptr<ReadbackHeap> m_readbackHeap;
        std::shared_ptr<BucketizedBufferAllocator> m_allocator;
        std::shared_ptr<CPUAllocator> m_cpuInputAllocator;
        std::shared_ptr<onnxruntime::KernelRegistry> m_kernelRegistry;
        std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap> m_internalRegInfoMap;
        mutable uint64_t m_partitionKernelPrefixVal = 0;
        bool m_closed = false;
        mutable std::chrono::time_point<std::chrono::steady_clock> m_lastUploadFlushTime;
        static constexpr std::chrono::milliseconds m_batchFlushInterval = std::chrono::milliseconds(10);
    };

    class DataTransfer : public onnxruntime::IDataTransfer
    {
    public:
        DataTransfer() = delete;

        DataTransfer(ExecutionProviderImpl* impl) : m_impl(impl)
        {
        }

        onnxruntime::common::Status CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst) const final
        {
            return m_impl->CopyTensor(src, dst);
        }

        onnxruntime::common::Status CopyTensors(const std::vector<onnxruntime::IDataTransfer::SrcDstPair>& src_dst_pairs) const
        {
            return m_impl->CopyTensors(src_dst_pairs);
        }

        bool CanCopy(const OrtDevice& srcDevice, const OrtDevice& dstDevice) const final
        {
              return (srcDevice.Type() == OrtDevice::GPU) ||
                     (dstDevice.Type() == OrtDevice::GPU);
        }

    private:
        ComPtr<ExecutionProviderImpl> m_impl;
    };

    class ExecutionProvider : public onnxruntime::IExecutionProvider
    {
    public:
        virtual ~ExecutionProvider();
        ExecutionProvider() = delete;

        explicit ExecutionProvider(
            IDMLDevice* dmlDevice,
            ID3D12CommandQueue* commandQueue,
            bool enableMetacommands,
            bool enableDynamicGraphFusion
        );

        std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const final override
        {
            return std::make_unique<DataTransfer>(m_impl.Get());
        }

        const void* GetExecutionHandle() const noexcept final override
        {
            return m_impl.Get();
        }

        std::shared_ptr<onnxruntime::KernelRegistry> GetKernelRegistry() const final override
        {
            return m_impl->GetKernelRegistry();
        }

        std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
            GetCapability(const onnxruntime::GraphViewer& graph,
                const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const final override;

        onnxruntime::common::Status OnSessionInitializationEnd() override
        {
            return m_impl->OnSessionInitializationEnd();
        }

        onnxruntime::Status Sync() const final override
        {
            // Completely wait until the device has completed all preceding tasks.
            // The application could have called SynchronizeBoundOutputs().
            m_impl->WaitForOutstandingWork();
            return Status::OK();
        }

        onnxruntime::Status OnRunEnd(bool /*sync_stream*/, const onnxruntime::RunOptions& /*run_options*/) final override
        {
            // Flush any pending work to the GPU, but don't block for completion, permitting it
            // to overlap other work.
            m_impl->Flush();
            return Status::OK();
        }

        void Flush()
        {
            return m_impl->Flush();
        }

        void ReleaseCompletedReferences()
        {
            return m_impl->ReleaseCompletedReferences();
        }

        ExecutionProviderImpl* GetImpl()
        {
            return m_impl.Get();
        }

        const ExecutionProviderImpl* GetImpl() const
        {
            return m_impl.Get();
        }

        bool DynamicGraphFusionEnabled() const
        {
            return m_impl->DynamicGraphFusionEnabled();
        }

        virtual std::vector<onnxruntime::AllocatorPtr> CreatePreferredAllocators() override
        {
            return m_impl->CreatePreferredAllocators();
        }

    private:
        ComPtr<ExecutionProviderImpl> m_impl;
    };

} // namespace Dml
