// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "IExecutionProvider.h"
#include "ExecutionProvider.h"
#include "PooledUploadHeap.h"
#include "ReadbackHeap.h"
#include "ExecutionContext.h"
#include "BucketizedBufferAllocator.h"
#include "MLOperatorAuthorImpl.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorHelper.h"
#include "AbiCustomRegistry.h"
#include "GraphPartitioner.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/framework/compute_capability.h"
#include "core/framework/fallback_cpu_capability.h"
#include "DmlCommittedResourceAllocator.h"
#include "DmlCommittedResourceWrapper.h"

#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
#define ERROR 0

#include "core/session/onnxruntime_c_api.h"
#include <wil/wrl.h>
#ifndef _GAMING_XBOX
#include <dxgi1_6.h>
#endif

#define ENABLE_GRAPH_COMPILATION

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml
{
    using namespace onnxruntime::common;

    ExecutionProvider::~ExecutionProvider()
    {
        if (m_impl)
        {
            m_impl->Close();
        }
    }

    static void CreateDmlKernelRegistry(
        _Out_ std::shared_ptr<onnxruntime::KernelRegistry>* registry,
        _Out_ std::shared_ptr<const InternalRegistrationInfoMap>* internalRegInfoMap)
    {
        ComPtr<AbiCustomRegistry> abiRegistry = wil::MakeOrThrow<AbiCustomRegistry>();
        Dml::RegisterDmlOperators(abiRegistry.Get());

        assert(abiRegistry->GetRegistries().size() == 1);

        auto customRegistry = *abiRegistry->GetRegistries().begin();
        *registry = customRegistry->GetKernelRegistry();
        *internalRegInfoMap = abiRegistry->GetInternalRegInfoMap();

        Dml::RegisterCpuOperatorsAsDml(registry->get());
    }

    ExecutionProvider::ExecutionProvider(
        IDMLDevice* dmlDevice,
        ID3D12CommandQueue* commandQueue,
        bool enableMetacommands) :
            IExecutionProvider(onnxruntime::kDmlExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0))
    {
        D3D12_COMMAND_LIST_TYPE queueType = commandQueue->GetDesc().Type;
        if (queueType != D3D12_COMMAND_LIST_TYPE_DIRECT && queueType != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        {
            // DML requires either DIRECT or COMPUTE command queues.
            ORT_THROW_HR(E_INVALIDARG);
        }

        ComPtr<ID3D12Device> device;
        GRAPHICS_THROW_IF_FAILED(commandQueue->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf())));

        m_impl = wil::MakeOrThrow<ExecutionProviderImpl>(dmlDevice, device.Get(), commandQueue, enableMetacommands);
    }

    std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
    ExecutionProvider::GetCapability(
        const onnxruntime::GraphViewer& graph,
        const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const
    {
#ifdef ENABLE_GRAPH_COMPILATION
        return m_impl->GetCapability(graph, kernel_lookup);
#else
        return onnxruntime::IExecutionProvider::GetCapability(graph, kernel_lookup);
#endif
    }

    void ExecutionProviderImpl::Close()
    {
        m_context->Close();
    }

    void ExecutionProviderImpl::WaitForOutstandingWork()
    {
        Flush();
        m_context->GetCurrentCompletionEvent().WaitForSignal();
    }

    HRESULT __stdcall ExecutionProviderImpl::AllocatePooledResource(
        size_t size,
        AllocatorRoundingMode roundingMode,
        ID3D12Resource **d3dResource,
        IUnknown** pooledResource
    ) const noexcept
    {
        ORT_TRY
        {
        ComPtr<IUnknown> allocation;
        allocation.Attach(static_cast<IUnknown* >(m_allocator->Alloc(size, roundingMode)));

        const auto* allocInfo = m_allocator->DecodeDataHandle(allocation.Get());

        ComPtr<ID3D12Resource> resource = allocInfo->GetResource();
        resource.CopyTo(d3dResource);
        *pooledResource = allocation.Detach();
        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    ID3D12Resource* __stdcall ExecutionProviderImpl::DecodeResource(void* allocation) const noexcept
    {
        ORT_TRY
        {
            const AllocationInfo* allocInfo = m_allocator->DecodeDataHandle(allocation);
            return allocInfo->GetResource();
        }
        ORT_CATCH_GENERIC
        {
            return nullptr;
        }
    }

// ORT release pipelines agent pools do not have 19H1 SDK installed which defines D3D_FEATURE_LEVEL_1_0_CORE.
// Once ORT/WinML github project can be built with VS2019, we can update these pools to use install the 19H1 SDK
// using the command line installer tool with VS2019
// Task 24384515: Update ORT AIInfra release agent pool to install 19H1 SDK on VM bootstrap
#define D3D_FEATURE_LEVEL_1_0_CORE_PRIVATE ((D3D_FEATURE_LEVEL)0x1000)

    ExecutionProviderImpl::ExecutionProviderImpl(IDMLDevice* dmlDevice, ID3D12Device* d3d12Device, ID3D12CommandQueue* queue, bool enableMetacommands)
        : m_d3d12Device(d3d12Device),
          m_dmlDevice(dmlDevice),
          m_areMetacommandsEnabled(enableMetacommands)
    {

        D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {};

        D3D_FEATURE_LEVEL featureLevelsList[] = {
            D3D_FEATURE_LEVEL_1_0_CORE_PRIVATE,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_12_1
        };

        featureLevels.NumFeatureLevels = ARRAYSIZE(featureLevelsList);
        featureLevels.pFeatureLevelsRequested = featureLevelsList;
        ORT_THROW_IF_FAILED(d3d12Device->CheckFeatureSupport(
            D3D12_FEATURE_FEATURE_LEVELS,
            &featureLevels,
            sizeof(featureLevels)
            ));

        D3D12_FEATURE_DATA_D3D12_OPTIONS4 featureOptions = {};
        if (SUCCEEDED(d3d12Device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS4,
            &featureOptions,
            sizeof(featureOptions))))
        {
            m_native16BitShaderOpsSupported = featureOptions.Native16BitShaderOpsSupported;
        }

        m_isMcdmDevice = (featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE_PRIVATE);
        m_areCustomHeapsSupported = !m_isMcdmDevice;

        if (m_isMcdmDevice) {

            // TODO: Ingest updated header file
            typedef struct D3D12_FEATURE_DATA_D3D12_OPTIONS19
            {
                BOOL MismatchingOutputDimensionsSupported;
                UINT SupportedSampleCountsWithNoOutputs;
                BOOL PointSamplingAddressesNeverRoundUp;
                BOOL RasterizerDesc2Supported;
                BOOL NarrowQuadrilateralLinesSupported;
                BOOL AnisoFilterWithPointMipSupported;
                UINT MaxSamplerDescriptorHeapSize;
                UINT MaxSamplerDescriptorHeapSizeWithStaticSamplers;
                UINT MaxViewDescriptorHeapSize;
                _Out_  BOOL ComputeOnlyCustomHeapSupported;
            } 	D3D12_FEATURE_DATA_D3D12_OPTIONS19;

            D3D12_FEATURE_DATA_D3D12_OPTIONS19 options19 = {};

            // The call may fail in which case the default value is false
            d3d12Device->CheckFeatureSupport((D3D12_FEATURE) 48 /*D3D12_FEATURE_D3D12_OPTIONS19*/, &options19, sizeof(options19));    
            m_areCustomHeapsSupported = options19.ComputeOnlyCustomHeapSupported;
        }

        m_context = std::make_shared<ExecutionContext>(m_d3d12Device.Get(), m_dmlDevice.Get(), queue);

        m_uploadHeap = std::make_unique<PooledUploadHeap>(m_d3d12Device.Get(), m_context);
        m_readbackHeap = std::make_unique<ReadbackHeap>(m_d3d12Device.Get(), m_context);

        CreateDmlKernelRegistry(&m_kernelRegistry, &m_internalRegInfoMap);

        m_lastUploadFlushTime = std::chrono::steady_clock::now();
    }

    std::vector<onnxruntime::AllocatorPtr> ExecutionProviderImpl::CreatePreferredAllocators() {
        if (!m_allocator)
        {
            // Create an allocator for D3D12 buffers used to hold tensor data. The returned buffers from the allocator
            // should be DEFAULT heap buffers which can be used as UAVs, and which start in UAV state.
            m_allocator = std::make_shared<BucketizedBufferAllocator>(m_d3d12Device.Get(),
                m_context,  // TODO(leca): REVIEW: Will it cause memory issue when m_context is released in EP while alloc is released in sessionState?
                CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                std::make_unique<DmlCommittedResourceAllocator>(m_d3d12Device.Get()));
            m_context->SetAllocator(m_allocator);
            // CPU Allocator used to create buffers for the MemcpyFromHost, Shape and Size operators.
            m_cpuInputAllocator = std::make_shared<CPUAllocator>(OrtMemType::OrtMemTypeCPUInput);
        }

        return std::vector<onnxruntime::AllocatorPtr>{m_allocator, m_cpuInputAllocator,};
    }

    HRESULT __stdcall ExecutionProviderImpl::GetD3DDevice(_COM_Outptr_ ID3D12Device** d3dDevice) const noexcept
    {
        m_d3d12Device.CopyTo(d3dDevice);
        _Analysis_assume_(*d3dDevice != nullptr);
        return S_OK;
    }
    
    HRESULT __stdcall ExecutionProviderImpl::GetCommandQueue(_COM_Outptr_ ID3D12CommandQueue** queue) const noexcept
    {
        return m_context->GetCommandQueue(queue);
    }

    HRESULT __stdcall ExecutionProviderImpl::GetDmlDevice(_COM_Outptr_ IDMLDevice** dmlDevice) const noexcept
    {
        m_dmlDevice.CopyTo(dmlDevice);
        _Analysis_assume_(*dmlDevice != nullptr);
        return S_OK;
    }

    HRESULT __stdcall ExecutionProviderImpl::ExecuteCommandList(
        ID3D12GraphicsCommandList* commandList,
        _Outptr_ ID3D12Fence** fence,
        _Out_ uint64_t* completionValue
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);
        m_context->ExecuteCommandList(commandList, fence, completionValue);

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall ExecutionProviderImpl::AddUAVBarrier() const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        m_context->AddUAVBarrier();

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall ExecutionProviderImpl::InitializeOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<const DML_BUFFER_BINDING> inputBindings
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        bool hasInputsToBind = false;
        std::vector<DML_BUFFER_BINDING> inputBufferBindings(inputBindings.size());

        for (size_t i = 0; i < inputBindings.size(); i++)
        {
            if (inputBindings[i].Buffer)
            {
                hasInputsToBind = true;
                inputBufferBindings[i] = { inputBindings[i].Buffer, inputBindings[i].Offset, inputBindings[i].SizeInBytes };
            }
        }

        DML_BINDING_DESC persistentResourceBindingDesc =
            persistentResourceBinding
            ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, persistentResourceBinding }
            : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        DML_BUFFER_ARRAY_BINDING inputBufferArrayDesc;
        inputBufferArrayDesc.BindingCount = gsl::narrow_cast<uint32_t>(inputBufferBindings.size());
        inputBufferArrayDesc.Bindings = inputBufferBindings.data();

        DML_BINDING_DESC inputArrayBindingDesc = hasInputsToBind ?
            DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER_ARRAY, &inputBufferArrayDesc } :
            DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        m_context->InitializeOperator(
            op,
            persistentResourceBindingDesc,
            inputArrayBindingDesc);

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall ExecutionProviderImpl::ExecuteOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<IMLOperatorTensor*> inputTensors,
        gsl::span<IMLOperatorTensor*> outputTensors
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        std::vector<uint32_t> shape;

        for (IMLOperatorTensor* tensor : inputTensors)
        {
            if (tensor)
            {
                shape.resize(tensor->GetDimensionCount());
                ORT_THROW_IF_FAILED(tensor->GetShape(tensor->GetDimensionCount(), shape.data()));

                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    return S_OK;
                }
            }
        }

        for (IMLOperatorTensor* tensor : outputTensors)
        {
            if (tensor)
            {
                shape.resize(tensor->GetDimensionCount());
                ORT_THROW_IF_FAILED(tensor->GetShape(tensor->GetDimensionCount(), shape.data()));

                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    return S_OK;
                }
            }
        }

        auto FillBindings = [this](auto& bufferBindings, auto& bindingDescs, auto& tensors)
        {
            for (IMLOperatorTensor* tensor : tensors)
            {
                if (tensor)
                {
                    assert(tensor->IsDataInterface());
                    const AllocationInfo* allocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(tensor).GetDataInterface().Get());
                    ID3D12Resource* resource = allocInfo->GetResource();
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
        FillBindings(inputBufferBindings, inputBindings, inputTensors);

        std::vector<DML_BUFFER_BINDING> outputBufferBindings;
        outputBufferBindings.reserve(outputTensors.size());
        std::vector<DML_BINDING_DESC> outputBindings;
        outputBindings.reserve(outputTensors.size());
        FillBindings(outputBufferBindings, outputBindings, outputTensors);

        ORT_THROW_IF_FAILED(ExecuteOperator(op, persistentResourceBinding, inputBindings, outputBindings));

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall ExecutionProviderImpl::ExecuteOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<DML_BINDING_DESC> inputTensors,
        gsl::span<DML_BINDING_DESC> outputTensors
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        DML_BINDING_DESC persistentResourceBindingDesc =
            persistentResourceBinding
            ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, persistentResourceBinding }
            : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        m_context->ExecuteOperator(
            op,
            persistentResourceBindingDesc,
            inputTensors,
            outputTensors);

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    static gsl::span<const std::byte> AsByteSpan(const void* data, size_t sizeInBytes)
    {
        return gsl::make_span(static_cast<const std::byte*>(data), sizeInBytes);
    }

    static gsl::span<std::byte> AsByteSpan(void* data, size_t sizeInBytes)
    {
        return gsl::make_span(static_cast<std::byte*>(data), sizeInBytes);
    }

    HRESULT __stdcall ExecutionProviderImpl::CopyTensor(IMLOperatorTensor* dst, IMLOperatorTensor* src) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        const size_t sourceSizeInBytes = ComputeByteSizeFromTensor(*src);
        const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst);
        ORT_THROW_HR_IF(E_INVALIDARG, dataSizeInBytes != sourceSizeInBytes); // Tensors must be the same size

        if (dataSizeInBytes == 0)
        {
            return S_OK;
        }

        if (src->IsCpuData() && !dst->IsCpuData())
        {
            //
            // CPU -> GPU copy (upload)
            //
            const AllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());

            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            const void* srcData = src->GetData();

            constexpr uint64_t dstOffset = 0;
            const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

            m_uploadHeap->BeginUploadToGpu(dstData, dstOffset, dstState, AsByteSpan(srcData, dataSizeInBytes));
            FlushUploadsIfReady();
        }
        else if (!src->IsCpuData() && dst->IsCpuData())
        {
            //
            // GPU -> CPU copy (readback)
            //

            void* dstData = dst->GetData();
            const AllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());

            ID3D12Resource* srcData = srcAllocInfo->GetResource();

            const uint64_t srcOffset = 0;
            const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

            // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
            m_readbackHeap->ReadbackFromGpu(AsByteSpan(dstData, dataSizeInBytes), srcData, srcOffset, srcState);
        }
        else if (!src->IsCpuData() && !dst->IsCpuData())
        {
            //
            // GPU -> GPU copy
            //
            const AllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());
            const AllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());

            ID3D12Resource* srcData = srcAllocInfo->GetResource();
            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            m_context->CopyBufferRegion(dstData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, srcData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, dataSizeInBytes);
        }
        else
        {
            // CPU -> CPU copies not supported
            ORT_THROW_HR(E_INVALIDARG);
        }

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall ExecutionProviderImpl::CopyTensors(gsl::span<IMLOperatorTensor*> dst, gsl::span<IMLOperatorTensor*> src) const noexcept
    {
        ORT_TRY
        {
        ORT_THROW_HR_IF(E_INVALIDARG, dst.size() != src.size());

        // Source and destination for batched GPU -> CPU copies
        std::vector<ID3D12Resource*> srcDatas;
        std::vector<void*> dstDatas;
        std::vector<uint32_t> dataSizesInBytes;

        assert(!m_closed);
        auto provider = const_cast<ExecutionProviderImpl*>(this);

        for (uint32_t i = 0; i < dst.size(); ++i)
        {
            // This batching implementation only handles GPU -> CPU copies.  Other copies do not require synchronization
            // and are batched across multiple calls to CopyTensor.
            if (src[i]->IsCpuData() || !dst[i]->IsCpuData())
            {
                ORT_THROW_IF_FAILED(CopyTensor(dst[i], src[i]));
                continue;
            }

            const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst[i]);
            ORT_THROW_HR_IF(E_INVALIDARG, dataSizeInBytes != ComputeByteSizeFromTensor(*src[i])); // Tensors must be the same size

            if (dataSizeInBytes == 0)
            {
                continue;
            }

            dataSizesInBytes.push_back(static_cast<uint32_t>(ComputeByteSizeFromTensor(*dst[i])));
            ORT_THROW_HR_IF(E_INVALIDARG, dataSizesInBytes.back() != ComputeByteSizeFromTensor(*src[i])); // Tensors must be the same size

            dstDatas.push_back(dst[i]->GetData());
            const AllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src[i]).GetDataInterface().Get());

            srcDatas.push_back(srcAllocInfo->GetResource());
        }

        const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

        // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
        m_readbackHeap->ReadbackFromGpu(dstDatas, dataSizesInBytes, srcDatas, srcState);

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE ExecutionProviderImpl::FillTensorWithPattern(
        IMLOperatorTensor* dst,
        gsl::span<const std::byte> rawValue // Data type agnostic rawValue, treated as raw bits
        ) const noexcept
    {
        ORT_TRY
        {
        auto mlTensor = MLOperatorTensor(dst).GetDataInterface();
        if (mlTensor != nullptr)
        {
            const AllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(mlTensor.Get());
            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            m_context->FillBufferWithPattern(dstData, rawValue);
        }

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall ExecutionProviderImpl::UploadToResource(ID3D12Resource* dstData, const void* srcData, uint64_t srcDataSize) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        m_uploadHeap->BeginUploadToGpu(dstData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, AsByteSpan(srcData, static_cast<size_t>(srcDataSize)));
        FlushUploadsIfReady();

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    void ExecutionProviderImpl::FlushUploadsIfReady() const
    {
        // Periodically flush uploads to make sure the GPU is not idle for too long
        if (std::chrono::steady_clock::now() - m_lastUploadFlushTime > m_batchFlushInterval)
        {
            Flush();
            m_lastUploadFlushTime = std::chrono::steady_clock::now();
        }
    }

    uint32_t ExecutionProviderImpl::GetSupportedDeviceDataTypeMask() const
    {
        // The DML provider registers all supported kernels up-front regardless of actual device capability,
        // but this is problematic later when executing the graph because DirectML will fail to create
        // the operator, and by that late phase, it's long past too late to recover. So, this function queries
        // the actual type capabilities so the partitioner may assigns nodes to the CPU if the GPU cannot
        // handle them, similar to the fallback in CUDAExecutionProvider::GetCapability for certain RNN/GRU/Conv
        // attributes.

        return Dml::GetSupportedDeviceDataTypeMask(m_dmlDevice.Get());
    }

    bool TryGetTensorDataType(
        const onnxruntime::NodeArg& nodeArg,
        _Out_ MLOperatorEdgeType* edgeType,
        _Out_ MLOperatorTensorDataType* onnxElementType
    )
    {
        *onnxElementType = MLOperatorTensorDataType::Undefined;
        *edgeType = MLOperatorEdgeType::Undefined;

        const ::onnx::TypeProto* typeProto = nodeArg.TypeAsProto();
        if (typeProto != nullptr)
        {
            const ::onnx::TypeProto_Tensor* tensorTypeProto;
            if (typeProto->has_tensor_type())
            {
                *edgeType = MLOperatorEdgeType::Tensor;
                tensorTypeProto = &typeProto->tensor_type();
            }
            else if (typeProto->has_sequence_type())
            {
                *edgeType = MLOperatorEdgeType::SequenceTensor;
                tensorTypeProto = &typeProto->sequence_type().elem_type().tensor_type();
            }
            else
            {
                return false;
            }

            if (tensorTypeProto->has_elem_type())
            {
                *onnxElementType = static_cast<MLOperatorTensorDataType>(tensorTypeProto->elem_type());
                return true;
            }
        }

        return false;
    }

    bool IsCpuOnDmlOperator(const onnxruntime::Node& node)
    {
        auto cpuOnDmlOperators = std::array<char*, 8>{
            "SequenceAt",
            "SequenceConstruct",
            "SequenceEmpty",
            "SequenceLength",
            "SequenceErase",
            "SequenceInsert",
            "OptionalGetElement",
            "OptionalHasElement"
        };

        for (auto& cpuOnDmlOperator : cpuOnDmlOperators)
        {
            if (strcmp(cpuOnDmlOperator, node.OpType().c_str()) == 0)
            {
                return true;
            }
        }
        return false;
    }

    bool IsDmlSequenceOperator(const onnxruntime::Node& node)
    {
        auto sequence_ops = std::array<char*, 1>{
            "ConcatFromSequence"
        };

        for (auto& sequence_op : sequence_ops)
        {
            if (strcmp(sequence_op, node.OpType().c_str()) == 0)
            {
                return true;
            }
        }
        return false;
    }

    bool IsCustomOpShader(const onnxruntime::Node& node)
    {
        auto custom_ops = std::array<char*, 3>{
            "DFT",
            "STFT",
            "GridSample"
        };

        for (auto& custom_op : custom_ops)
        {
            if (strcmp(custom_op, node.OpType().c_str()) == 0)
            {
                return true;
            }
        }
        return false;
    }

    bool DoesNodeContainSupportedDataTypes(
        const onnxruntime::Node& node,
        _In_opt_ const InternalRegistrationInfo* regInfo,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        bool native16BitShaderOpsSupported
        )
    {
        std::vector<onnxruntime::NodeArg const*> constantCpuInputs;

        if (regInfo != nullptr)
        {
            // Collect the list of CPU-bound input tensors, needed when checking 64-bit fallback
            // or for other data types like int-8 which may be supported for CPU inputs but not
            // GPU inputs.
            auto inputDefinitions = node.InputDefs();
            for (uint32_t i : regInfo->requiredConstantCpuInputs)
            {
                if (i < inputDefinitions.size())
                {
                    constantCpuInputs.push_back(inputDefinitions[i]);
                }
            }
        }

        // Assume data types are supported until proven otherwise.
        bool nodeContainsSupportedDataTypes = true;

        // Callback to check each node's data type against registered operator support.
        std::function<void(const onnxruntime::NodeArg& nodeArg, bool isInput)> nodeCallback = [&](const onnxruntime::NodeArg& nodeArg, bool isInput) -> void
        {
            // Get the tensor element data type for this node, comparing against what the device actually supports.
            // Use the enumeration from the proto instead of nodeArg.Type() which returns a string.

            // Reject node if undefined data type or non-tensor, as DML cannot handle it.
            MLOperatorEdgeType edgeType;
            MLOperatorTensorDataType onnxElementType;
            if (!TryGetTensorDataType(nodeArg, &edgeType, &onnxElementType))
            {
                // We shouldn't have arrived here because (1) no DML operators should have been
                // registered which use non-tensor types (2) ONNX validation should have already
                // been done, checking for the right kind of inputs and attributes. In theory,
                // this branch could be reached with a bad custom operator or malformed file. If
                // a legitimate case reaches here and DML needs to support a new input/output type
                // besides tensors, then remove the assert.
                assert(false);
                nodeContainsSupportedDataTypes = false;
                return;
            }

            if (onnxElementType == MLOperatorTensorDataType::Float16 &&
                !native16BitShaderOpsSupported &&
                IsCustomOpShader(node))
            {
                nodeContainsSupportedDataTypes = false;
                return;
            }

            // Allow nodeArgs that are SequenceTensor when they are actually implemented by CPU Kernels.
            if (edgeType == MLOperatorEdgeType::SequenceTensor)
            {
                if (!IsCpuOnDmlOperator(node) && !IsDmlSequenceOperator(node))
                {
                    nodeContainsSupportedDataTypes = false;
                }
                return;
            }

            // Reject node for unknown DML data types.
            DML_TENSOR_DATA_TYPE dmlElementType = GetDmlDataTypeFromMlDataTypeNoThrow(onnxElementType);
            if (dmlElementType == DML_TENSOR_DATA_TYPE_UNKNOWN)
            {
                nodeContainsSupportedDataTypes = false;
                return;
            }

            // Succeed if the tensor is CPU-bound, as the CPU-side reading code is generic enough
            // to handle multiple types regardless of GPU capability (typically these are just
            // scalars or simple 1D arrays).
            bool isConstantCpuInput = isInput && std::find(constantCpuInputs.begin(), constantCpuInputs.end(), &nodeArg) != constantCpuInputs.end();
            if (isConstantCpuInput)
            {
                // Leave nodeContainsSupportedDataTypes alone.
                return;
            }

            bool isDataTypeSupported = (1 << dmlElementType) & supportedDeviceDataTypeMask;

            // Reject node if the data type is unsupported by the device.
            if (!isDataTypeSupported)
            {
                nodeContainsSupportedDataTypes = false;
                return;
            }

            // Otherwise the node supports the tensor data type.
        };

        // Check whether the node uses any data types which are unsupported by the device.
        node.ForEachDef(nodeCallback);

        return nodeContainsSupportedDataTypes;
    }

    bool ExecutionProviderImpl::IsNodeSupportedByDml(
        const onnxruntime::Node& node,
        const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup,
        uint32_t supportedDeviceDataTypeMask // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        ) const
    {
        const onnxruntime::KernelCreateInfo* createInfo = kernel_lookup.LookUpKernel(node);
        if (!createInfo)
        {
            return false;
        }

        auto regInfoIter = m_internalRegInfoMap->find(createInfo->kernel_def.get());
        std::shared_ptr<InternalRegistrationInfo> internalRegInfo;
        if (regInfoIter != m_internalRegInfoMap->end())
        {
            internalRegInfo = regInfoIter->second;
            if (internalRegInfo->supportQuery && !internalRegInfo->supportQuery(node))
            {
                return false;
            }
        }

        // Check whether the node uses any data types which are unsupported by the device.
        if (!DoesNodeContainSupportedDataTypes(node, internalRegInfo.get(), supportedDeviceDataTypeMask, m_native16BitShaderOpsSupported))
        {
            return false;
        }

        return true;
    }

    std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
    ExecutionProviderImpl::GetCapability(
        const onnxruntime::GraphViewer& graph,
        const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup) const
    {
        uint32_t deviceDataTypeMask = GetSupportedDeviceDataTypeMask(); // Each bit corresponds to each DML_TENSOR_DATA_TYPE.

        std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> result;

        // Get the list of node indices in toplogical order, so nodes are visited before
        // downstream nodes consuming them.
        const std::vector<onnxruntime::NodeIndex>& toplogicalOrder = graph.GetNodesInTopologicalOrder();

        std::vector<onnxruntime::NodeIndex> tentativeNodes;
        tentativeNodes.reserve(toplogicalOrder.size());

        for (onnxruntime::NodeIndex nodeIndex : toplogicalOrder)
        {
            const onnxruntime::Node& node = *graph.GetNode(nodeIndex);
            const auto* kernelInfo = kernel_lookup.LookUpKernel(node);
            if (kernelInfo != nullptr)
            {
                tentativeNodes.push_back(nodeIndex);
            }
        }

        // Get the list of nodes that should stay on the CPU
        auto cpuPreferredNodes = GetCpuPreferredNodes(graph, kernel_lookup, tentativeNodes);

        for (size_t nodeIndex : toplogicalOrder)
        {
            const onnxruntime::Node& node = *graph.GetNode(nodeIndex);
            if (IsNodeSupportedByDml(node, kernel_lookup, deviceDataTypeMask)
                && cpuPreferredNodes.find(nodeIndex) == cpuPreferredNodes.end())
            {
                std::unique_ptr<onnxruntime::IndexedSubGraph> subGraph = std::make_unique<onnxruntime::IndexedSubGraph>();
                subGraph->nodes = {nodeIndex};
                result.push_back(std::make_unique<onnxruntime::ComputeCapability>(std::move(subGraph)));
            }
        }
        return result;
    }

    bool IsGpuTensor(const onnxruntime::Tensor& tensor)
    {
        return strcmp(tensor.Location().name, onnxruntime::CPU) &&
            !(tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUOutput || tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUInput);
    }

    Status ExecutionProviderImpl::CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst) const
    {
        assert(!m_closed);

        auto provider = const_cast<ExecutionProviderImpl*>(this);

        TensorWrapper destInternal(
            &dst,
            IsGpuTensor(dst),
            provider,
            true);

        TensorWrapper srcInternal(
            const_cast<onnxruntime::Tensor*>(&src),
            IsGpuTensor(src),
            provider,
            true);

        ORT_THROW_IF_FAILED(CopyTensor(&destInternal, &srcInternal));

        return onnxruntime::common::Status::OK();
    }

    Status ExecutionProviderImpl::CopyTensors(const std::vector<onnxruntime::IDataTransfer::SrcDstPair>& src_dst_pairs) const
    {
        // Source and destination for batched GPU -> CPU copies
        std::vector<ID3D12Resource*> srcDatas;
        std::vector<void*> dstDatas;
        std::vector<uint32_t> dataSizesInBytes;

        assert(!m_closed);
        auto provider = const_cast<ExecutionProviderImpl*>(this);

        for (uint32_t i = 0; i < src_dst_pairs.size(); ++i)
        {
            // This batching implementation only handles GPU -> CPU copies.  Other copies do not require synchronization
            // and are batched across multiple calls to CopyTensor.
            if (!IsGpuTensor(src_dst_pairs[i].src) || IsGpuTensor(src_dst_pairs[i].dst))
            {
                ORT_RETURN_IF_ERROR(CopyTensor(src_dst_pairs[i].src, src_dst_pairs[i].dst));
                continue;
            }

            TensorWrapper srcWrapper = TensorWrapper(
                const_cast<onnxruntime::Tensor*>(&src_dst_pairs[i].src.get()),
                true,
                provider,
                true);

            TensorWrapper dstWrapper = TensorWrapper(
                &src_dst_pairs[i].dst.get(),
                false,
                provider,
                true);

            const size_t dataSizeInBytes = ComputeByteSizeFromTensor(dstWrapper);
            ORT_THROW_HR_IF(E_INVALIDARG, dataSizeInBytes != ComputeByteSizeFromTensor(srcWrapper)); // Tensors must be the same size

            if (dataSizeInBytes == 0)
            {
                return onnxruntime::common::Status::OK();
            }

            dataSizesInBytes.push_back(static_cast<uint32_t>(ComputeByteSizeFromTensor(dstWrapper)));
            ORT_THROW_HR_IF(E_INVALIDARG, dataSizesInBytes[i] != ComputeByteSizeFromTensor(srcWrapper)); // Tensors must be the same size

            dstDatas.push_back(dstWrapper.GetData());
            const AllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(&srcWrapper).GetDataInterface().Get());

            srcDatas.push_back(srcAllocInfo->GetResource());
        }

        const uint64_t srcOffset = 0;
        const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

        // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
        m_readbackHeap->ReadbackFromGpu(dstDatas, dataSizesInBytes, srcDatas, srcState);

        return onnxruntime::common::Status::OK();
    }

    void __stdcall ExecutionProviderImpl::Flush() const
    {
        assert(!m_closed);
        m_context->Flush();
    }

    void ExecutionProviderImpl::ReleaseCompletedReferences()
    {
         m_context->ReleaseCompletedReferences();
    }

    void ExecutionProviderImpl::QueueReference(IUnknown* object)
    {
        assert(!m_closed);
        m_context->QueueReference(object);
    }

    void ExecutionProviderImpl::GetShadowCopyIfRequired(
        bool isInternalOperator,
        IUnknown* data,
        IUnknown** dataCopy) const
    {
        assert(!m_closed);

        *dataCopy = data;
        data->AddRef();
    }

    void ExecutionProviderImpl::GetABIDataInterface(
        bool isInternalOperator,
        IUnknown* data,
        IUnknown** abiData) const
    {
        assert(!m_closed);

        if (isInternalOperator)
        {
            *abiData = data;
            data->AddRef();
        }
        else
        {
#ifdef _GAMING_XBOX
            ComPtr<GraphicsUnknownWrapper> wrappedResource = Microsoft::WRL::Make<GraphicsUnknownWrapper>(m_allocator->DecodeDataHandle(data)->GetResource());
            *abiData = wrappedResource.Detach();
#else
            ComPtr<ID3D12Resource> resource = m_allocator->DecodeDataHandle(data)->GetResource();
            *abiData = resource.Detach();
#endif
        }
    }

    uint64_t ExecutionProviderImpl::TryGetPooledAllocationId(
        IUnknown* data,
        bool isInternalOperator)
    {
        assert(!isInternalOperator);
        return m_allocator->DecodeDataHandle(data)->GetPooledResourceId();
    }

    void ExecutionProviderImpl::GetABIExecutionInterfaceAndInvalidateState(
        bool isInternalOperator,
        IUnknown** abiExecutionObject) const
    {
        assert(!m_closed);

        if (isInternalOperator)
        {
            ComPtr<IUnknown> thisPtr = const_cast<IExecutionProvider*>(static_cast<const IExecutionProvider*>(this));
            *abiExecutionObject = thisPtr.Detach();
        }
        else
        {
            ComPtr<ID3D12GraphicsCommandList> commandList;
            m_context->GetCommandListForRecordingAndInvalidateState(commandList.GetAddressOf());
#ifdef _GAMING_XBOX
            ComPtr<GraphicsUnknownWrapper> wrappedCommandList = Microsoft::WRL::Make<GraphicsUnknownWrapper>(commandList.Get());
            *abiExecutionObject = wrappedCommandList.Detach();
#else
            *abiExecutionObject = commandList.Detach();
#endif
        }
    }

    bool ExecutionProviderImpl::TransitionsRequiredForOperator(
        bool isInternalOperator
    )
    {
        // External operators receive resources in Common state, while internal operators receive
        // them in UAV state. Resources are otherwise kept in UAV state (or are promotable to UAV).
        return !isInternalOperator;
    }

    void ExecutionProviderImpl::TransitionResourcesForOperator(
        bool isBeforeOp,
        uint32_t resourceCount,
        IUnknown** resources
    )
    {
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        barriers.reserve(resourceCount);

        for (uint32_t i = 0; i < resourceCount; ++i)
        {
            ComPtr<ID3D12Resource> resource;
            ORT_THROW_IF_FAILED(resources[i]->QueryInterface(resource.GetAddressOf()));

            // Custom operators receive resources in Common state and must return them to Common
            // state when finished.  Resources are otherwise kept in UAV state (or are promotable to UAV).
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
                resource.Get(),
                isBeforeOp ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON,
                isBeforeOp ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            ));
        }

        if (!barriers.empty())
        {
            m_context->ResourceBarrier(barriers);
        }
    }

    D3D12_COMMAND_LIST_TYPE __stdcall ExecutionProviderImpl::GetCommandListTypeForQueue() const
    {
        return m_context->GetCommandListTypeForQueue();
    }

    bool __stdcall ExecutionProviderImpl::IsMcdmDevice() const noexcept
    {
        return m_isMcdmDevice;
    }

    bool __stdcall ExecutionProviderImpl::CustomHeapsSupported() const noexcept
    {
        return m_areCustomHeapsSupported;
    }

    bool __stdcall ExecutionProviderImpl::MetacommandsEnabled() const noexcept
    {
        return m_areMetacommandsEnabled;
    }

    std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap>
    ExecutionProviderImpl::GetInternalRegistrationInfoMap() const
    {
        return m_internalRegInfoMap;
    }

    std::shared_ptr<onnxruntime::IAllocator> ExecutionProviderImpl::GetGpuAllocator()
    {
        return m_allocator;
    }

    std::shared_ptr<onnxruntime::IAllocator> ExecutionProviderImpl::GetCpuInputAllocator()
    {
        return m_cpuInputAllocator;
    }

    onnxruntime::common::Status ExecutionProviderImpl::OnSessionInitializationEnd()
    {
        // Flush and trim resources, including staging memory used to upload weights.
        // This reduces memory usage immediately after session creation, and avoids
        // performance impact of deallocation during first evaluation.
        Flush();
        m_context->GetCurrentCompletionEvent().WaitForSignal();
        m_context->ReleaseCompletedReferences();
        m_uploadHeap->Trim();

        // Allocations after this point are potentially transient and their sizes are
        // rounded to enable pooling.
        m_allocator->SetDefaultRoundingMode(AllocatorRoundingMode::Enabled);

        return onnxruntime::common::Status::OK();
    }

    std::unique_ptr<onnxruntime::IExecutionProvider> CreateExecutionProvider(
        IDMLDevice* dmlDevice,
        ID3D12CommandQueue* commandQueue,
        bool enableMetacommands)
    {
        return std::make_unique<Dml::ExecutionProvider>(dmlDevice, commandQueue, enableMetacommands);
    }

    ID3D12Resource* GetD3D12ResourceFromAllocation(onnxruntime::IAllocator* allocator, void* ptr)
    {
        Dml::BucketizedBufferAllocator* pAllocationInfo = static_cast<Dml::BucketizedBufferAllocator*>(allocator);
        return pAllocationInfo->DecodeDataHandle(ptr)->GetResource();
    }

    void FlushContext(onnxruntime::IExecutionProvider* provider)
    {
        ExecutionProvider* dmlexecutionprovider = static_cast<Dml::ExecutionProvider*>(provider);
        dmlexecutionprovider->Flush();
    }

    void ReleaseCompletedReferences(onnxruntime::IExecutionProvider * provider)
    {
        ExecutionProvider* dmlexecutionprovider = static_cast<Dml::ExecutionProvider*>(provider);
        dmlexecutionprovider->ReleaseCompletedReferences();
    }

    onnxruntime::common::Status CopyTensor(
        onnxruntime::IExecutionProvider* provider,
        const onnxruntime::Tensor& src,
        onnxruntime::Tensor& dst
    )
    {
        ExecutionProvider* dmlexecutionprovider = static_cast<Dml::ExecutionProvider*>(provider);
        return dmlexecutionprovider->GetImpl()->CopyTensor(src, dst);
    }

    void* CreateGPUAllocationFromD3DResource(ID3D12Resource* pResource)
    {
        uint64_t pooledResourceId = 0; // Not a pooled resource

        ComPtr<DmlResourceWrapper> resourceWrapper;
        wil::MakeOrThrow<DmlCommittedResourceWrapper>(pResource).As(&resourceWrapper);

        ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(nullptr, 0, pooledResourceId, resourceWrapper.Get(), (size_t)pResource->GetDesc().Width);
        return allocInfo.Detach();
    }
    void FreeGPUAllocation(void* ptr)
    {
        ComPtr<AllocationInfo> allocInfo;
        allocInfo.Attach(static_cast<AllocationInfo*>(ptr));
    }

} // namespace Dml
