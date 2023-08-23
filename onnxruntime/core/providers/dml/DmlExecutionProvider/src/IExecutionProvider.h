// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <d3d12.h>

#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"

interface IDMLCompiledOperator;
struct DML_BUFFER_BINDING;
struct DML_BINDING_DESC;

namespace Dml
{
    struct Binding
    {
        // Non-null if required at the stage where it is used, i.e. Initialization
        IMLOperatorTensor* tensor;

        UINT64 sizeInBytes;
    };

    // DML specific interface into the execution provider, which avoids any dependencies with
    // internal Lotus data types.
    interface __declspec(uuid("b2488edb-fad2-4704-a6d2-5b5b129d4b8e"))
    IExecutionProvider : public IUnknown
    {
    public:
        STDMETHOD(GetD3DDevice)(_COM_Outptr_ ID3D12Device** d3dDevice) const noexcept = 0;

        STDMETHOD(GetDmlDevice)(_COM_Outptr_ IDMLDevice** dmlDevice) const noexcept = 0;

        STDMETHOD(ExecuteCommandList)(
            ID3D12GraphicsCommandList* commandList,
            _Outptr_ ID3D12Fence** fence,
            _Out_ uint64_t* completionValue
            ) const noexcept = 0;

        STDMETHOD(AddUAVBarrier)() const noexcept = 0;

        STDMETHOD(InitializeOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<const DML_BUFFER_BINDING> inputTensors
            ) const noexcept = 0;

        STDMETHOD(ExecuteOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<IMLOperatorTensor*> inputTensors,
            gsl::span<IMLOperatorTensor*> outputTensors
            ) const noexcept = 0;

        STDMETHOD(ExecuteOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<DML_BINDING_DESC> inputTensors,
            gsl::span<DML_BINDING_DESC> outputTensors
            ) const noexcept = 0;

        STDMETHOD(CopyTensor)(IMLOperatorTensor* dst, IMLOperatorTensor* src) const noexcept = 0;
        STDMETHOD(CopyTensors)(gsl::span<IMLOperatorTensor*> dst, gsl::span<IMLOperatorTensor*> src) const noexcept = 0;

        STDMETHOD(FillTensorWithPattern)(
            IMLOperatorTensor* dst,
            gsl::span<const std::byte> value
            ) const noexcept = 0;

        STDMETHOD(UploadToResource)(ID3D12Resource* dstData, const void* srcData, uint64_t srcDataSize) const noexcept = 0;

        STDMETHOD_(D3D12_COMMAND_LIST_TYPE, GetCommandListTypeForQueue)() const noexcept = 0;
        STDMETHOD_(void, Flush)() const noexcept = 0;

        STDMETHOD_(ID3D12Resource*, DecodeResource)(void* allocation) const noexcept = 0;
        STDMETHOD(AllocatePooledResource(size_t size, AllocatorRoundingMode roundingMode, ID3D12Resource **d3dResource, IUnknown* *pooledResource)) const noexcept = 0;

        STDMETHOD_(bool, IsMcdmDevice)() const noexcept = 0;
        STDMETHOD_(bool, CustomHeapsSupported)() const noexcept = 0;
        STDMETHOD_(bool, MetacommandsEnabled)() const noexcept = 0;
    };
} // namespace Dml
