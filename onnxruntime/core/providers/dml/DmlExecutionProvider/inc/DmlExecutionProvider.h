// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
interface IMLOperatorRegistry;

#include "core/common/status.h"
#include "core/framework/data_transfer.h"
#include "IWinmlExecutionProvider.h"

namespace onnxruntime
{
    class IExecutionProvider;
    class IAllocator;
    class CustomRegistry;
    class InferenceSession;
    class KernelRegistry;
}

enum class AllocatorRoundingMode
{
    Disabled = 0,
    Enabled = 1,
};

namespace Dml
{
    constexpr GUID ExecutionContextGUID = {0x50fd773b, 0x4462, 0x4b28, {0x98, 0x9e, 0x8c, 0xa0, 0x54, 0x05, 0xbd, 0x4a}};
    constexpr GUID UploadHeapGUID = {0x125235f9, 0xef41, 0x4043, {0xa4, 0x9d, 0xdd, 0xc9, 0x61, 0xe7, 0xdb, 0xee}};
    constexpr GUID ReadbackHeapGUID = {0x00d32df8, 0xea2d, 0x40bf, {0xa4, 0x47, 0x9c, 0xb4, 0xbc, 0xf1, 0x1d, 0x5e}};
    constexpr GUID AllocatorGUID = {0xe9fe9103, 0x3503, 0x42c7, {0xa0, 0x69, 0x9f, 0x03, 0xc1, 0x1d, 0x5e, 0xdd}};

    std::unique_ptr<onnxruntime::IExecutionProvider> CreateExecutionProvider(
        IDMLDevice* dmlDevice,
        ID3D12CommandQueue* commandQueue,
        bool enableMetacommands = true);

    ID3D12Resource* GetD3D12ResourceFromAllocation(onnxruntime::IAllocator* allocator, void* ptr);
    void FlushContext(onnxruntime::IExecutionProvider* provider);
    void ReleaseCompletedReferences(onnxruntime::IExecutionProvider* provider);

    onnxruntime::common::Status CopyTensor(
        onnxruntime::IExecutionProvider* provider,
        const onnxruntime::Tensor& src, onnxruntime::Tensor& dst
    );

    void* CreateGPUAllocationFromD3DResource(ID3D12Resource* pResource);
    void FreeGPUAllocation(void* ptr);

    void RegisterDmlOperators(IMLOperatorRegistry* registry);
    void RegisterCpuOperatorsAsDml(onnxruntime::KernelRegistry* registry);

} // namespace Dml
