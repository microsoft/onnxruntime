// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    // Represents a fence which will be signaled at some point (usually by the GPU).
    struct GpuEvent
    {
        uint64_t fenceValue;
        ComPtr<ID3D12Fence> fence;

        bool IsSignaled() const
        {
            return fence->GetCompletedValue() >= fenceValue;
        }

        // Blocks until IsSignaled returns true.
        void WaitForSignal() const
        {
            if (IsSignaled())
                return; // early-out

            wil::unique_handle h(CreateEvent(nullptr, TRUE, FALSE, nullptr));
            ORT_THROW_LAST_ERROR_IF(!h);

            ORT_THROW_IF_FAILED(fence->SetEventOnCompletion(fenceValue, h.get()));

            WaitForSingleObject(h.get(), INFINITE);
        }
    };

} // namespace Dml
