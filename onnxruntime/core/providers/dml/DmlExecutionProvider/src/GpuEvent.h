// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <wil/wrl.h>
#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"
#include "core/common/spin_pause.h"

namespace Dml
{
    // Represents a fence which will be signaled at some point (usually by the GPU).
    struct GpuEvent
    {
        uint64_t fenceValue;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;

        bool IsSignaled() const
        {
            return fence->GetCompletedValue() >= fenceValue;
        }

        // Blocks until IsSignaled returns true.
        void WaitForSignal(bool cpuSyncSpinningEnabled) const
        {
            if (IsSignaled())
                return; // early-out

            if (cpuSyncSpinningEnabled)
            {
                while (!IsSignaled())
                {
                    // We keep spinning until the fence gets signaled
                    onnxruntime::concurrency::SpinPause();
                }
            }
            else
            {
                wil::unique_handle h(CreateEvent(nullptr, TRUE, FALSE, nullptr));
                ORT_THROW_LAST_ERROR_IF(!h);
                ORT_THROW_IF_FAILED(fence->SetEventOnCompletion(fenceValue, h.get()));
                WaitForSingleObject(h.get(), INFINITE);
            }
        }
    };

} // namespace Dml
