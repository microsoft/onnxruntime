// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/trt/trt_kernel.h"

namespace onnxruntime
{

//Information needed to construct trt execution providers.
struct TRTExecutionProviderInfo
{
    int device_id{0};
};

// Logical device representation.
class TRTExecutionProvider : public IExecutionProvider
{
public:
    TRTExecutionProvider();
    virtual ~TRTExecutionProvider();

    std::vector<std::unique_ptr<ComputeCapability>>
            GetCapability(const onnxruntime::GraphViewer& graph,
                          const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;


    std::string Type() const override
    {
        return onnxruntime::kTRTExecutionProvider;
    }

    Status CopyTensor(const Tensor& src, Tensor& dst) const override;

    const void* GetExecutionHandle() const noexcept override
    {
        return nullptr;
    }

    virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

private:
    int device_id_;
};

}  // namespace onnxruntime

