/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once

#include "hailo/hailort.hpp"
#include "core/framework/execution_provider.h"
#include "hailo_op_manager.h"

namespace onnxruntime {

using hailort::VDevice;

struct HailoExecutionProviderInfo {
    bool create_arena = true;

    explicit HailoExecutionProviderInfo(bool use_arena) : create_arena(use_arena) {}
    HailoExecutionProviderInfo() = default;
};

// Logical device representation.
class HailoExecutionProvider : public IExecutionProvider {
public:
    explicit HailoExecutionProvider(const HailoExecutionProviderInfo& info);
    virtual ~HailoExecutionProvider();

    std::vector<std::unique_ptr<ComputeCapability>>
    GetCapability(const onnxruntime::GraphViewer& graph,
        const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

    virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

private:
    InlinedVector<NodeIndex> GetSupportedNodes(const GraphViewer& graph_viewer) const;
    HailoOpManager m_op_manager;
};

}  // namespace onnxruntime