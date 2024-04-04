// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "vaip/custom_op.h"
namespace vaip {
using namespace ::onnxruntime;
std::unique_ptr<ComputeCapability> XirSubgraphToComputeCapability1(const onnxruntime::GraphViewer& graph, vaip_core::ExecutionProvider* ep, size_t index);
std::vector<std::unique_ptr<ComputeCapability>> GetComputeCapabilityOps(const onnxruntime::GraphViewer& graph,
                                                                        vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>* ep, const std::set<std::string>& all_not_support_optypes);

}  // namespace vaip
