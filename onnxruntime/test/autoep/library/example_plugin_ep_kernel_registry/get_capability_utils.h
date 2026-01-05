// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>
#include <unordered_set>
#include "../plugin_ep_utils.h"

// Returns nodes that should be assigned to CPU EP instead of this example EP to avoid costly I/O copies.
// Based on GetCpuPreferredNodes from onnxruntime/core/framework/fallback_cpu_capability.cc
OrtStatus* GetCpuPreferredNodes(const OrtGraph& ort_graph, OrtEpGraphSupportInfo& graph_support_info,
                                const OrtLogger& logger, gsl::span<const OrtNode* const> tentative_nodes,
                                /*out*/ std::unordered_set<const OrtNode*>& cpu_preferred_nodes);
