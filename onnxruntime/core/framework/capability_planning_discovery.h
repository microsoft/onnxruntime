// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/capability_planning.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

// The caller supplies source identity/revision and effective workload/configuration facts.
// Repeated node constraints are intersected. Failure leaves query unchanged.
Status NormalizeCapabilityPlanningQuery(
    const GraphViewer& graph_viewer,
    CapabilityPlanningQuery request,
    std::shared_ptr<const CapabilityPlanningQuery>& query);

// Queries the unfiltered source view without applying placement or committing resources.
// Failure clears result; diagnostics are returned in Status.
Status QueryCapabilityPlanning(
    const IExecutionProvider& ep,
    const GraphViewer& graph_viewer,
    const IExecutionProvider::IKernelLookup& kernel_lookup,
    const GraphOptimizerRegistry& graph_optimizer_registry,
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    CapabilityPlanningQueryResult& result);

}  // namespace onnxruntime
