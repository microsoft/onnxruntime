// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/capability_planning_discovery.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <utility>

#include "core/common/common.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace {

bool SourceViewContainsNode(const GraphViewer& graph, size_t node_index) {
  return node_index < static_cast<size_t>(graph.MaxNodeIndex()) && graph.GetNode(node_index) != nullptr;
}

Status ValidateSourceView(const GraphViewer& graph, const CapabilityPlanningQuery& query) {
  if (graph.GetFilterInfo() != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning requires the unfiltered source graph view.");
  }
  for (const auto& constraint : query.node_constraints) {
    if (!SourceViewContainsNode(graph, constraint.node_index)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Constraint refers to a node outside the source view.");
    }
  }
  for (const auto& assignment : query.tentative_assignments) {
    if (!SourceViewContainsNode(graph, assignment.node_index)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Tentative assignment refers to a node outside the source view.");
    }
  }
  return Status::OK();
}

Status NormalizePolicies(InlinedVector<KernelDispatchPolicy>& policies) {
  for (const auto policy : policies) {
    switch (policy) {
      case KernelDispatchPolicy::Auto:
      case KernelDispatchPolicy::Latency:
      case KernelDispatchPolicy::Memory:
      case KernelDispatchPolicy::Safe:
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown planning policy intent.");
    }
  }
  std::sort(policies.begin(), policies.end());
  policies.erase(std::unique(policies.begin(), policies.end()), policies.end());
  return Status::OK();
}

}  // namespace

Status NormalizeCapabilityPlanningQuery(
    const GraphViewer& graph_viewer,
    CapabilityPlanningQuery request,
    std::shared_ptr<const CapabilityPlanningQuery>& query) {
  std::sort(request.node_constraints.begin(), request.node_constraints.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.node_index < rhs.node_index; });
  InlinedVector<CapabilityNodeConstraint> constraints;
  constraints.reserve(request.node_constraints.size());
  for (auto& constraint : request.node_constraints) {
    if (constraint.permitted_policies.has_value()) {
      if (auto status = NormalizePolicies(*constraint.permitted_policies); !status.IsOK()) {
        return status;
      }
    }
    if (constraints.empty() || constraints.back().node_index != constraint.node_index) {
      constraints.push_back(std::move(constraint));
      continue;
    }
    auto& merged = constraints.back();
    if (constraint.required_ep_instance_id.has_value()) {
      if (merged.required_ep_instance_id.has_value() &&
          merged.required_ep_instance_id != constraint.required_ep_instance_id) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Conflicting required EP assignments.");
      }
      merged.required_ep_instance_id = std::move(constraint.required_ep_instance_id);
    }
    if (constraint.permitted_policies.has_value()) {
      if (merged.permitted_policies.has_value()) {
        InlinedVector<KernelDispatchPolicy> intersection;
        std::set_intersection(merged.permitted_policies->begin(), merged.permitted_policies->end(),
                              constraint.permitted_policies->begin(), constraint.permitted_policies->end(),
                              std::back_inserter(intersection));
        merged.permitted_policies = std::move(intersection);
      } else {
        merged.permitted_policies = std::move(constraint.permitted_policies);
      }
    }
  }
  request.node_constraints = std::move(constraints);

  std::sort(request.tentative_assignments.begin(), request.tentative_assignments.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.node_index < rhs.node_index; });
  InlinedVector<CapabilityNodeAssignment> assignments;
  assignments.reserve(request.tentative_assignments.size());
  for (auto& assignment : request.tentative_assignments) {
    if (!assignments.empty() && assignments.back().node_index == assignment.node_index) {
      if (assignments.back().ep_instance_id != assignment.ep_instance_id) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Conflicting tentative EP assignments.");
      }
      continue;
    }
    assignments.push_back(std::move(assignment));
  }
  request.tentative_assignments = std::move(assignments);

  const auto snapshot = std::make_shared<const CapabilityPlanningQuery>(std::move(request));
  if (auto status = ValidateCapabilityPlanningQuery(snapshot); !status.IsOK()) {
    return status;
  }
  if (auto status = ValidateSourceView(graph_viewer, *snapshot); !status.IsOK()) {
    return status;
  }
  query = snapshot;
  return Status::OK();
}

Status QueryCapabilityPlanning(
    const IExecutionProvider& ep,
    const GraphViewer& graph_viewer,
    const IExecutionProvider::IKernelLookup& kernel_lookup,
    const GraphOptimizerRegistry& graph_optimizer_registry,
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    CapabilityPlanningQueryResult& result) {
  result = CapabilityPlanningQueryResult{};
  if (auto status = ValidateCapabilityPlanningQuery(query); !status.IsOK()) {
    return status;
  }
  if (auto status = ValidateSourceView(graph_viewer, *query); !status.IsOK()) {
    return status;
  }
  if (*query->execution_device != ep.GetDevice()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning query device does not match the EP instance.");
  }

  CapabilityPlanningQueryResult discovered;
  Status status;
  ORT_TRY {
    auto capabilities = ep.GetCapability(graph_viewer, kernel_lookup, graph_optimizer_registry, query);
    discovered.capabilities.assign(std::make_move_iterator(capabilities.begin()),
                                   std::make_move_iterator(capabilities.end()));
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() { status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what()); });
  }
  if (!status.IsOK()) {
    return status;
  }
  if (discovered.capabilities.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "EP returned no planning-safe offers.");
  }
  discovered.support = CapabilityPlanningSupport::PlanningSafe;
  if (auto validation = ValidateCapabilityPlanningResult(query, discovered); !validation.IsOK()) {
    return validation;
  }
  for (const auto& capability : discovered.capabilities) {
    for (const auto node_index : capability->sub_graph->nodes) {
      if (!SourceViewContainsNode(graph_viewer, node_index)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning offer contains a node outside the source view.");
      }
    }
  }
  result = std::move(discovered);
  return Status::OK();
}

}  // namespace onnxruntime
