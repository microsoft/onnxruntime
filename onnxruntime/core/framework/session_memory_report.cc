// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_memory_report.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>

#include "core/common/safeint.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace {

OrtDevice MemoryLocation(const OrtDevice& device) {
  return OrtDevice(device.Type(), device.MemType(), device.Vendor(), device.Id());
}

void AddBytes(size_t& destination, size_t bytes) {
  destination = SafeInt<size_t>(destination) + bytes;
}

Status CollectInitializers(const SessionState& session_state, bool before_kernel_creation,
                           SessionMemoryReport& report) {
  InlinedHashSet<const Tensor*> seen;
  for (const auto& [index, value] : session_state.GetInitializedTensors()) {
    if (!value.IsTensor()) {
      report.uncovered_terms.push_back(MakeString("Non-tensor initializer at value index ", index));
      continue;
    }

    const Tensor& tensor = value.Get<Tensor>();
    // Copies of one OrtValue share the Tensor, not just its name or shape.
    if (!seen.insert(&tensor).second) {
      continue;
    }
    if (tensor.IsDataTypeString()) {
      report.uncovered_terms.push_back(MakeString("String initializer storage at value index ", index));
      continue;
    }

    auto& device = report.devices[MemoryLocation(tensor.Location().device)];
    AddBytes(before_kernel_creation ? device.initializers_before_kernel_creation_bytes
                                    : device.initializers_after_prepack_bytes,
             tensor.SizeInBytes());
  }
  return Status::OK();
}

Status ResolveRoot(const SequentialExecutionPlan& plan, size_t index, size_t& root) {
  root = index;
  for (size_t followed = 0; followed <= plan.allocation_plan.size(); ++followed) {
    ORT_RETURN_IF(root >= plan.allocation_plan.size(), "Invalid memory-report allocation root: ", root);
    const auto& allocation = plan.allocation_plan[root];
    if (allocation.alloc_kind != AllocKind::kReuse && allocation.alloc_kind != AllocKind::kShare) {
      return Status::OK();
    }
    ORT_RETURN_IF(allocation.reused_buffer < 0, "Negative memory-report allocation root");
    root = static_cast<size_t>(allocation.reused_buffer);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cycle in memory-report allocation aliases");
}

Status StaticTensorBytes(const NodeArg* arg, MLDataType value_type, std::optional<size_t>& bytes) {
  if (arg == nullptr || value_type == nullptr || !value_type->IsTensorType() || arg->Shape() == nullptr) {
    return Status::OK();
  }
  const auto* element_type = static_cast<const TensorTypeBase*>(value_type)->GetElementType();
  if (element_type == DataTypeImpl::GetType<std::string>()) {
    return Status::OK();
  }
  TensorShapeVector dimensions;
  for (const auto& dimension : arg->Shape()->dim()) {
    if (!dimension.has_dim_value() || dimension.dim_value() < 0) {
      return Status::OK();
    }
    dimensions.push_back(dimension.dim_value());
  }
  size_t storage_bytes = 0;
  ORT_RETURN_IF_ERROR(Tensor::CalculateTensorStorageSize(
      element_type, TensorShape(dimensions), /*alignment=*/0, storage_bytes));
  bytes = storage_bytes;
  return Status::OK();
}

void SummarizeLifetimes(SessionMemoryReport& report) {
  struct Event {
    size_t allocate = 0;
    size_t release = 0;
  };
  std::map<OrtDevice, std::map<size_t, Event>> events;
  for (const auto& allocation : report.allocations) {
    if (!allocation.size_bytes.has_value()) {
      continue;
    }
    auto& device_events = events[allocation.location];
    AddBytes(device_events[allocation.first_step].allocate, *allocation.size_bytes);
    const size_t release_step = SafeInt<size_t>(allocation.last_step) + 1;
    AddBytes(device_events[release_step].release, *allocation.size_bytes);
  }
  for (const auto& [location, device_events] : events) {
    SafeInt<size_t> live_bytes = 0;
    auto& peak = report.devices[location].modeled_run_live_peak_bytes;
    for (const auto& [step, event] : device_events) {
      ORT_UNUSED_PARAMETER(step);
      live_bytes -= event.release;
      live_bytes += event.allocate;
      peak = std::max(peak, static_cast<size_t>(live_bytes));
    }
  }
}

std::string BytesOrUnavailable(const std::optional<size_t>& bytes) {
  return bytes.has_value() ? std::to_string(*bytes) : "unavailable";
}

}  // namespace

Status CaptureSessionMemoryInitializers(const SessionState& session_state, SessionMemoryReport& report) {
  return CollectInitializers(session_state, /*before_kernel_creation=*/true, report);
}

Status CompleteSessionMemoryReport(
    const SessionState& session_state, const MemoryReportWorkspaceMap& workspace,
    const NodeWorkspaceReservationMap* reservations,
    SessionMemoryReport& report) {
  const auto* plan = session_state.GetExecutionPlan();
  ORT_RETURN_IF(plan == nullptr, "A session memory report requires an execution plan");
  const auto& graph = session_state.GetGraphViewer();
  const auto& options = session_state.GetSessionOptions();
  const auto& initializers = session_state.GetInitializedTensors();
  ORT_RETURN_IF_ERROR(CollectInitializers(session_state, /*before_kernel_creation=*/false, report));

  const auto stream_count = std::count_if(
      plan->execution_plan.begin(), plan->execution_plan.end(),
      [](const auto& stream) { return stream != nullptr && !stream->steps_.empty(); });
  report.ordered_lifetimes = options.execution_mode == ExecutionMode::ORT_SEQUENTIAL && stream_count <= 1;
#ifdef ENABLE_TRAINING
  report.ordered_lifetimes = false;
  report.uncovered_terms.push_back("Training lifetimes and retained training state are not modeled");
#endif
  if (!report.ordered_lifetimes) {
    report.uncovered_terms.push_back(
        "Parallel/multi-stream execution: modeled roots conservatively overlap for the entire Run; "
        "host launch order is not GPU completion");
  }

  InlinedHashMap<size_t, size_t> node_steps;
  if (report.ordered_lifetimes) {
    for (const auto& stream : plan->execution_plan) {
      if (stream == nullptr) {
        continue;
      }
      for (const auto& step : stream->steps_) {
        const auto index = step->GetNodeIndex();
        if (graph.GetNode(index) != nullptr && node_steps.count(index) == 0) {
          node_steps.emplace(index, node_steps.size());
        }
      }
    }
  }
  const size_t end_step = graph.NumberOfNodes();
  InlinedHashMap<size_t, size_t> births;
  InlinedHashSet<size_t> retained;
  const auto& names = session_state.GetOrtValueNameIdxMap();
  for (const auto* output : graph.GetOutputs()) {
    int index;
    ORT_RETURN_IF_ERROR(names.GetIdx(output->Name(), index));
    retained.insert(static_cast<size_t>(index));
  }
  for (const auto& node : graph.Nodes()) {
    const auto step = node_steps.find(node.Index());
    if (report.ordered_lifetimes && step == node_steps.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Memory report cannot locate node in execution plan: ", node.Name());
    }
    for (const auto* output : node.OutputDefs()) {
      if (!output->Exists()) {
        continue;
      }
      int index;
      ORT_RETURN_IF_ERROR(names.GetIdx(output->Name(), index));
      births.emplace(static_cast<size_t>(index), report.ordered_lifetimes ? step->second : 0);
    }
    if (!node.GetAttributeNameToSubgraphMap().empty()) {
      report.uncovered_terms.push_back(
          MakeString("Control-flow/subgraph backing and loop-carried or accumulated values at node '", node.Name(), "'"));
    }
  }

  InlinedVector<size_t> roots;
  roots.reserve(plan->allocation_plan.size());
  for (size_t index = 0; index < plan->allocation_plan.size(); ++index) {
    size_t root;
    ORT_RETURN_IF_ERROR(ResolveRoot(*plan, index, root));
    roots.push_back(root);
  }

  InlinedHashMap<const Tensor*, size_t> initializer_owners;
  InlinedHashMap<size_t, size_t> initializer_aliases;
  for (const auto& [index, value] : initializers) {
    ORT_RETURN_IF(index < 0 || static_cast<size_t>(index) >= roots.size(), "Invalid initializer index in memory report");
    if (value.IsTensor()) {
      const auto root = roots[static_cast<size_t>(index)];
      const auto [it, inserted] = initializer_owners.emplace(&value.Get<Tensor>(), root);
      if (!inserted) {
        initializer_aliases.emplace(root, it->second);
      }
    }
  }
  for (auto& root : roots) {
    const auto alias = initializer_aliases.find(root);
    if (alias != initializer_aliases.end()) {
      root = alias->second;
    }
  }

  struct Root {
    SessionMemoryAllocation allocation;
    bool unknown_size = false;
    bool retained = false;
  };
  std::map<size_t, Root> live_roots;
  for (size_t index = 0; index < roots.size(); ++index) {
    const auto& allocation = plan->allocation_plan[index];
    if (allocation.alloc_kind == AllocKind::kNotSet) {
      continue;
    }
    const auto root_index = roots[index];
    const auto& root_plan = plan->allocation_plan[root_index];
    const auto initializer = initializers.find(static_cast<int>(root_index));
    if (root_plan.alloc_kind == AllocKind::kAllocateStatically && initializer == initializers.end()) {
      continue;  // PrePack released this source; packed destinations have separate, unresolved ownership.
    }

    std::string name;
    ORT_RETURN_IF_ERROR(names.GetName(static_cast<int>(index), name));
    const auto birth = births.find(index);
    const size_t first_step = birth == births.end() ? 0 : birth->second;
    std::optional<size_t> bytes;
    OrtDevice location = MemoryLocation(root_plan.location);
    if (initializer != initializers.end() && initializer->second.IsTensor()) {
      const auto& tensor = initializer->second.Get<Tensor>();
      location = MemoryLocation(tensor.Location().device);
      if (!tensor.IsDataTypeString()) {
        bytes = tensor.SizeInBytes();
      }
    } else {
      ORT_RETURN_IF_ERROR(StaticTensorBytes(graph.GetNodeArg(name), allocation.value_type, bytes));
    }

    auto [it, inserted] = live_roots.try_emplace(root_index);
    auto& root = it->second;
    if (inserted) {
      root.allocation = {name, location, bytes, first_step, end_step};
    } else {
      root.allocation.first_step = std::min(root.allocation.first_step, first_step);
      if (bytes.has_value()) {
        root.allocation.size_bytes = std::max(root.allocation.size_bytes.value_or(0), *bytes);
      }
    }
    root.retained = root.retained || retained.count(index) > 0 ||
                    root_plan.alloc_kind == AllocKind::kPreExisting ||
                    root_plan.alloc_kind == AllocKind::kAllocateStatically;
    if (!bytes.has_value()) {
      root.unknown_size = true;
      report.uncovered_terms.push_back(MakeString("Unavailable tensor storage for value '", name, "'"));
    }
  }

  if (report.ordered_lifetimes) {
    for (const auto& [node_index, step] : node_steps) {
      if (node_index >= plan->node_release_list.size()) {
        continue;
      }
      for (const auto action_index : plan->node_release_list[node_index]) {
        ORT_RETURN_IF(action_index >= plan->release_actions.size(), "Invalid memory-report release action");
        const auto& action = plan->release_actions[action_index];
        ORT_RETURN_IF(action.value_index >= roots.size(), "Invalid memory-report release value");
        auto root = live_roots.find(roots[action.value_index]);
        if (action.ref_count == 1 && root != live_roots.end() && !root->second.retained) {
          root->second.allocation.last_step = step;
        }
      }
    }
  }
  for (auto& [index, root] : live_roots) {
    ORT_UNUSED_PARAMETER(index);
    if (root.unknown_size) {
      root.allocation.size_bytes.reset();
    }
    report.devices.try_emplace(root.allocation.location);
    report.allocations.push_back(std::move(root.allocation));
  }

  const bool parallel_prepack =
      options.config_options.GetConfigOrDefault(kOrtSessionOptionsEnableParallelPrepack, "0") == "1";
  for (const auto& node : graph.Nodes()) {
    const auto* kernel = session_state.GetKernel(node.Index());
    if (kernel == nullptr) {
      report.uncovered_terms.push_back(MakeString("Unavailable kernel at node '", node.Name(), "'"));
      continue;
    }
    const auto location = MemoryLocation(kernel->Info().GetDevice(OrtMemTypeDefault));
    auto& device = report.devices[location];
    if (reservations != nullptr) {
      const auto reservation = reservations->find(node.Index());
      if (reservation != reservations->end()) {
        AddBytes(device.l1_persistent_prepack_estimate_bytes, reservation->second.persistent_prepack_bytes);
        if (parallel_prepack) {
          AddBytes(device.l1_initialization_scratch_estimate_bytes, reservation->second.initialization_scratch_bytes);
        } else {
          device.l1_initialization_scratch_estimate_bytes =
              std::max(device.l1_initialization_scratch_estimate_bytes,
                       reservation->second.initialization_scratch_bytes);
        }
      }
    }

    const auto requirements = workspace.find(node.Index());
    if (requirements == workspace.end() || requirements->second.empty()) {
      report.uncovered_terms.push_back(MakeString("Unavailable workspace declaration for node '", node.Name(), "'"));
      continue;
    }
    const auto step = report.ordered_lifetimes ? node_steps.at(node.Index()) : 0;
    for (const auto& requirement : requirements->second) {
      report.allocations.push_back({MakeString("workspace:", node.Name(), ":", requirement.slot_id), location, requirement.size_bytes,
                                    step, report.ordered_lifetimes ? step : end_step});
    }
  }
  SummarizeLifetimes(report);
  report.uncovered_terms.push_back(
      "Physical initialization/Run peaks: allocator growth, padding, fragmentation, retained pools, and reserve");
  report.uncovered_terms.push_back(
      "Packed-weight allocation identities/sharing, kernel-constructor scratch, and runtime preparation");
  report.uncovered_terms.push_back(
      "Caller-owned input/KV backing, actual I/O locations/aliases/capacities, and output retention across Runs");
  report.uncovered_terms.push_back(
      "Library/EP-private allocations, capture-retained buffers, and concurrent Runs/shared-session admission");
  report.uncovered_terms.push_back(
      "Workload/phase bounds and one active Run are assumptions; max-shape hints are not enforced bounds");
  std::sort(report.uncovered_terms.begin(), report.uncovered_terms.end());
  report.uncovered_terms.erase(std::unique(report.uncovered_terms.begin(), report.uncovered_terms.end()),
                               report.uncovered_terms.end());
  return Status::OK();
}

InlinedVector<SessionMemoryAllocatorSnapshot> SessionState::GetMemoryBackingAllocatorSnapshots() const {
  InlinedVector<SessionMemoryAllocatorSnapshot> snapshots;
  InlinedHashSet<const IAllocator*> seen;
  auto collect = [&](const AllocatorMap& allocators) {
    for (const auto& [info, allocator] : allocators) {
      ORT_UNUSED_PARAMETER(info);
      if (!seen.insert(allocator.get()).second) {
        continue;
      }
      SessionMemoryAllocatorSnapshot snapshot{allocator->Info().name,
                                              MemoryLocation(allocator->Info().device), std::nullopt, std::nullopt};
      AllocatorStats stats;
      allocator->GetStats(&stats);
      // IAllocator's default returns all zeros and has no support bit. Do not call that proven zero.
      if (stats.total_allocated_bytes > 0) {
        snapshot.reported_allocated_bytes = SafeInt<size_t>(stats.total_allocated_bytes);
      }
      if (stats.bytes_in_use > 0) {
        snapshot.reported_in_use_bytes = SafeInt<size_t>(stats.bytes_in_use);
      }
      snapshots.push_back(std::move(snapshot));
    }
  };
  collect(*allocators_);
  if (initializer_allocators_ != nullptr) {
    collect(*initializer_allocators_);
  }
  return snapshots;
}

void SessionState::LogMemoryBackingReport(const char* stage) const {
  if (!memory_backing_report_.has_value()) {
    return;
  }
  const auto& report = *memory_backing_report_;
  LOGS(logger_, INFO) << "Session memory backing report [" << stage << "] graph '" << GetGraphViewer().Name()
                      << "': conditional one-Run model; ordered lifetimes=" << report.ordered_lifetimes
                      << ", uncovered terms=" << report.uncovered_terms.size()
                      << "; modeled live bytes and allocator snapshots must not be added";
  for (const auto& [location, device] : report.devices) {
    LOGS(logger_, INFO) << "Session memory model " << location.ToString()
                        << ": initializers before kernels=" << device.initializers_before_kernel_creation_bytes
                        << ", initializers after PrePack=" << device.initializers_after_prepack_bytes
                        << ", L1 prepack estimate sum=" << device.l1_persistent_prepack_estimate_bytes
                        << ", L1 initialization scratch estimate=" << device.l1_initialization_scratch_estimate_bytes
                        << ", known modeled Run live peak=" << device.modeled_run_live_peak_bytes
                        << ", initialization backing peak=" << BytesOrUnavailable(device.initialization_backing_peak_bytes)
                        << ", Run backing peak=" << BytesOrUnavailable(device.run_backing_peak_bytes);
  }
  for (const auto& snapshot : GetMemoryBackingAllocatorSnapshots()) {
    LOGS(logger_, INFO) << "Allocator-wide snapshot '" << snapshot.name << "' " << snapshot.location.ToString()
                        << ": reported allocated=" << BytesOrUnavailable(snapshot.reported_allocated_bytes)
                        << ", reported in use=" << BytesOrUnavailable(snapshot.reported_in_use_bytes)
                        << "; may include other sessions; not a peak or a session budget";
  }
  for (const auto& term : report.uncovered_terms) {
    LOGS(logger_, VERBOSE) << "Session memory report uncovered: " << term;
  }
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
