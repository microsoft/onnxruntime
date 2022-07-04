// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include "core/framework/iexecutor.h"
#include "core/framework/stream_handles.h"
//#include "core/framework/execution_plan_base.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/basic_types.h"
#include "core/framework/allocation_planner.h"
#include <vector>
#include <unordered_map>

namespace onnxruntime {

class SessionState;
struct ReleasePlan;
struct AllocPlanPerValue;
struct ParallelExecutionPlanImpl; 
class ISequentialPlannerContext;

// Specify how many logic streams for each provider type
using ProviderStreamMap = std::unordered_map<std::string, int>;
// Each set contains ops which should be grouped in an independent logic stream
using OpStreamMap = std::vector<std::vector<std::string>>;

class DeviceStreamColloectionImpl;
class DeviceStreamColloection {
 public:
  DeviceStreamColloection(size_t num_streams);
  ~DeviceStreamColloection();
  void SetDeviceStream(size_t, std::unique_ptr<Stream> stream);
  void SetDeviceStream(size_t, Stream* stream);
  const std::vector<Stream*>& GetStreams() const;
  size_t NumStreams() const;

 private:
  std::unique_ptr<DeviceStreamColloectionImpl> impl_;
};

class ParallelExecutionPlan : public SequentialExecutionPlan {
 public:
  ParallelExecutionPlan(const SessionState& session_state,
                        const ProviderStreamMap& provider_stream_map,
                        const OpStreamMap& op_stream_map = {});
  ~ParallelExecutionPlan();

  common::Status BindToDeviceStream(Stream* parent_stream, DeviceStreamColloection& device_stream_map) const;

  common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger,
                         const DeviceStreamColloection& device_streams,
                         const bool& terminate_flag = false, 
                         const bool only_execute_path_to_fetches = false);
  size_t NumStreams() const;
  const std::vector<int>& GetRefCounts() const;
  const std::vector<AllocPlanPerValue>& GetAllocPlanPerValue() const;
  std::unique_ptr<ParallelExecutionPlanImpl> impl_;
  std::unique_ptr<ReleasePlan> GenerateReleasePlan() const;
  const std::unordered_map<size_t, size_t>& GetValueToStreamMap() const;
  void GenerateReusePlan(const ISequentialPlannerContext&);
};

//////////////////////////////////////////////////// REFACTORED CLASSES /////////////////////////////////////////////////////////////
/*
1. struct ExecutionPlan, containing logic streams, per-value allocation plan and release plan
2. ExcutionPlanner, fill up an ExecutionPlan with valid logic streams ...
3. ExcutorPlanExecutor, run the ExecutionPlan
*/
struct ExecutionPlanImpl;
struct ExecutionPlannerImpl;
class Node;
class GraphViewer;
class ExecutionProviders;

struct ExecutionPlan {
  ExecutionPlan();
  ~ExecutionPlan();
  size_t NumStreams() const;
  common::Status BindToDeviceStream(Stream* parent_stream,
                                    DeviceStreamColloection& device_stream_map,
                                    IStreamCommandHandleRegistry& stream_handle_registry) const;
  const std::vector<AllocPlanPerValue>& GetAllocationPlan();
  const std::unordered_map<size_t, size_t>& GetValueToStreamMap() const;
  std::unique_ptr<ExecutionPlanImpl> impl_;
};

struct ExecutionPlanner {
  ExecutionPlanner(const Node* parent_node,
                   const GraphViewer& graph_viewer,
                   const std::vector<const NodeArg*>& outer_scope_node_args,
                   const ExecutionProviders& providers,
                   const KernelCreateInfoMap& kernel_create_info_map,
                   const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps,
                   const std::unordered_map<OrtValueName, OrtMemoryInfo>& outer_scope_node_arg_to_location_map,
                   const OrtValueNameIdxMap& ort_value_name_idx_map,
                   IStreamCommandHandleRegistry& stream_handle_registry,
                   const ProviderStreamMap& provider_stream_map,
                   const OpStreamMap& op_stream_map,
                   const ISequentialPlannerContext& context);

  ~ExecutionPlanner();

  onnxruntime::Status CreatePlan(ExecutionPlan& plan);

 private:
  std::unique_ptr<ExecutionPlannerImpl> planner_impl_;
};

struct ExecutionPlanExecutor {
  onnxruntime::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                              const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                              std::vector<OrtValue>& fetches,
                              const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                              const logging::Logger& logger,
                              const DeviceStreamColloection& device_streams,
                              const bool& terminate_flag,
                              const bool only_execute_path_to_fetches);
};

}  // namespace onnxruntime