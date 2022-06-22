// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include "core/framework/iexecutor.h"
#include "core/framework/stream_handles.h"
//#include "core/framework/execution_plan_base.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/basic_types.h"
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

}  // namespace onnxruntime