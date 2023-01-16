// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/common/inlined_containers.h"
#include "core/framework/alloc_kind.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_plan_base.h"
#include "core/graph/graph.h"

namespace onnxruntime {
// Every ml-value has a unique name and is assigned a unique integral number.
// While we use names at static-planning time, the goal is that at runtime
// (that is, at inference time), there is no need to refer to names, and only
// the integer index is used (e.g., to index into appropriate vectors in
// the ExecutionFrame).
using OrtValueIndex = int;
using OrtValueName = std::string;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
// pair of start and end program counters,according to the execution plan
using IntervalT = std::pair<size_t, size_t>;
#endif

class SessionState;
class StreamExecutionContext;
class SessionScope;

// Captures information required to allocate/reuse buffer for a ml-value
struct AllocPlanPerValue {
  AllocKind alloc_kind{AllocKind::kNotSet};
  MLDataType value_type{nullptr};
  OrtMemoryInfo location;
  // reused_buffer is valid only if alloc_kind == kReuse. It indicates
  // which OrtValue's buffer must be reused for this OrtValue.
  OrtValueIndex reused_buffer{0};
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  IntervalT life_interval{0, 0};
  IntervalT allocate_interval{0, 0};
  OrtValueIndex inplace_reuse{-1};  // No in-place reuse
#endif
#ifdef ENABLE_STRIDED_TENSORS
  // is_strided_tensor indicates if this OrtValue is strided tensor.
  // If alloc_kind is kReuse, it reuses one of the node inputs (like Expand),
  // if alloc_kind is kAllocate, it will only allocate required buffer size (like ConstantOfShape).
  bool is_strided_tensor{false};
#endif

  class ProgramCounter {
   public:
    ProgramCounter() = default;
    void AddStart(size_t start) {
      ORT_ENFORCE(starts_.size() == ends_.size(), "Previous entry was not terminated.");
      ORT_ENFORCE(starts_.empty() || start > ends_.back(), "Invalid 'start'. Value is smaller than previous 'end'.");
      starts_.push_back(start);
    }

    void AddEnd(size_t end) {
      ORT_ENFORCE(starts_.size() == ends_.size() + 1, "No matching 'start' entry.");
      ORT_ENFORCE(end >= starts_.back(), "Invalid 'end'. Value is larger than 'start'.");
      ends_.push_back(end);
    }

    // return true if there are entries, and the number of start/end pairs match.
    // validity of the individual start/end values is checked when they are added.
    bool HasValidEntries() const {
      return !starts_.empty() && starts_.size() == ends_.size();
    }

    const std::vector<size_t>& Starts() const { return starts_; }
    const std::vector<size_t>& Ends() const { return ends_; }

   private:
    std::vector<size_t> starts_;
    std::vector<size_t> ends_;
  };

  ProgramCounter program_counter;

 public:
  AllocPlanPerValue() : location(CPU, OrtInvalidAllocator) {}
};

using NotificationIndex = size_t;

// SequentialExecutionPlan: This is the data that is produced by a static
// planner for a sequential execution, to be used by a SequentialExecutor.
struct SequentialExecutionPlan : public ExecutionPlanBase {
  // Allocation plan:
  // ExecutionFrame::GetOrCreateTensor() should use the following information
  // to decide whether to allocate a new buffer or reuse an existing buffer

  // The following vector is indexed by OrtValueIndex
  std::vector<AllocPlanPerValue> allocation_plan;

  // The following vector contains any initializer tensors that must be allocated sequentially.
  std::vector<OrtValueIndex> initializer_allocation_order;

  // The following vector contains any activation tensors that must be allocated sequentially.
  std::vector<OrtValueIndex> activation_allocation_order;

  // A execution step in the execution step.
  // we explicitly encoding the cross-stream synchronization
  // in the execution pan, so we wwill mainly have following
  // types of steps:
  // 1. Kernel Launch
  // 2. Activate notification
  // 3. Wait on a notificaiton
  class ExecutionStep {
   public:
    virtual ~ExecutionStep() {}
    virtual Status Execute(StreamExecutionContext& ctx,
                           size_t stream_idx,
                           SessionScope& session_scope,
                           const bool& terminate_flag,
                           bool& continue_flag) = 0;
    virtual std::string ToString() const = 0;
#ifdef ENABLE_TRAINING
    // the partial execution mode for training needs special handling for barrier
    virtual bool IsBarrier() const { return false; }
#endif
  };
  // LogicStream is a sequence of execution steps that can be executed independetly.
  // The steps within a sequence are executed in order, and happened on the same device.
  struct LogicStream {
    std::vector<std::unique_ptr<ExecutionStep>> steps_;
    const OrtDevice& device_;
#ifdef ENABLE_TRAINING
    std::vector<NodeIndex> step_pc;
#endif
   public:
    LogicStream(const OrtDevice& device) : device_(device) {}
  };
  // a execution plan is composed by multiple logic stream.
  // by default all the nodes with the same device will be group in to the same stream.
  InlinedVector<std::unique_ptr<LogicStream>> execution_plan;
  // the map from ort_value index to the logic stream index.
  InlinedHashMap<size_t, size_t> value_to_stream_map;

  // If a tensor is consumed by multiple logic streams,
  // we can't static predict when will this tensor can be
  // released. So we design this ref-count based release plan.
  // if the ref count is 1, we still can static determin when
  // the tensor can be released.
  struct ReleaseAction {
    size_t value_index;
    // 0 - no release needed
    // 1 - can be statically determined where to release
    // >1 - can't statically determined, need ref counting.
    size_t ref_count{0};
  };

  std::vector<ReleaseAction> release_actions;

  // for each node, which values need to be freed after kernel execution.
  // indexed by node index
  // elements in node_release_list[i] is the index in release_actions.
  std::vector<std::vector<size_t>> node_release_list;
  // for each notification, what is the stream-idx of the its owner.
  std::vector<size_t> notification_owners;
  // key: notification index.
  // value:  {stream_idx, step_idx}
  // giving a notificaiton, we used this map to figure out what is the downstream steps it need to trigger.
  InlinedHashMap<onnxruntime::NotificationIndex, std::vector<std::pair<size_t, size_t>>> downstream_map;

  size_t num_barriers{0};

#ifdef ENABLE_TRAINING
  InlinedVector<NodeIndex> node_execution_order_in_training;
#endif

  const std::vector<AllocPlanPerValue>& GetAllocationPlan() const {
    return allocation_plan;
  }

  const InlinedHashMap<size_t, size_t>& GetValueToStreamMap() const {
    return value_to_stream_map;
  }

  const OrtMemoryInfo& GetLocation(size_t ort_value_index) const override {
    return allocation_plan[ort_value_index].location;
  }

  void SetLocation(size_t ort_value_index, const struct OrtMemoryInfo& info) override {
    allocation_plan[ort_value_index].location = info;
  }

  InlinedHashSet<OrtMemoryInfo> GetAllLocations() const override {
    InlinedHashSet<OrtMemoryInfo> locations;
    locations.reserve(allocation_plan.size());
    for (auto& alloc_plan : allocation_plan) {
      locations.insert(alloc_plan.location);
    }
    return locations;
  }

  size_t NumberOfValidStreams() const {
    size_t count = 0;
    for (auto& stream : execution_plan) {
      if (!stream->steps_.empty())
        count++;
    }
    return count;
  }
};

// Output details of an execution plan:
std::ostream& operator<<(std::ostream& out, std::pair<const SequentialExecutionPlan*, const SessionState*> planinfo);
}  // namespace onnxruntime
