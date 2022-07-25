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

// Captures information required to allocate/reuse buffer for a ml-value
struct AllocPlanPerValue {
  AllocKind alloc_kind{AllocKind::kNotSet};
  MLDataType value_type{nullptr};
  OrtMemoryInfo location;
  // reused_buffer is valid only if alloc_kind == kReuse. It indicates
  // which OrtValue's buffer must be reused for this OrtValue.
  OrtValueIndex reused_buffer{0};
  // if the value is used in async kernel, a fence object would be created
  // note the fence object would be shared between MLValues reusing the same buffer
  bool create_fence_if_async{false};
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  IntervalT life_interval{0, 0};
  IntervalT allocate_interval{0, 0};
  OrtValueIndex inplace_reuse{-1}; //No in-place reuse
#endif
#ifdef ENABLE_TRAINING
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

  // The following indicates the order in which nodes should be executed and the
  // ml-values to be free after each node's execution:

  // NodeExecutionPlan: represents execution data for a single node
  struct NodeExecutionPlan {
    // node to be executed;
    onnxruntime::NodeIndex node_index;

    // ml-values to be freed after node execution:
    // for (auto i = free_from_index; i <= free_to_index; i++)
    //    free ml-value corresponding to ml-value-index to_be_freed[i]
    int free_from_index;
    int free_to_index;

    explicit NodeExecutionPlan(onnxruntime::NodeIndex index) : node_index(index), free_from_index(1), free_to_index(0) {}
  };

  // Execution_plan: represents the nodes in the sequential order to be executed
  std::vector<NodeExecutionPlan> execution_plan;

  // Records whether a given node has fence on its input or output, key is node index.
  InlinedVector<bool> node_has_fence;

  // to_be_freed: vector elements represent indices of ml-values to be freed (as described above)
  InlinedVector<OrtValueIndex> to_be_freed;

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

  // Whether a given node needs fence check or not.
  bool NodeHasFence(onnxruntime::NodeIndex node_index) const {
    return node_has_fence[node_index];
  }
};

// Output details of an execution plan:
std::ostream& operator<<(std::ostream& out, std::pair<const SequentialExecutionPlan*, const SessionState*> planinfo);
}  // namespace onnxruntime
