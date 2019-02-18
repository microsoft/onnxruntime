// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/iexecutor.h"
#include "core/framework/ml_value.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

class SessionState;
class MLValuePatternPlanner;
struct MemoryPatternGroup;
class NodeIndexInfo;

struct MLValueAllocationParameters {
  MLValueAllocationParameters() = default;
  MLValueAllocationParameters(const TensorShape* shape)
      : tensor_shape{shape} {}

  const TensorShape& GetTensorShape() const {
    static const TensorShape s_empty_tensor_shape;
    return tensor_shape != nullptr ? *tensor_shape : s_empty_tensor_shape;
  }

 private:
  const TensorShape* tensor_shape{};
  // todo: is there any parameter needed for ml types?
};

class ExecutionFrame {
 public:
  ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                 const std::vector<std::string>& output_names,
                 const std::vector<MLValue>& fetches,
                 // optional custom allocators. key is index in fetches
                 const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                 const SessionState& session_state);

  ~ExecutionFrame();

  // TODO: These two AllocateMLValue... methods are in the API purely for unit test usage.
  // Fix the unit tests so they set an execution plan that results in these methods being called by
  // GetOrCreateNodeOutputMLValue instead
  Status AllocateMLValueTensorSelfOwnBuffer(int mlvalue_index,
                                            MLDataType element_type,
                                            const OrtAllocatorInfo& location,
                                            const TensorShape& shape,
                                            bool create_fence = false);

  Status AllocateMLValueTensorPreAllocateBuffer(int mlvalue_index_to_allocate,
                                                int mlvalue_index_reuse,
                                                MLDataType element_type,
                                                const OrtAllocatorInfo& location,
                                                const TensorShape& shape,
                                                bool create_fence = false);
  const MLValue& GetMLValue(int mlvalue_index) const {
    ORT_ENFORCE(mlvalue_index >= 0 && static_cast<size_t>(mlvalue_index) < all_values_.size());
    return all_values_[mlvalue_index];
  }

  MLValue& GetMutableMLValue(int mlvalue_index) {
    ORT_ENFORCE(mlvalue_index >= 0 && static_cast<size_t>(mlvalue_index) < all_values_.size());
    return all_values_[mlvalue_index];
  }

  // Get the index for the first entry of the given node.
  int GetNodeOffset(onnxruntime::NodeIndex index) const;

  // Return nullptr if index map to an value that is an unused optional input/output
  const MLValue* GetNodeInputOrOutputMLValue(int index) const;
  MLValue* GetMutableNodeInputOrOutputMLValue(int index);

  // TO DO: make it thread safe
  // This method is not thread safe!
  // Return S_OK and nullptr if index map to an value that is an unused optional input/output
  Status GetOrCreateNodeOutputMLValue(int index,
                                      const MLValueAllocationParameters& parameters,
                                      MLValue*& p_mlvalue);

  AllocatorPtr GetAllocator(const OrtAllocatorInfo& info);

  Status ReleaseMLValue(int mlvalue_idx);

  const SessionState& GetSessionState() const {
    return session_state_;
  }

  Status GeneratePatterns(MemoryPatternGroup* out) const;

  bool HasPlan() const {
    return planner_ != nullptr;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExecutionFrame);

  void Init(const std::unordered_map<std::string, MLValue>& feeds,
            const std::vector<std::string>& output_names,
            const std::vector<MLValue>& fetches,
            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators);

  common::Status AllocateAsPerAllocationPlan(int mlvalue_index,
                                             const MLValueAllocationParameters& parameters);

  Status AllocateMLValueTensorSelfOwnBufferHelper(int mlvalue_index,
                                                  MLDataType element_type,
                                                  const OrtAllocatorInfo& location,
                                                  const TensorShape& shape,
                                                  bool create_fence);

  Status AllocateTensorWithPreAllocateBufferHelper(MLValue* p_mlvalue,
                                                   void* pBuffer,
                                                   MLDataType element_type,
                                                   const OrtAllocatorInfo& location,
                                                   const TensorShape& shape);

  void TraceAllocate(int mlvalue_idx, size_t size);

  void TraceFree(int mlvalue_idx);

  const SequentialExecutionPlan::AllocPlanPerValue& GetAllocationPlan(int mlvalue_idx);

  Status status_;

  const NodeIndexInfo& node_index_info_;

  // All the intermediate values for the entire graph.
  // Input and Output values are passed in by executors
  std::vector<MLValue> all_values_;

  // i-th kernel is still waiting for pending_counts_[i] inputs.
  std::vector<int> pending_counts_;  // not used currently

  // map of index to custom allocator
  std::unordered_map<int, IExecutor::CustomAllocator> custom_allocators_;

  const SessionState& session_state_;

  // If we already have cached memory pattern on these input shapes
  // Use this mem pattern that create a big chunk for all the internal
  // kernel's input/output tensors.
  const MemoryPatternGroup* mem_patterns_;

  // If no cached memory pattern, and we enable the memory pattern optimization
  // use this planner_ to trace the memory allocation in current executor.
  std::unique_ptr<MLValuePatternPlanner> planner_;

  // Record the ml value indices for output values. we won't include those
  // values' allocation in memory pattern, as they can't be shared.
  std::vector<int> output_indices_;

  // Big chunks on different locations that will be used by mem_pattern.
  std::map<OrtAllocatorInfo, BufferUniquePtr> buffers_;
};
}  // namespace onnxruntime
