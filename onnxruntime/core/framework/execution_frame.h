// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/iexecutor.h"
#include "core/framework/ml_value.h"
#include "core/framework/node_index_info.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

class SessionState;
class MLValueNameIdxMap;
class MLValuePatternPlanner;
struct MemoryPatternGroup;
class NodeIndexInfo;

class IExecutionFrame {
 protected:
  IExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                  const std::unordered_map<int, OrtValue>& initializers, const std::vector<int>& fetch_mlvalue_idxs,
                  const std::vector<OrtValue>& fetches, const MLValueNameIdxMap& ort_value_idx_map,
                  const NodeIndexInfo& node_index_info);

 public:
  virtual ~IExecutionFrame();

  // Get the index for the first entry of the given node.
  int GetNodeOffset(NodeIndex index) const {
    return node_index_info_.GetNodeOffset(index);
  }

  // Return nullptr if index map to an value that is an unused optional input/output
  const OrtValue* GetNodeInputOrOutputMLValue(int index) const;
  OrtValue* GetMutableNodeInputOrOutputMLValue(int index);

  // TO DO: make it thread safe
  // This method is not thread safe!
  // Return S_OK and nullptr if index map to an value that is an unused optional input/output
  // Shape is required for tensors but not traditional ML values.
  Status GetOrCreateNodeOutputMLValue(int index, const TensorShape* shape, OrtValue*& p_ort_value);

  /**
   * write the output values to the 'fetches' vector
   * Don't access the values after SessionState is destroyed 
   */
  Status GetOutputs(std::vector<OrtValue>& fetches);

  AllocatorPtr GetAllocator(const OrtAllocatorInfo& info) const;

  Status ReleaseMLValue(int ort_value_idx);

 protected:
  // get the ort_value_idx from NodeIndexInfo
  int GetNodeIdxToMLValueIdx(int index) const;

  OrtValue& GetMutableMLValue(int ort_value_index) { return const_cast<OrtValue&>(GetMLValue(ort_value_index)); }

  virtual Status ReleaseMLValueImpl(int ort_value_idx);

  // returns true if the ort_value_idx is an output from the graph
  bool IsOutput(int ort_value_idx) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IExecutionFrame);

  void Init(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
            const std::unordered_map<int, OrtValue>& initializers, const std::vector<int>& fetch_mlvalue_idxs,
            const std::vector<OrtValue>& fetches, const MLValueNameIdxMap& ort_value_idx_map);

  const OrtValue& GetMLValue(int ort_value_index) const {
    ORT_ENFORCE(ort_value_index >= 0 && static_cast<size_t>(ort_value_index) < all_values_.size());
    return all_values_[ort_value_index];
  }

  virtual AllocatorPtr GetAllocatorImpl(const OrtAllocatorInfo& info) const = 0;
  virtual Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) = 0;

  const NodeIndexInfo& node_index_info_;

  // All the intermediate values for the entire graph.
  // Input and Output values are passed in by executors
  std::vector<OrtValue> all_values_;

  const std::vector<int> fetch_mlvalue_idxs_;
};

class ExecutionFrame final : public IExecutionFrame {
 public:
  ExecutionFrame(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds,
                 const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                 // optional custom allocators. key is index in fetches
                 const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                 const SessionState& session_state);

  ~ExecutionFrame() override;

  // TODO: These two AllocateMLValue... methods are in the API purely for unit test usage.
  // Fix the unit tests so they set an execution plan that results in these methods being called by
  // GetOrCreateNodeOutputMLValue instead
  Status AllocateMLValueTensorSelfOwnBuffer(OrtValue& ort_value, int ort_value_index, MLDataType element_type,
                                            const OrtAllocatorInfo& location, const TensorShape& shape,
                                            bool create_fence = false);

  Status AllocateMLValueTensorPreAllocateBuffer(OrtValue& ort_value, int ort_value_index_reuse, MLDataType element_type,
                                                const OrtAllocatorInfo& location, const TensorShape& shape,
                                                bool create_fence = false);

  // thread-safe
  Status GeneratePatterns(MemoryPatternGroup* out) const;

  bool HasMemoryPatternPlanner() const {
    return planner_ != nullptr;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExecutionFrame);

  AllocatorPtr GetAllocatorImpl(const OrtAllocatorInfo& info) const override;
  Status ReleaseMLValueImpl(int ort_value_idx) override;
  Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx, const TensorShape* shape) override;

  common::Status AllocateAsPerAllocationPlan(OrtValue& ort_value, int ort_value_index, const TensorShape* shape);

  Status AllocateMLValueTensorSelfOwnBufferHelper(OrtValue& ort_value, int ort_value_index, MLDataType element_type,
                                                  const OrtAllocatorInfo& location, const TensorShape& shape,
                                                  bool create_fence);

  Status AllocateTensorWithPreAllocateBufferHelper(OrtValue& ort_value, void* pBuffer, MLDataType element_type,
                                                   const OrtAllocatorInfo& location, const TensorShape& shape);

  void TraceAllocate(int ort_value_idx, size_t size);
  void TraceFree(int ort_value_idx);

  const AllocPlanPerValue& GetAllocationPlan(int ort_value_idx);

  const SessionState& session_state_;

  // map of index to custom allocator
  std::unordered_map<int, IExecutor::CustomAllocator> custom_allocators_;

  // If we already have cached memory pattern on these input shapes
  // Use this mem pattern that create a big chunk for all the internal
  // kernel's input/output tensors.
  const MemoryPatternGroup* mem_patterns_;

  // If no cached memory pattern, and we enable the memory pattern optimization
  // use this planner_ to trace the memory allocation in current executor.
  std::unique_ptr<MLValuePatternPlanner> planner_;

  // Big chunks on different locations that will be used by mem_pattern.
  std::map<OrtAllocatorInfo, BufferUniquePtr> buffers_;
};
}  // namespace onnxruntime
