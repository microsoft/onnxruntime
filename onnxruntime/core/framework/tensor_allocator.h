// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include <core/common/common.h>
#include <core/graph/onnx_protobuf.h>
#include <core/framework/allocator.h>
#include <core/framework/tensor.h>
#include "execution_plan_base.h"

namespace onnxruntime {
class ExecutionPlanBase;
class SessionState;
class MemBuffer;
struct MemoryPatternGroup;

class ITensorAllocator {
 public:
  // Create an ITensorAllocator instance based on enable_mem_pattern
  static std::unique_ptr<ITensorAllocator> Create(bool enable_mem_pattern, const ExecutionPlanBase& execution_plan,
                                                  const SessionState& session_state,
                                                  std::vector<BufferUniquePtr>& weights_buffers);

  AllocatorPtr GetAllocator(const OrtMemoryInfo& memory_info);

  /**
   *
   * \param planned_memory_sizes_in_byte The sizes of memory allocated inside FinalizePlan on different devices.
   *
   * When there is no more tensor to trace, call this function to finalize the
   * allocation.
   */
  virtual common::Status FinalizePlan(std::unordered_map<std::string, size_t>& planned_memory_sizes_in_byte) = 0;

  /**
   * Handing out buffers reserved in @see #Trace() via parameter buf_out,
   * or, in the case of not reserved tensor, returns an allocator so that
   * the caller can take care of the dynamic buffer allocation.
   * buf_out and alloc_out, one and only one can be non-null
   * 
   * @param ort_value_index [In]   int id of the tensor 
   * @param name            [In]   name of the tensor
   * @param buf_out         [Out]  pre reserved buffer, if not null
   * @param alloc_out       [Out]  allocator based on tensor's location, if not null
   * @return 
  */
  virtual common::Status GetPreallocatedBuffer(int ort_value_index, const char* name,
                                               std::unique_ptr<MemBuffer>& buf_out,
                                               AllocatorPtr& alloc_out) = 0;

  virtual const MemoryPatternGroup& GetMemPatterns() = 0;
  /**
   * Reserve memory for ort_value_index
   */
  virtual common::Status Trace(int ort_value_index, const ONNX_NAMESPACE::TensorProto* value) = 0;

  virtual ~ITensorAllocator() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ITensorAllocator);

 protected:
  explicit ITensorAllocator(const SessionState& session_state) : session_state_(session_state) {}

 private:
  const SessionState& session_state_;
};

}  // namespace onnxruntime
