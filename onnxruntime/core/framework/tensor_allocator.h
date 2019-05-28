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
class ExecutionProviders;
class MemBuffer;

class ITensorAllocator {
 private:
  const ExecutionProviders& exec_providers_;

 public:
  AllocatorPtr GetAllocator(const OrtAllocatorInfo& allocator_info);

  virtual common::Status FinalizePlan() = 0;
  /**
   *
   * \param ort_value_index The index in planner
   * \param name Tensor name. Only for logging purpose
   * \param out The allocated buffer
   *
   * When it succeeded, p could be NULL if the tensor with 'ort_value_index' will not have any element
   */
  virtual common::Status GetPreallocatedBuffer(int ort_value_index, const char* name,
                                               std::unique_ptr<MemBuffer>& out) = 0;
  /**
   * Reserve memory for ort_value_index
   */
  virtual common::Status Trace(int ort_value_index, const ONNX_NAMESPACE::TensorProto* value) = 0;
  virtual ~ITensorAllocator() = default;
  explicit ITensorAllocator(const ExecutionProviders& exec_providers) : exec_providers_(exec_providers) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ITensorAllocator);

  static std::unique_ptr<ITensorAllocator> Create(bool enable_mem_pattern, const ExecutionPlanBase& execution_plan,
                                                  const ExecutionProviders& exec_providers,
                                                  std::vector<BufferUniquePtr>& weights_buffers);
};

}  // namespace onnxruntime
