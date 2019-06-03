// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include "tensor_allocator.h"
#include "mem_pattern.h"
#include "ml_value_patterns_planner.h"
#include "utils.h"

namespace onnxruntime {
class ExecutionProviders;

class SimpleTensorAllocator : public ITensorAllocator {
 private:
  MemoryPatternGroup mem_patterns_;
  std::vector<BufferUniquePtr>& weights_buffers_;
  const ExecutionPlanBase& seq_plan_;

 private:
  std::unordered_map<int, const ONNX_NAMESPACE::TensorProto*> values_;

 public:
  SimpleTensorAllocator(const ExecutionPlanBase& execution_plan, const ExecutionProviders& exec_providers,
                        std::vector<BufferUniquePtr>& weights_buffers)
      : ITensorAllocator(exec_providers),
        weights_buffers_(weights_buffers),
        seq_plan_(execution_plan) {}
  common::Status FinalizePlan() override { return Status::OK(); }
  common::Status GetPreallocatedBuffer(int ort_value_index, const char* name, std::unique_ptr<MemBuffer>& out) override;
  common::Status Trace(int id, const ONNX_NAMESPACE::TensorProto* value) override;
};
}  // namespace onnxruntime
