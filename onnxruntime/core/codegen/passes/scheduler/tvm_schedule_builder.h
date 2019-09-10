// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/passes/scheduler/tvm_scheduler.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// TVMScheduleBuilder contains all applicable TVM scheduler passes.
// Scheduler passes are stored in multiple dispatchers
// that check different conditions of a tvm::Tensor.

// If a tvm::Tensor satisfies more than one TVM scheduler passes,
// the first dispatched pass will be applied.

class TVMScheduleBuilder {
 public:
  // TODO: add more parameter in consructor to support different target
  TVMScheduleBuilder(const std::string& name);
  ~TVMScheduleBuilder() = default;

  void DumpAllSchedulers() const;

  Status Evaluate(
      const tvm::Tensor& tensor,
      const Node* node,
      CodeGenContext& ctx,
      ScheduleContext& sched);

  void InsertDispatcher(std::unique_ptr<TVMScheduleDispatcher>&& ptr);
  void ClearDispatcher();

  const std::string& Name() const;

 private:
  std::vector<std::unique_ptr<TVMScheduleDispatcher>> dispatchers_;
  std::string name_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TVMScheduleBuilder);
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
