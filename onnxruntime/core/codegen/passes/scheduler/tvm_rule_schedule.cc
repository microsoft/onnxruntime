// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/scheduler/all_schedules.h"

#include "core/codegen/passes/scheduler/schedule_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

// This is for debug
bool TVM_SCHEDULER_CLASS(AlwaysRoot, GenericTVMRule)::Evaluate(
    const tvm::te::Tensor& tensor,
    const onnxruntime::Node*,
    CodeGenContext&,
    ScheduleContext& ctx_sched) {
  return InsertRootSchedule(tensor, ctx_sched);
}

// For External tvm::te::Tensor
bool TVM_SCHEDULER_CLASS(Extern, GenericTVMRule)::Evaluate(
    const tvm::te::Tensor& tensor,
    const onnxruntime::Node*,
    CodeGenContext&,
    ScheduleContext& ctx_sched) {
  bool status = InsertRootScheduleAndClosure(tensor, ctx_sched);
  bool status_input = InputRootSchedule(tensor, ctx_sched);
  return status || status_input;
}

// For Reduce Compute tvm::te::Tensor
bool TVM_SCHEDULER_CLASS(Reduce, GenericTVMRule)::Evaluate(
    const tvm::te::Tensor& tensor,
    const onnxruntime::Node*,
    CodeGenContext&,
    ScheduleContext& ctx_sched) {
  return InsertRootScheduleAndClosure(tensor, ctx_sched);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
