// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"

#include "core/codegen/passes/scheduler/schedule_utils.h"

namespace onnxruntime {
namespace nuphar {

// This is for UseCount
bool TVM_SCHEDULER_CLASS(True, NupharX86UseCount)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched) {
  bool status_vec = TryVectorizationX86(tensor, ctx_codegen, ctx_sched);
  bool status_r_and_c = tvm_codegen::InsertRootScheduleAndClosure(tensor, ctx_sched);
  return status_vec || status_r_and_c;
}

bool TVM_SCHEDULER_CLASS(False, NupharX86UseCount)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    tvm_codegen::CodeGenContext&,
    tvm_codegen::ScheduleContext& ctx_sched) {
  return tvm_codegen::TryInlineSchedule(tensor, ctx_sched);
}

}  // namespace nuphar
}  // namespace onnxruntime
