// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"

#include "core/codegen/target/generic/scheduler/schedule_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

// This is for UseCount
bool TVM_SCHEDULER_CLASS(True, NupharX86UseCount)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    CodeGenContext&,
    ScheduleContext& ctx_sched) {
  // TODO change it to the value from Target
  int64_t natural_vector_size = 16;

  bool status_vec = TryVectorization(tensor, natural_vector_size, ctx_sched);
  bool status_r_and_c = InsertRootScheduleAndClosure(tensor, ctx_sched);
  return status_vec || status_r_and_c;
}

bool TVM_SCHEDULER_CLASS(False, NupharX86UseCount)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    CodeGenContext&,
    ScheduleContext& ctx_sched) {
  return TryInlineSchedule(tensor, ctx_sched);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
