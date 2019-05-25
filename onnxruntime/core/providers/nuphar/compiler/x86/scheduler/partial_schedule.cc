// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"

#include "core/codegen/target/generic/scheduler/schedule_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

// This is for ReuseCount
bool TVM_SCHEDULER_CLASS(True, NupharX86PartialResult)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    CodeGenContext&,
    ScheduleContext& ctx_sched) {
  return TryInlineSchedule(tensor, ctx_sched);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
