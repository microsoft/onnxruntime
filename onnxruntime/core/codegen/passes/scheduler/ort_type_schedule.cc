// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/scheduler/all_schedules.h"

#include "core/codegen/passes/scheduler/schedule_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

bool TVM_SCHEDULER_CLASS(Softmax, GenericOrtOpType)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    CodeGenContext&,
    ScheduleContext& ctx_sched) {
  // compute root the exp since it is reused more than once
  auto& tensor_exp = tensor->op->InputTensors()[0];
  return InsertRootSchedule(tensor_exp, ctx_sched);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
