// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/passes/scheduler/tvm_scheduler.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

DECLARE_SCHEDULE_DISPATCHER_CLASS(NupharX86UseCount)
DECLARE_SCHEDULE_DISPATCHER_CLASS(NupharX86PartialResult)
DECLARE_SCHEDULE_DISPATCHER_CLASS(NupharX86Tensorize)

DECLARE_TVM_SCHEDULER_CLASS(Extern, NupharX86TVMRule)
DECLARE_TVM_SCHEDULER_CLASS(Reduce, NupharX86TVMRule)

DECLARE_TVM_SCHEDULER_CLASS(MatMulInteger, NupharX86Tensorize)
DECLARE_TVM_SCHEDULER_CLASS(MatMulInteger16, NupharX86Tensorize)
DECLARE_TVM_SCHEDULER_CLASS(Softmax, NupharX86OrtOpType)
DECLARE_TVM_SCHEDULER_CLASS(Gemm, NupharX86OrtOpType)
DECLARE_TVM_SCHEDULER_CLASS(Conv, NupharX86OrtOpType)
DECLARE_TVM_SCHEDULER_CLASS(MatMul, NupharX86OrtOpType)
DECLARE_TVM_SCHEDULER_CLASS(Split, NupharX86OrtOpType)

DECLARE_TVM_SCHEDULER_CLASS(True, NupharX86UseCount)
DECLARE_TVM_SCHEDULER_CLASS(False, NupharX86UseCount)

DECLARE_TVM_SCHEDULER_CLASS(True, NupharX86PartialResult)

// utilities
bool TryVectorizationX86(
    const tvm::Tensor& tensor,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched);

bool InputRootScheduleWithVectorizationX86(
    const tvm::Tensor& tensor,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched);

bool TryParallelX86(
    const tvm::Tensor& tensor,
    int64_t to_dim,  // fuse dims before to_dim for parallel schedule, 0 to fuse all but last dim
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm_codegen::ScheduleContext& ctx_sched);

constexpr auto kNupharScheduleNoParallel = "nuphar_schedule_no_parallel";

}  // namespace nuphar
}  // namespace onnxruntime
