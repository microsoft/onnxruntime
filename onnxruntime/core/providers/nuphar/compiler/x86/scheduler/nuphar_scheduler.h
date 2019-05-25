// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>
#include "core/codegen/target/tvm_scheduler.h"

namespace onnxruntime {
namespace tvm_codegen {

DECLARE_SCHEDULE_DISPATCHER_CLASS(NupharX86UseCount)
DECLARE_SCHEDULE_DISPATCHER_CLASS(NupharX86PartialResult)
DECLARE_SCHEDULE_DISPATCHER_CLASS(NupharX86Tensorize)

DECLARE_TVM_SCHEDULER_CLASS(Extern, NupharX86TVMRule)
DECLARE_TVM_SCHEDULER_CLASS(Reduce, NupharX86TVMRule)

DECLARE_TVM_SCHEDULER_CLASS(MatMulInteger, NupharX86Tensorize)
DECLARE_TVM_SCHEDULER_CLASS(Softmax, NupharX86OrtOpType)
DECLARE_TVM_SCHEDULER_CLASS(Gemm, NupharX86OrtOpType)
DECLARE_TVM_SCHEDULER_CLASS(Conv, NupharX86OrtOpType)
DECLARE_TVM_SCHEDULER_CLASS(MatMul, NupharX86OrtOpType)

DECLARE_TVM_SCHEDULER_CLASS(True, NupharX86UseCount)
DECLARE_TVM_SCHEDULER_CLASS(False, NupharX86UseCount)

DECLARE_TVM_SCHEDULER_CLASS(True, NupharX86PartialResult)

}  // namespace tvm_codegen
}  // namespace onnxruntime
