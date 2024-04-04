// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/passes/scheduler/tvm_scheduler.h"

namespace onnxruntime {
namespace tvm_codegen {

// AlwaysRoot is for debug purpose
DECLARE_TVM_SCHEDULER_CLASS(AlwaysRoot, GenericTVMRule)
// Create schedule for TVM Rule
DECLARE_TVM_SCHEDULER_CLASS(Extern, GenericTVMRule)
DECLARE_TVM_SCHEDULER_CLASS(Reduce, GenericTVMRule)

// Crete scheduler for ORT OpType, Softmax
DECLARE_TVM_SCHEDULER_CLASS(Softmax, GenericOrtOpType)

}  // namespace tvm_codegen
}  // namespace onnxruntime
