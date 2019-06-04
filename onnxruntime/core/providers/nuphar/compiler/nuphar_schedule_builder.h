// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <tvm/tvm.h>
#include "core/common/common.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"

// TODO change name space
namespace onnxruntime {
namespace tvm_codegen {

// Traverse iterates tvm::Array<tvm::Tensor> a single node
// and builds the whole schedule (in CodeGenContext)
tvm::Schedule CreateSchedule(const tvm::Array<tvm::Tensor>& outs,
                             NupharCodeGenCtx& ctx_codegen);

}  // namespace tvm_codegen
}  // namespace onnxruntime
