// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/loop.h"
#include "core/providers/nuphar/runtime/control_flow/loop_exec_ctx.h"
#include "core/codegen/common/profile.h"

// TODO: refactor it
#include "core/providers/nuphar/runtime/control_flow/scan_exec_ctx.h"

namespace onnxruntime {
namespace tvm_codegen {

void LoopExecBlock::Run(NupharComputeCtx* compute_ctx) {
  if (compute_ctx->HasInitialized()) {
    UpdateContext(compute_ctx);
  } else {
    InitContext(compute_ctx);
  }

  LoopExecCtx* loop_cf_ctx = compute_ctx->GetControlFlowCtx();
  const NupharFuncInfo& func_info = compute_ctx->GetNupharFuncInfo();
  const tvm::runtime::PackedFunc& func = func_info.packed_func;
  int num_args = gsl::narrow<int>(func_info.input_count + func_info.output_count);

  // Note tvm_args holds ptr of std::vector<TVMValue> not value, so we only need to assign once.
  tvm::TVMArgs tvm_args(compute_ctx->GetTVMValues().data(), func_info.type_codes.data(), num_args);
  tvm::TVMRetValue rvalue;

  // Do it sequentially sicne it is a sequential ExecBlock
  while (loop_cf_ctx->IsValid()) {
    // Note FillTVMArgs would change values of std::vector<DLTensor> and std::vector<TVMValue>, not ptr.
    loop_cf_ctx->FillTVMArgs(compute_ctx);

    CODEGEN_PROFILER_EVENT(loop_CallPacked);
    func.CallPacked(tvm_args, &rvalue);

    loop_cf_ctx->Advance(func_info.cf_info.get());
  }
  loop_cf_ctx->LoopFinalize();
}

void LoopExecBlock::InitContext(NupharComputeCtx* compute_ctx) {
  LoopExecCtx* loop_cf_ctx = compute_ctx->GetControlFlowCtx();
  loop_cf_ctx->InitContext(compute_ctx);
}

void LoopExecBlock::UpdateContext(NupharComputeCtx* compute_ctx) {
  LoopExecCtx* loop_cf_ctx = compute_ctx->GetControlFlowCtx();
  loop_cf_ctx->UpdateContext(compute_ctx);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
