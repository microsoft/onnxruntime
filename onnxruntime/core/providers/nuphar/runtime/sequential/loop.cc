// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/loop.h"
#include "core/providers/nuphar/runtime/control_flow/loop_exec_ctx.h"
#include "core/codegen/common/profile.h"

// TODO: refactor it
#include "core/providers/nuphar/runtime/control_flow/scan_exec_ctx.h"

namespace onnxruntime {
namespace tvm_codegen {

LoopExecBlock::LoopExecBlock(const NupharFuncInfo* func_info, const std::string& name)
    : ExecBlock(func_info, name, "LoopExecBlock") {}

void LoopExecBlock::Run(KernelComputeCtx* partition_compute_ctx) {
  if (!initialized_) {
    partition_compute_ctx->CreateFuncComputeCtx(func_info_);
    InitContext(partition_compute_ctx);
    initialized_ = true;
  } else {
    partition_compute_ctx->UpdateFuncComputeCtx(func_info_);
    UpdateContext(partition_compute_ctx);
  }
  FuncComputeCtx& subgraph_compute_ctx = partition_compute_ctx->GetFuncComputeCtx(func_info_);

  const tvm::runtime::PackedFunc& func = func_info_->packed_func;
  int num_func_args = gsl::narrow<int>(func_info_->func_input_count + func_info_->func_output_count);

  // Note tvm_args holds ptr of std::vector<TVMValue> not value, so we only need to assign once.
  tvm::TVMArgs tvm_args(subgraph_compute_ctx.lvalues.data(),
                        func_info_->type_codes.data(),
                        num_func_args);
  tvm::TVMRetValue rvalue;

  // Do it sequentially sicne it is a sequential ExecBlock
  while (subgraph_compute_ctx.loop_cf_ctx->IsValid()) {
    // Note FillTVMArgs would change values of std::vector<DLTensor> and std::vector<TVMValue>, not ptr.
    subgraph_compute_ctx.loop_cf_ctx->FillTVMArgs(partition_compute_ctx, func_info_);

    // Profiling event (no op for non-profiling build)
    CODEGEN_PROFILER_EVENT(loop_CallPacked);

    func.CallPacked(tvm_args, &rvalue);

    subgraph_compute_ctx.loop_cf_ctx->Advance(func_info_->cf_info.get());
  }
  subgraph_compute_ctx.loop_cf_ctx->LoopFinalize();
}

void LoopExecBlock::InitContext(KernelComputeCtx* kernel_compute_ctx) {
  FuncComputeCtx& subgraph_compute_ctx = kernel_compute_ctx->GetFuncComputeCtx(func_info_);

  ORT_ENFORCE_DEBUG(nullptr == subgraph_compute_ctx.loop_cf_ctx);
  if (nullptr != func_info_->cf_info) {
    if (ScanExecInfo::IsType(func_info_->cf_info.get())) {
      subgraph_compute_ctx.loop_cf_ctx = std::make_unique<ScanExecCtx>();
    }
  }

  subgraph_compute_ctx.loop_cf_ctx->InitContext(kernel_compute_ctx, func_info_);
}

void LoopExecBlock::UpdateContext(KernelComputeCtx* kernel_compute_ctx) {
  FuncComputeCtx& subgraph_compute_ctx = kernel_compute_ctx->GetFuncComputeCtx(func_info_);

  ORT_ENFORCE_DEBUG(nullptr != subgraph_compute_ctx.loop_cf_ctx);
  subgraph_compute_ctx.loop_cf_ctx->UpdateContext(kernel_compute_ctx, func_info_);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
