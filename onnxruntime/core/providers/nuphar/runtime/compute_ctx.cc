// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/control_flow/scan_exec_ctx.h"

namespace onnxruntime {
namespace tvm_codegen {

NupharComputeCtx::NupharComputeCtx(
    const nuphar::NupharRuntimeHandle* handle,
    std::unordered_map<std::string, int64_t>& realized_dims,
    const NupharFuncInfo& func_info,
    DataAllocFunc data_alloc_func)
    : data_alloc_func_(data_alloc_func),
      func_info_(func_info),
      handle_(handle),
      realized_dims_(realized_dims) {
  // create ControlFlowCtx based on the ControlInfo
  if (nullptr != func_info_.cf_info) {
    if (ScanExecInfo::IsType(func_info_.cf_info.get())) {
      loop_cf_ctx_ = std::make_unique<ScanExecCtx>();
    }
  }
}

void NupharComputeCtx::Bind(OpKernelContext* op_kernel_ctx) {
  op_kernel_ctx_ = op_kernel_ctx;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
