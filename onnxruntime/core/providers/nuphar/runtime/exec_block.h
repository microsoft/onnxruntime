// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/nuphar/compiler/func_info.h"
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"

namespace onnxruntime {
namespace nuphar {

// base class for Execution
class ExecBlock {
 public:
  ExecBlock(
      const NupharFuncInfo* info,
      const std::string& name,
      const std::string& type)
      : func_info_(info), name_(name), type_(type) {}

  virtual ~ExecBlock() = default;

  const std::string& Name() const {
    return name_;
  }

  const std::string& Type() const {
    return type_;
  }

  virtual void Run(KernelComputeCtx* compute_ctx) = 0;
  virtual void InitContext(KernelComputeCtx* compute_ctx) const = 0;
  virtual void UpdateContext(KernelComputeCtx* compute_ctx) const = 0;
  virtual void BlockFinalizer(KernelComputeCtx* kernel_compute_ctx) const {};

 protected:
  const NupharFuncInfo* func_info_;
  std::string name_;  // name_ is for debug
  std::string type_;  // type_ is for debug

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExecBlock);
};

void CreateExecBlock(
    std::vector<std::unique_ptr<ExecBlock>>& exec_blocks,
    const NupharFuncInfo* info,
    const NupharSubgraphUnit& subgraph,
    bool enable_tiling = false);

}  // namespace nuphar
}  // namespace onnxruntime
