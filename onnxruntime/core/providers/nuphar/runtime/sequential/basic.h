// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace nuphar {

// BasicExecBlock is most common execuction block
// It does not contain C++ control flow during execution
class BasicExecBlock : public ExecBlock {
 public:
  BasicExecBlock(const NupharFuncInfo* info,
                 const std::string& name)
      : ExecBlock(info, name, "BasicExecBlock") {}

  BasicExecBlock(const NupharFuncInfo* info,
                 const std::string& name,
                 const std::string& type)
      : ExecBlock(info, name, type) {}

  virtual void Run(KernelComputeCtx* compute_ctx) override;
  void InitContext(KernelComputeCtx* compute_ctx) const override;
  void UpdateContext(KernelComputeCtx* compute_ctx) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BasicExecBlock);
};

}  // namespace nuphar
}  // namespace onnxruntime
