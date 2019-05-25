// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// BasicExecBlock is most common execuction block
// It does not contain C++ control flow during execution
class BasicExecBlock : public ExecBlock {
 public:
  BasicExecBlock(const std::string& name)
      : ExecBlock(name, "BasicExecBlock") {}

  void Run(NupharComputeCtx* compute_ctx) override;
  void InitContext(NupharComputeCtx* compute_ctx) override;
  void UpdateContext(NupharComputeCtx* compute_ctx) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BasicExecBlock);
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
