// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// LoopExecBlock is an ExecBlock for regular loops.
// It is mainly for Scan, LSTM, GRU, RNN, those recurrences.
// It ONLY works a single nested loop for NOW.
// It ONLY works a loop body without other controll flow.

class LoopExecBlock : public ExecBlock {
 public:
  LoopExecBlock(const std::string& name)
      : ExecBlock(name, "LoopExecBlock") {}

  void Run(NupharComputeCtx* compute_ctx) override;
  void InitContext(NupharComputeCtx* compute_ctx) override;
  void UpdateContext(NupharComputeCtx* compute_ctx) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LoopExecBlock);
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
