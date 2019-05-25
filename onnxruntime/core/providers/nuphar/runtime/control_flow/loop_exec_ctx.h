// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/compiler/nuphar_func_ctx.h"

namespace onnxruntime {
namespace tvm_codegen {

class NupharComputeCtx;

class LoopExecCtx {
 public:
  LoopExecCtx() {}

  virtual void InitContext(NupharComputeCtx* compute_ctx) = 0;
  virtual void UpdateContext(NupharComputeCtx* compute_ctx) = 0;
  virtual void FillTVMArgs(NupharComputeCtx* compute_ctx) = 0;

  virtual void LoopFinalize() = 0;
  // Marching to next loop iteration
  virtual void Advance(const ControlFlowInfo* cf_info) = 0;
  virtual bool IsValid() {
    return current_loop_step_ < max_loop_step_;
  }

 protected:
  std::vector<int> sequence_lens_;

  // current sequence index that are going to run
  int current_loop_step_;
  int min_loop_step_;
  int max_loop_step_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
