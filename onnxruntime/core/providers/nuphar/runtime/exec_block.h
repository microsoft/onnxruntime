// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace tvm_codegen {

class ExecBlock {
 public:
  ExecBlock(
      const std::string& name,
      const std::string& type)
      : name_(name), type_(type) {}

  const std::string& Name() const {
    return name_;
  }

  const std::string& Type() const {
    return type_;
  }

  // TODO: remove node, after spliting shape update and pointer update
  // TODO: remove CodeGenContext after splitting codegen and runtime
  virtual void Run(NupharComputeCtx* compute_ctx) = 0;
  virtual void InitContext(NupharComputeCtx* compute_ctx) = 0;
  virtual void UpdateContext(NupharComputeCtx* compute_ctx) = 0;

 protected:
  std::string name_;  // name_ is for debug
  std::string type_;  // type_ is for debug

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExecBlock);
};  // namespace tvm_codegen

}  // namespace tvm_codegen
}  // namespace onnxruntime
