// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/target/tvm_op_creator.h"
#include "core/providers/nuphar/compiler/nuphar_handle.h"
#include "core/codegen/target/tvm_ir_builder.h"
#include "core/codegen/target/tvm_schedule_builder.h"
#include "core/codegen/target/weight_layout.h"  // relocate to target

namespace onnxruntime {
namespace tvm_codegen {

// TVMCodeGenManager contains all registries
// including 1) TVM IR builder registry
//           2) Weight layout transformer registry
//           3) TVM scheduler registry, etc.
// These registries include all applicable passes for specific arch
// AND might also include non-applicable passes, like passes for another arch.

// TVMCodeGenManager keeps the ownerships of all registries, passes,
// and planners.

// TVMCodeGenManager also sets NupharCodeGenHandle for a specific arch.

class TVMCodeGenManager {
 public:
  TVMCodeGenManager();

  // TODO add a list of condition to handle dynamic registration
  void Initialization();

  // TODO: add target as an input
  void SetCodeGenHandle(NupharCodeGenHandle* handle);

 private:
  std::unique_ptr<OpIRRegistry> op_ir_registry_;
  std::unique_ptr<WeightLayoutRegistry> layout_registry_;
  std::unique_ptr<TVMScheduleRegistry> schedule_registry_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
