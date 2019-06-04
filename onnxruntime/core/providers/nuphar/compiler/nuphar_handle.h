// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/allocator.h"  // TODO: get rid of this
#include "core/codegen/common/target_info.h"
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"  // TODO: get rid of this
#include "core/codegen/target/weight_layout.h"
#include "core/codegen/common/handle.h"
#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

//class CodeGenTarget;  // add back after refactoing
class TVMIRBuilder;
class TVMScheduleBuilder;
//class WeightLayoutRegistry;
//struct ShapeExprContext; // add back after refactoing

// TVM is a wrapper containing CodeGen related setting
// TODO: make this the Base
// TODO: create one for nuphar
struct NupharCodeGenHandle : codegen::CodeGenHandle {
  std::shared_ptr<TVMIRBuilder> op_ir_builder;           // keep
  std::shared_ptr<TVMScheduleBuilder> schedule_builder;  // keep
  // maybe add a layout
  WeightLayoutRegistry* layout_registry;
  bool enable_per_node_parallelized;  // TODO: change to config

  bool allow_unaligned_buffers;  // move to another place

  AllocatorPtr allocator;                             // remove
  std::shared_ptr<ShapeExprContext> shape_inference;  // remove

  // TODO: add more planner here
  // such as schedule Planner and Weight layout planner
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
