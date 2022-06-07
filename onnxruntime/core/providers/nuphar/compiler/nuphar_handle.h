// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/codegen/common/handle.h"
#include "core/codegen/common/target_info.h"
#include "core/codegen/passes/weight_layout/weight_layout.h"
#include "core/framework/allocator.h"                             // TODO: get rid of this
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"  // TODO: get rid of this

namespace onnxruntime {

// forwarding
namespace tvm_codegen {
class TVMIRBuilder;
class TVMScheduleBuilder;
}  // namespace tvm_codegen

namespace nuphar {

// TVM is a wrapper containing CodeGen related setting
// TODO: make this the Base
// TODO: create one for nuphar
struct NupharCodeGenHandle : codegen::CodeGenHandle {
  std::shared_ptr<tvm_codegen::TVMIRBuilder> op_ir_builder;           // keep
  std::shared_ptr<tvm_codegen::TVMScheduleBuilder> schedule_builder;  // keep
  // maybe add a layout
  tvm_codegen::WeightLayoutRegistry* layout_registry;
  int64_t parallel_min_workloads;

  bool allow_unaligned_buffers;  // move to another place

  AllocatorPtr allocator;                             // remove
  std::shared_ptr<ShapeExprContext> shape_inference;  // remove
};

}  // namespace nuphar
}  // namespace onnxruntime
