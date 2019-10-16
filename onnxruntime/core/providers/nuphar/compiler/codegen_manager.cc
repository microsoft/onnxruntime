// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/codegen_manager.h"

#include "core/codegen/passes/op_ir_creator/all_ops.h"
#include "core/codegen/passes/scheduler/all_schedules.h"
#include "core/codegen/passes/weight_layout/transpose_2d.h"
#include "core/codegen/passes/weight_layout/tiling_2d.h"
#include "core/codegen/passes/weight_layout/vertical_stripes_2d.h"
#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"
#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"

namespace onnxruntime {
namespace codegen {
// explicit instantiation
template class RegistryBase<tvm_codegen::WeightLayout>;
}  // namespace codegen

namespace nuphar {

////  All Creator instance registration
// 1. Create Customized Op IR creator instances

// BEGIN: NupharTVM X86 IR creator classes

#define ADD_OP_ITEM(name) \
  op_ir_registry->Register(std::move(onnxruntime::make_unique<NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(name)>()));

#define POOL_OP(OP) ADD_OP_ITEM(OP)
#define REDUCE_V_OP(name) ADD_OP_ITEM(name)
#define UNARY_OP(name) ADD_OP_ITEM(name)

static void RegisterAllNupharX86OpIRCreators(tvm_codegen::OpIRRegistry* op_ir_registry) {
  LIST_ALL_X86_OPS()
}

#undef ADD_OP_ITEM
#undef POOL_OP
#undef REDUCE_V_OP
#undef UNARY_OP

// END: NupharTVM X86 IR creator classes

// 2. Create Scheduler instances
// BEGIN: Nuphar Scheduler classes

static void RegisterAllNupharSchedulers(tvm_codegen::TVMScheduleRegistry* sched_registry) {
  // Add Generic TVM Rule schedules
  sched_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::TVM_SCHEDULER_CLASS(AlwaysRoot, GenericTVMRule)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::TVM_SCHEDULER_CLASS(Extern, GenericTVMRule)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::TVM_SCHEDULER_CLASS(Reduce, GenericTVMRule)>()));

  // Add Generic OpType schedules
  sched_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::TVM_SCHEDULER_CLASS(Softmax, GenericOrtOpType)>()));

  // Add NupharX86 TVM Rule schedules
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(Extern, NupharX86TVMRule)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(Reduce, NupharX86TVMRule)>()));

  // Add NupharX86 Tensorization schedules
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(MatMulInteger, NupharX86Tensorize)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(MatMulInteger16, NupharX86Tensorize)>()));

  // Add NupharX86 OpType schedules
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(Softmax, NupharX86OrtOpType)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(Split, NupharX86OrtOpType)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(Gemm, NupharX86OrtOpType)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(MatMul, NupharX86OrtOpType)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(Conv, NupharX86OrtOpType)>()));

  // Add NupharX86 use count schedules
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(True, NupharX86UseCount)>()));
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(False, NupharX86UseCount)>()));

  // Add NupharX86 partial result schedules
  sched_registry->Register(
      std::move(onnxruntime::make_unique<TVM_SCHEDULER_CLASS(True, NupharX86PartialResult)>()));
}

// END: Nuphar Scheduler classes

// 3. Create Weight layout instances
// BEGIN: Nuphar Weight Layouts classes
static void RegisterAllNupharWeightLayouts(tvm_codegen::WeightLayoutRegistry* layout_registry) {
  // AVX512
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTiling2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8, 64)));
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTiling2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16, 64)));
  // AVX2
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTiling2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8, 32)));
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTiling2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16, 32)));
  // AVX
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTiling2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8, 16)));
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTiling2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16, 16)));

  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutVerticalStripe2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, 8)));
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTranspose2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)));
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTranspose2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8)));
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTranspose2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)));
  layout_registry->Register(
      std::move(onnxruntime::make_unique<tvm_codegen::WeightLayoutTranspose2D>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16)));
}

// END: Nuphar Weight Layouts classes

//// All Plugins for Nuphar provider
// 1. Plugin IR creator classes

// BEGIN: Nuphar TVM X86 IR creator classes
#define ADD_OP_ITEM(name) \
  dispatcher->Register(#name, registry->Get(NUPHAR_TVM_X86_OP_IR_CREATOR_STRING(name)));

#define POOL_OP(OP) ADD_OP_ITEM(OP)
#define REDUCE_V_OP(name) ADD_OP_ITEM(name)
#define UNARY_OP(name) ADD_OP_ITEM(name)

static void RegisterNupharX86Dispatcher(const std::shared_ptr<tvm_codegen::TVMIRBuilder>& builder,
                                        const tvm_codegen::OpIRRegistry* registry) {
  auto dispatcher = onnxruntime::make_unique<tvm_codegen::OP_IR_DISPATCHER_CLASS(OpType)>("OptypeNupharTVMX86Creators");
  LIST_ALL_X86_OPS()
  builder->InsertDispatcher(std::move(dispatcher));
}

#undef ADD_OP_ITEM
#undef POOL_OP
#undef REDUCE_V_OP
#undef UNARY_OP
// END: Nuphar TVM X86 IR creator classes

// 2 Plugin Scheduler classes

// BEGIN: TVM rule Scheduler
static void RegisterNupharX86TVMRuleSchedulers(const std::shared_ptr<tvm_codegen::TVMScheduleBuilder>& builder,
                                               const tvm_codegen::TVMScheduleRegistry* registry) {
  auto dispatcher = onnxruntime::make_unique<tvm_codegen::SCHEDULE_DISPATCHER_CLASS(TVMOpRule)>("NupharX86TVMRuleSchedulers");

  // Register a scheduler for TVM External Tensor
  dispatcher->Register(tvm_codegen::GetTVMOpRule(tvm_codegen::TVMOpRuleType::Extern),
                       registry->Get(TVM_SCHEDULER_STRING(Extern, NupharX86TVMRule)));
  // Register a scheduler for TVM Reduce Tensor
  dispatcher->Register(tvm_codegen::GetTVMOpRule(tvm_codegen::TVMOpRuleType::ComputeReduce),
                       registry->Get(TVM_SCHEDULER_STRING(Reduce, NupharX86TVMRule)));

  builder->InsertDispatcher(std::move(dispatcher));
}
// END: TVM rule Scheduler

// BEGIN: Tensorization Scheduler
static void RegisterNupharX86TensorizeScheduler(const std::shared_ptr<tvm_codegen::TVMScheduleBuilder>& builder,
                                                const tvm_codegen::TVMScheduleRegistry* registry) {
  auto dispatcher = onnxruntime::make_unique<SCHEDULE_DISPATCHER_CLASS(NupharX86Tensorize)>("NupharOrtTensorizeScheduler");

  // Register a scheduler for Ort MatMulInteger
  dispatcher->Register("MatMulInteger",
                       registry->Get(TVM_SCHEDULER_STRING(MatMulInteger, NupharX86Tensorize)));
  builder->InsertDispatcher(std::move(dispatcher));
}
// END: Tensorization Scheduler

// BEGIN: ORT OpType Scheduler
static void RegisterNupharX86OrtOpTypeSchedulers(const std::shared_ptr<tvm_codegen::TVMScheduleBuilder>& builder,
                                                 const tvm_codegen::TVMScheduleRegistry* registry) {
  auto dispatcher = onnxruntime::make_unique<tvm_codegen::SCHEDULE_DISPATCHER_CLASS(OrtOpType)>("NupharX86OrtOpTypeSchedulers");

  // Register a scheduler for Ort Softmax OpType
  dispatcher->Register("Softmax",
                       registry->Get(TVM_SCHEDULER_STRING(Softmax, NupharX86OrtOpType)));

  dispatcher->Register("Split",
                       registry->Get(TVM_SCHEDULER_STRING(Split, NupharX86OrtOpType)));

  builder->InsertDispatcher(std::move(dispatcher));
}
// END: ORT OpType Scheduler

// BEGIN: Reuse Count Analysis Scheduler
static void RegisterNupharX86UseCountSchedulers(const std::shared_ptr<tvm_codegen::TVMScheduleBuilder>& builder,
                                                const tvm_codegen::TVMScheduleRegistry* registry) {
  auto dispatcher = onnxruntime::make_unique<SCHEDULE_DISPATCHER_CLASS(NupharX86UseCount)>("NupharX86UseCountSchedulers");

  // Register a scheduler for Reuse count > 1
  dispatcher->Register("True",
                       registry->Get(TVM_SCHEDULER_STRING(True, NupharX86UseCount)));

  // Register a scheduler for Reuse count <= 1
  dispatcher->Register("False",
                       registry->Get(TVM_SCHEDULER_STRING(False, NupharX86UseCount)));

  builder->InsertDispatcher(std::move(dispatcher));
}
// END: Reuse Count Analysis Scheduler

// BEGIN: Partial Result Scheduler
static void RegisterNupharX86PartialResultSchedulers(const std::shared_ptr<tvm_codegen::TVMScheduleBuilder>& builder,
                                                     const tvm_codegen::TVMScheduleRegistry* registry) {
  auto dispatcher = onnxruntime::make_unique<SCHEDULE_DISPATCHER_CLASS(NupharX86PartialResult)>("NupharX86PartialResultSchedulers");
  dispatcher->Register("True",
                       registry->Get(TVM_SCHEDULER_STRING(True, NupharX86PartialResult)));

  builder->InsertDispatcher(std::move(dispatcher));
}
// END: Partial Result Scheduler

TVMCodeGenManager::TVMCodeGenManager() {
  op_ir_registry_ = onnxruntime::make_unique<tvm_codegen::OpIRRegistry>();
  layout_registry_ = onnxruntime::make_unique<tvm_codegen::WeightLayoutRegistry>();
  schedule_registry_ = onnxruntime::make_unique<tvm_codegen::TVMScheduleRegistry>();
}

void TVMCodeGenManager::Initialization() {
  RegisterAllNupharX86OpIRCreators(op_ir_registry_.get());
  RegisterAllGenericOpIRCreators(op_ir_registry_.get());

  RegisterAllNupharWeightLayouts(layout_registry_.get());
  RegisterAllNupharSchedulers(schedule_registry_.get());
}

// TODO Add isa support
void TVMCodeGenManager::SetCodeGenHandle(NupharCodeGenHandle* handle) {
  // layout registry
  handle->layout_registry = layout_registry_.get();

  // Op IR creators
  handle->op_ir_builder =
      std::make_shared<tvm_codegen::TVMIRBuilder>("Nuphar_Op_IR_Builder");
  RegisterNupharX86Dispatcher(handle->op_ir_builder, op_ir_registry_.get());
  RegisterGenericOrtOpTypeDispatcher(handle->op_ir_builder, op_ir_registry_.get());

  // Schedulers
  handle->schedule_builder =
      std::make_shared<tvm_codegen::TVMScheduleBuilder>("Nuphar_Schedule_Builder");

  RegisterNupharX86TVMRuleSchedulers(handle->schedule_builder, schedule_registry_.get());
  RegisterNupharX86TensorizeScheduler(handle->schedule_builder, schedule_registry_.get());
  RegisterNupharX86OrtOpTypeSchedulers(handle->schedule_builder, schedule_registry_.get());
  RegisterNupharX86UseCountSchedulers(handle->schedule_builder, schedule_registry_.get());
  RegisterNupharX86PartialResultSchedulers(handle->schedule_builder, schedule_registry_.get());
}

}  // namespace nuphar
}  // namespace onnxruntime
