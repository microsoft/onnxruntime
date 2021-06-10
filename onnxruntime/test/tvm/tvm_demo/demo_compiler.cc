// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/tvm/tvm_demo/demo_compiler.h"

#include "core/codegen/passes/scheduler/schedule_utils.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/codegen/passes/op_ir_creator/tvm_ir_builder.h"
#include "core/codegen/passes/scheduler/tvm_scheduler.h"
#include "core/codegen/passes/scheduler/tvm_schedule_builder.h"

#include <tvm/tvm.h>
#include <tvm/build_module.h>

namespace onnxruntime {
namespace tvm_demo {

// Create a dummy demo handle
static codegen::CodeGenHandle demo_handle;
// Create a dummy demo codegen context
static tvm_codegen::CodeGenContext demo_codegen_ctx(&demo_handle);

// Translate an Ort graph into tvm IR
// Note this function is specific for this demo.
// This function uses specific way for graph traversal or constructing tvm placeholders.
// It may or may not work for a universal Ort graph.
// For a more general example, please check nuphar provider.
DemoTVMTensorCtx BuildTVMIR(const onnxruntime::Graph& graph) {
  // Create OpIRRegistry that holds all OpIRCreators
  std::unique_ptr<tvm_codegen::OpIRRegistry> op_ir_registry =
      std::make_unique<tvm_codegen::OpIRRegistry>();

  // Register all generic OpIRCreators
  tvm_codegen::RegisterAllGenericOpIRCreators(op_ir_registry.get());

  // Create OpIRBuilder
  std::shared_ptr<tvm_codegen::TVMIRBuilder> op_ir_builder =
      std::make_shared<tvm_codegen::TVMIRBuilder>("Demo_Op_IR_Builder");

  // Attach all generic OpIRCreators from op_ir_registry to op_ir_builder
  tvm_codegen::RegisterGenericOrtOpTypeDispatcher(op_ir_builder, op_ir_registry.get());

  // Create DemoTVMTensorCtx holdings tvm IR
  DemoTVMTensorCtx result;

  // Local lookup from name to tvm::Tensor
  std::unordered_map<std::string, tvm::Tensor> tvm_tensors;

  // Note this is a simplified traversal that works specifically for this demo
  // but may or may not work for an univerisal model.
  // For more general traversal, please check nuphar provider.
  for (auto& node : graph.Nodes()) {
    tvm::Array<tvm::Tensor> inputs;
    tvm::Array<tvm::Tensor> outputs;

    // Get inputs
    for (auto& def : node.InputDefs()) {
      const std::string& name = def->Name();
      auto iter = tvm_tensors.find(name);
      // Always create placeholder when not finding a tensor
      // Note it is for this demo.
      // It may or may not work for a universal graph.
      if (iter == tvm_tensors.end()) {
        tvm_tensors[name] =
            tvm::placeholder(ShapeToTvmArray(def, demo_codegen_ctx),
                             tvm_codegen::ToTvmType(TensorProtoDataType(def)),
                             name + "_placeholder");
      }
      inputs.push_back(tvm_tensors[name]);
    }

    // call OpIBuilder's Evaluate to build tvm IR
    op_ir_builder->Evaluate(inputs, node, demo_codegen_ctx, outputs);

    // Store outputs
    for (size_t def_id = 0; def_id < node.OutputDefs().size(); ++def_id) {
      const NodeArg* def = node.OutputDefs()[def_id];
      tvm_tensors[def->Name()] = outputs[def_id];
    }
  }

  // put inputs to DemoTVMTensorCtx
  for (auto& input : graph.GetInputs()) {
    result.inputs.push_back(tvm_tensors[input->Name()]);
  }

  // check initializer
  for (auto& initializer : graph.GetAllInitializedTensors()) {
    result.inputs.push_back(tvm_tensors[initializer.first]);
  }

  // Only one output in this demo
  auto& output = graph.GetOutputs()[0];
  result.outputs.push_back(tvm_tensors[output->Name()]);
  return result;
}

// Declare a Demo scheduler that always inserts compute_inline
DECLARE_TVM_SCHEDULER_CLASS(AlwaysInline, DemoTVM)

// Define a Demo scheduler's Evaluate that always inserts compute_inline
bool TVM_SCHEDULER_CLASS(AlwaysInline, DemoTVM)::Evaluate(
    const tvm::Tensor& tensor,
    const Node*,
    tvm_codegen::CodeGenContext&,
    tvm_codegen::ScheduleContext& ctx_sched) {
  return TryInlineSchedule(tensor, ctx_sched);
}

// Register the always inline Scheduler to sched_registry
static void RegisterAlwaysInlineScheduler(tvm_codegen::TVMScheduleRegistry* sched_registry) {
  sched_registry->Register(
      std::make_unique<TVM_SCHEDULER_CLASS(AlwaysInline, DemoTVM)>());
}

// Declare a schedule dispatcher that always dispatches the always inline Scheduler
DECLARE_SCHEDULE_DISPATCHER_CLASS(DemoTVM)

// Use a predefined key as DemoKey to dispatch the scheduler
constexpr auto predefined_key = "DemoKey";

// Define the schedule dispatcher's Find function
// that always dispatches the always inline Scheduler
// Note this dispatcher always returning a predefined_key is only for demo purpose.
// In practice, a dispatcher returns a key by checking tvm::Tensor, Node,
// or even meta data stored in CodeGenContext.
// Derived CodeGenContext allows compiler developers to store their specific meta data.
// For more detailed example, please check nuphar provider.
tvm_codegen::Scheduler* SCHEDULE_DISPATCHER_CLASS(DemoTVM)::Find(
    const tvm::Tensor&, const Node*, tvm_codegen::CodeGenContext&) {
  return DispatcherBase::Get(predefined_key);
}

// Attach the always inline Scheduler to the above dispatcher
// and then attach the dispatcher to the scheduler builder
static void AttachAlwaysInlineScheduler(const std::shared_ptr<tvm_codegen::TVMScheduleBuilder>& builder,
                                        const tvm_codegen::TVMScheduleRegistry* registry) {
  auto dispatcher = std::make_unique<SCHEDULE_DISPATCHER_CLASS(DemoTVM)>("DemoSchedulers");

  // Using a predefined_key
  dispatcher->Register(predefined_key,
                       registry->Get(TVM_SCHEDULER_STRING(AlwaysInline, DemoTVM)));

  builder->InsertDispatcher(std::move(dispatcher));
}

// Traverse tvm::Tensor and then schedule them
// Note this traversal is simplified and specific for this demo.
// For a more general traversal, please check nuphar provider.
static void TraverseAndSchedule(
    std::shared_ptr<tvm_codegen::TVMScheduleBuilder>& schedule_builder,
    const tvm::Tensor& tensor,
    tvm_codegen::ScheduleContext& ctx_schedule) {
  schedule_builder->Evaluate(tensor, nullptr, demo_codegen_ctx, ctx_schedule);

  // Traverse tensor's children (inputs)
  for (auto& t : tensor->op->InputTensors()) {
    // check whether it is a non-trivial tensor by checking its input size
    if (t->op->InputTensors().size() > 0) {
      TraverseAndSchedule(schedule_builder, t, ctx_schedule);
    }
  }
}

// Create a TVM schedule by always inserting tvm's compute_inline.
// Note this schedule is specific for this demo.
// In practice, always inline might lead to bad performance
// or even illegal loop transformation for some backends.
// For a more general example, please check nuphar provider.
tvm::Schedule CreateSchedule(const DemoTVMTensorCtx& ctx) {
  // Create TVMScheduleRegistry that holds all Scheduler
  std::unique_ptr<tvm_codegen::TVMScheduleRegistry> schedule_registry =
      std::make_unique<tvm_codegen::TVMScheduleRegistry>();

  // Register the always inline Scheduler to schedule_registry
  RegisterAlwaysInlineScheduler(schedule_registry.get());

  // Create a DemoScheduleBuilder
  std::shared_ptr<tvm_codegen::TVMScheduleBuilder> schedule_builder =
      std::make_shared<tvm_codegen::TVMScheduleBuilder>("Demo_Schedule_Builder");

  // Attach the demo inline scheduler to the schedule_builder
  AttachAlwaysInlineScheduler(schedule_builder, schedule_registry.get());

  // Create scheudule object
  tvm::Array<tvm::Operation> out_ops;
  for (auto& t : ctx.outputs) {
    out_ops.push_back(t->op);
  }

  // Create scheudule context
  tvm_codegen::ScheduleContext ctx_schedule(out_ops);

  // Traverse tvm::Tensor in a DFS way, and then schedule
  for (auto& t : ctx.outputs) {
    TraverseAndSchedule(schedule_builder, t, ctx_schedule);
  }

  // Make sure all outputs compute_root (tvm's requirement)
  for (auto& t : ctx.outputs) {
    tvm_codegen::InsertRootSchedule(t, ctx_schedule);
  }

  return ctx_schedule.schedule;
}

// Build TVM Module with a schedule using tvm's stackvm.
// Note in real practice, please change stackvm to other backends.
// For a more detailed example, please check nuphar provider.
tvm::runtime::Module BuildStackVMModule(tvm::Schedule schedule,
                                        tvm::BuildConfig config,
                                        tvm::Array<tvm::Tensor> tvm_args,
                                        std::vector<std::string>& target_func_names) {
  auto target = tvm::target::stackvm();
  std::string func_name = "func";
  auto args = tvm::Array<tvm::Tensor>(tvm_args);
  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
  auto lowered = lower(schedule, args, "func", binds, config);
  // Uncomment the following line to dump lowered func
  // std::cout << "Dumping lowered func: " << lowered[0]->body;
  target_func_names.push_back(func_name);
  return build(lowered, target, tvm::Target(), config);
}

}  // namespace tvm_demo
}  // namespace onnxruntime
