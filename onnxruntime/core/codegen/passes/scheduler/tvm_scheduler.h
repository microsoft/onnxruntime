// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/codegen/common/creator.h"
#include "core/codegen/common/registry.h"
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/graph/graph.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

// These are current generic TVMOpRule we used.
enum class TVMOpRuleType : int {
  Extern = 0,
  ComputeReduce = 1,
  ComputeRegular = 2,
  AlwaysRoot = 3,  // for debug
  NoRule,
};

const std::string& GetTVMOpRule(const tvm::Tensor& tensor);
const std::string& GetTVMOpRule(TVMOpRuleType rule);

// These are current generic ScheduleType in tvm_codegen
enum class ScheduleType : int {
  ScheduleNone = 0,
  ScheduleInline = 1,
  ScheduleAt = 2,
  ScheduleRoot = 3,
  ScheduleClosure = 4,
};

// Data struct to bundle tvm::Schedule and scheduled tensor
struct ScheduleContext {
  ScheduleContext(const tvm::Array<tvm::Operation>& ops) {
    schedule = tvm::create_schedule(ops);
  }
  tvm::Schedule schedule;
  std::map<const tvm::Node*, ScheduleType> scheduled_tensors;
};

// Scheduler inserts a tvm::Schedule content to a tvm::Tensor
using Scheduler = codegen::CreatorBase<
    const tvm::Tensor&,
    const Node*,
    tvm_codegen::CodeGenContext&,
    ScheduleContext&,
    bool>;

// TVMScheduleDispatcher is the base dispatcher for TVM Schedule Builder
// It checks whether a pair of {tvm::Tensor, Ort Node} satisfying a criteria (in Find)
// and dispatches a corresponding Scheduler.
class TVMScheduleDispatcher : public codegen::DispatcherBase<Scheduler*> {
 public:
  TVMScheduleDispatcher(const std::string& name)
      : DispatcherBase(name) {}

  virtual ~TVMScheduleDispatcher() = default;

  virtual Scheduler* Find(const tvm::Tensor&,
                          const Node*,
                          tvm_codegen::CodeGenContext&) = 0;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TVMScheduleDispatcher);
};

// Macro returns an Schedulers' dispatcher's name
#define SCHEDULE_DISPATCHER_CLASS(TYPE) \
  TVM##TYPE##Schedulers

// Macro declares an Schedulers' dispatcher
#define DECLARE_SCHEDULE_DISPATCHER_CLASS(TYPE)                                       \
  class SCHEDULE_DISPATCHER_CLASS(TYPE) : public tvm_codegen::TVMScheduleDispatcher { \
   public:                                                                            \
    TVM##TYPE##Schedulers(const std::string& name)                                    \
        : TVMScheduleDispatcher(name) {}                                              \
    ~TVM##TYPE##Schedulers() = default;                                               \
    tvm_codegen::Scheduler* Find(const tvm::Tensor&,                                  \
                                 const Node*,                                         \
                                 tvm_codegen::CodeGenContext&) override;              \
                                                                                      \
   private:                                                                           \
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TVM##TYPE##Schedulers);                     \
  };

// Common dispatchers are listed here
// For a special pattern, it can be created later.
// One dispatcher is based on Ort OpType
DECLARE_SCHEDULE_DISPATCHER_CLASS(OrtOpType)
// One dispatcher is based on TVMOpRule
DECLARE_SCHEDULE_DISPATCHER_CLASS(TVMOpRule)
// One dispatcher is based Ort NodeArg name
DECLARE_SCHEDULE_DISPATCHER_CLASS(OrtOpName)

// Scheduler Registry is a registry holds all Schedulers
using TVMScheduleRegistry = codegen::RegistryBase<Scheduler>;

// Macro declares TVM scheduler class
#define DECLARE_TVM_SCHEDULER_CLASS(OP, PRETFIX)       \
  DECLARE_CREATOR_CLASS(OP, PRETFIX##Scheduler,        \
                        const tvm::Tensor&,            \
                        const Node*,                   \
                        tvm_codegen::CodeGenContext&,  \
                        tvm_codegen::ScheduleContext&, \
                        bool)

// Macro returns TVM scheduler's name with prefix
#define TVM_SCHEDULER_CLASS(OP, PREFIX) \
  CREATOR_CLASS(OP, PREFIX##Scheduler)

// Macro returns TVM scheduler's name as string
#define TVM_SCHEDULER_STRING(OP, PREFIX) \
  STRINGIZE(TVM_SCHEDULER_CLASS(OP, PREFIX))

// Macro returns TVM scheduler's name with prefix and arch
#define TVM_SCHEDULER_CLASS_EX(OP, PREFIX, ARCH) \
  CREATOR_CLASS(OP, PREFIX##ARCH##Scheduler)

// Macro declares TVM scheduler class with prefix and arch
#define DECLARE_TVM_SCHEDULER_CLASS_EX(OP, PREFIX, ARCH) \
  DECLARE_TVM_SCHEDULER_CLASS(OP, PREFIX##ARCH)

}  // namespace tvm_codegen
}  // namespace onnxruntime
