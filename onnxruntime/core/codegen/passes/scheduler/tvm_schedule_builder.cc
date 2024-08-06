// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/scheduler/tvm_schedule_builder.h"

#include "core/codegen/common/op_macro.h"
#include "core/codegen/common/settings.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace tvm_codegen {

TVMScheduleBuilder::TVMScheduleBuilder(const std::string& name)
    : name_(name) {
}

const std::string& TVMScheduleBuilder::Name() const {
  return name_;
}

void TVMScheduleBuilder::InsertDispatcher(std::unique_ptr<TVMScheduleDispatcher>&& ptr) {
  dispatchers_.push_back(std::move(ptr));
}

void TVMScheduleBuilder::ClearDispatcher() {
  dispatchers_.clear();
}

void TVMScheduleBuilder::DumpAllSchedulers() const {
  std::ostringstream stream;
  int count = 0;
  stream << "[CODEGEN_DUMP_SCHEDULE]" << std::endl;
  for (auto& d : dispatchers_) {
    stream << "************ TVM Scheduler Dispatcher "
           << count << " : "
           << d->Name()
           << " ************" << std::endl;

    d->ForEach([&stream](const std::string& key, Scheduler* op) {
      stream << "Key " << key
             << ", Creator " << op->Name() << std::endl;
    });

    ++count;
  }

  LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << stream.str();
}

Status TVMScheduleBuilder::Evaluate(
    const tvm::Tensor& tensor,
    const Node* node,
    CodeGenContext& ctx_codegen,
    ScheduleContext& sched) {
  Scheduler* candidate = nullptr;

  for (auto& d : dispatchers_) {
    candidate = d->Find(tensor, node, ctx_codegen);
    if (nullptr != candidate)
      break;
  }

  bool enable_dump_schedule = codegen::CodeGenSettings::Instance().HasOption(codegen::CodeGenSettings::kCodeGenDumpSchedule);

  if (nullptr == candidate) {
    if (nullptr != node)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Not implemented: ", node->OpType());
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Not implemented an internal tvm::Tensor: ", tensor->op->name);
  }

  bool status = candidate->Evaluate(tensor, node, ctx_codegen, sched);

  if (enable_dump_schedule) {
    std::ostringstream stream;
    if (nullptr != node) {
      stream << std::endl;
      stream << "[CODEGEN_DUMP_SCHEDULE] "
             << "Schedule Node: " << node->Name() << std::endl;
    } else {
      stream << std::endl;
    }

    if (status) {
      stream << "[CODEGEN_DUMP_SCHEDULE] "
             << "Schedule tvm::Tesnor "
             << tensor->op->name
             << " with "
             << candidate->Name() << std::endl;
    } else {
      stream << "[CODEGEN_DUMP_SCHEDULE] "
             << "Schedule tvm::Tesnor "
             << tensor->op->name
             << " is suppressed " << std::endl;
    }

    LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << stream.str();
  }

  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
