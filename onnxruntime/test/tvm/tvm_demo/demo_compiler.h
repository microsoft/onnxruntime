// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include <tvm/te/tensor.h>
#include <tvm/te/schedule.h>
#include <tvm/runtime/module.h>

namespace onnxruntime {
namespace tvm_demo {
// A Demo data structure to hold tvm IR and context
struct DemoTVMTensorCtx {
  tvm::Array<tvm::te::Tensor> inputs;
  tvm::Array<tvm::te::Tensor> outputs;
};

// Translate an Ort graph into tvm IR
DemoTVMTensorCtx BuildTVMIR(const onnxruntime::Graph& graph);

// Create a demo schedule for the tvm IR
tvm::te::Schedule CreateSchedule(const DemoTVMTensorCtx& ctx);

// Build a demo tvm module with the tvm IR and schedule
tvm::runtime::Module BuildStackVMModule(tvm::te::Schedule schedule,
                                        tvm::Array<tvm::te::Tensor> tvm_args,
                                        std::vector<std::string>& target_func_names);

}  // namespace tvm_demo
}  // namespace onnxruntime
