// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include <tvm/tvm.h>
#include <tvm/build_module.h>

namespace onnxruntime {
namespace tvm_demo {
// A Demo data structure to hold tvm IR and context
struct DemoTVMTensorCtx {
  tvm::Array<tvm::Tensor> inputs;
  tvm::Array<tvm::Tensor> outputs;
};

// Translate an Ort graph into tvm IR
DemoTVMTensorCtx BuildTVMIR(const onnxruntime::Graph& graph);

// Create a demo schedule for the tvm IR
tvm::Schedule CreateSchedule(const DemoTVMTensorCtx& ctx);

// Build a demo tvm module with the tvm IR and schedule
tvm::runtime::Module BuildStackVMModule(tvm::Schedule schedule,
                                        tvm::BuildConfig config,
                                        tvm::Array<tvm::Tensor> tvm_args,
                                        std::vector<std::string>& target_func_names);

}  // namespace tvm_demo
}  // namespace onnxruntime
