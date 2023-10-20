#pragma once

#include "core/framework/func_api.h"

namespace onnxruntime {
// if we are export the fused function to dll, the function will still in the same binary as onnxruntime
// use std function to give execution provider some chance to capture some state.
using CreateFunctionStateFunc = std::function<int(ComputeContext*, FunctionState*)>;
using ComputeFunc = std::function<Status(FunctionState, const OrtApi*, OrtKernelContext*)>;
using DestroyFunctionStateFunc = std::function<void(FunctionState)>;

struct NodeComputeInfo {
  CreateFunctionStateFunc create_state_func;
  ComputeFunc compute_func;
  DestroyFunctionStateFunc release_state_func;
};
}
