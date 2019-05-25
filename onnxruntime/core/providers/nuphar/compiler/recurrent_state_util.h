// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/target/tvm_context.h"

namespace onnxruntime {
namespace tvm_codegen {

// TODO: change location to loop-related place
// A pair of parameters of State, like (h_in, h_out) or (c_in, c_out)
using LoopState = std::vector<std::pair<tvm::Tensor, tvm::Tensor>>;

void CreateRecurrentStates(const Node& node,
                           LoopState& loop_states);

}  // namespace tvm_codegen
}  // namespace onnxruntime
