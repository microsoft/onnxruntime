// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <algorithm>
#include <tvm/build_module.h>
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/compiler/tvm_initializer.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/nuphar_func_ctx.h"
#include "core/providers/nuphar/compiler/nuphar_handle.h"
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "gsl/gsl_util"

namespace onnxruntime {
namespace tvm_codegen {

// TODO: Add comments of TVMCompiler after refactoring it
class TVMCompiler {
 public:
  TVMCompiler(const Node& node,
              InitializerMap& initializer_lut,
              const NupharCodeGenHandle* handle);

  // Build builds tvm IR and apply passes
  Status Build();

  // BuildSubgraph builds tvm IR and apply passes for a subgraph
  Status BuildSubgraph();

  // Lower lowers the built tvm IR to llvm ir and compiles it
  Status Lower(tvm::Target tvm_target,
               tvm::Target tvm_host_target,
               NupharFuncInfo* ctx_func);

  // A temp port to let runtime access Node
  // TODO: remove this after runtime refactoring
  const Node& GetNode() {
    return node_;
  }

  // A temp port to let runtime access CodeGenContext
  // TODO: remove this after runtime refactoring
  const NupharCodeGenCtx& GetCodeGenContext() {
    return context_;
  }

 private:
  // move to another place
  void GetStateTensors(tvm::Array<tvm::Tensor>& in_state_tensors,
                       tvm::Array<tvm::Tensor>& out_state_tensors);

  const Node& node_;

  size_t num_initializers_in_graph_inputs_;

  NupharCodeGenCtx context_;

  tvm::Array<tvm::Tensor> tvm_args_;
  tvm::Array<tvm::Tensor> tvm_outputs_;

  // runtime related code
  // TODO pleass remove it
  size_t num_actual_outputs_;
  enum : int {
    OutputAliased = -1,
  };
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
