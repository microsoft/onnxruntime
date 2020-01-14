// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"
#include "core/providers/nuphar/compiler/func_info.h"
#include "core/providers/nuphar/compiler/initializer_info.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/nuphar_handle.h"
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "gsl/gsl"

#include <algorithm>
#include <tvm/build_module.h>

namespace onnxruntime {
namespace nuphar {

class NupharCompiler {
 public:
  NupharCompiler(const Node& node,
                 const std::map<std::string, const Tensor*>& initializers,
                 std::unordered_map<std::string, std::unique_ptr<Tensor>>& generated_initializers,
                 const NupharCodeGenHandle* handle);

  NupharCompiler(const nuphar::NupharSubgraphUnit& subgraph,
                 std::unordered_map<std::string, std::unique_ptr<Tensor>>& generated_initializers,
                 const NupharCodeGenHandle* handle);

  // Build builds tvm IR and apply passes
  Status Build(const nuphar::NupharSubgraphUnit& subgraph);

  // Lower lowers the built tvm IR to llvm ir and compiles it
  Status Lower(const nuphar::NupharSubgraphUnit& subgraph,
               tvm::Target tvm_target,
               tvm::Target tvm_host_target,
               NupharFuncInfo* ctx_func,
               nuphar::OrtSubgraphAllocationInfo* partition_info);

  tvm::runtime::PackedFunc GetLoweredPackedFunc(
      const std::string& func_name,
      tvm::Target tvm_target,
      tvm::Target tvm_host_target,
      const tvm::BuildConfig& config,
      const std::string& subgraph_type,
      const std::string& subgraph_name);

 private:
  size_t num_initializers_in_graph_inputs_;

  // BuildSubgraph builds tvm IR and apply passes for a subgraph
  Status BuildSubgraph(const Node& node);

  NupharCodeGenCtx context_;

  tvm::Array<tvm::Tensor> tvm_args_;
  tvm::Array<tvm::Tensor> tvm_outputs_;
};

}  // namespace nuphar
}  // namespace onnxruntime
