// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"

#include "core/providers/nuphar/common/nuphar_subgraph.h"

namespace onnxruntime {
namespace nuphar {

// CreateTVMIR function traverses a GraphViewer
// and builds tvm ir (and store them in CodeGenContext)
// based on corresponding ORT ir
Status CreateTVMIR(const GraphViewer& graph,
                   NupharCodeGenCtx& ctx_codegen,
                   bool use_placeholder_for_input);

// CreateTVMIR function traverses a single node
// and builds tvm ir (and store them in CodeGenContext)
// based on corresponding ORT ir
Status CreateTVMIR(const Node& node,
                   NupharCodeGenCtx& ctx_codegen);

// CreateTVMIR function traverses a NupharSubgraphUnit
// and builds tvm ir (and store them in CodeGenContext)
// based on corresponding ORT ir
Status CreateTVMIR(const nuphar::NupharSubgraphUnit& subgraph,
                   NupharCodeGenCtx& ctx_codegen);

}  // namespace nuphar
}  // namespace onnxruntime
