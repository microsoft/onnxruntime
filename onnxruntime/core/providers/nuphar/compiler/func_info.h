// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"

#include <tvm/tvm.h>
#include <tvm/build_module.h>
#include <type_traits>

namespace onnxruntime {
namespace nuphar {

enum class ControlFlowInfoType : unsigned int {
  Scan = 1,
};

// abstract class for control flow info
struct ControlFlowInfo {
 private:
  ControlFlowInfoType type;

 public:
  ControlFlowInfo(ControlFlowInfoType _type) : type(_type) {}

  virtual ~ControlFlowInfo() = default;

  DYN_PROMOTE_BASE(ControlFlowInfo, ControlFlowInfoType, type)
};

// Add Promote support for ControlFlowInfo
// Note here we need to use DYN_PROMOTE instread of DYNAMIC_PROMOTE
// since ControlFlowInfo is a critical path
DYN_PROMOTE(ControlFlowInfo)

// NupharFuncInfo holds tvm::runtime::PackedFunc (the generated function)
// And corresponding static meta information to call it, like number of argument and offset
// Note NupharFuncInfo includes ONLY parameters from codegen
// but DOES NOT include any runtime information.

// The owner of NupharFuncInfo is currently NupharKernelState.
// NupharFuncInfo is created in NupharCompiler and is consumed by ExecBlock
// Note all of vectors use numbers of PackedFunc's parameters as vector bounds
// (meaning vector.size() == numbers of PackedFunc's parameters)
// except those denoted with ort, which use numbers of ort op's parameters as vector bounds.
// -1 might be inserted a bubble to keep positions and sizes for later lookup.
struct NupharFuncInfo {
  // speicial value for *_func_indices
  enum : int {
    Index_NonAliasedOutput = -1,
  };

  // PackedFunc name
  std::string name;

  // PackedFunc
  tvm::runtime::PackedFunc packed_func;

  // TVM DLDevice
  DLDeviceType device_type;

  struct FuncArgMeta {
    MLDataType dtype;
    // shapes with dimensions statically know or inferred at compile time
    // symbolic dim would have Dimension_Unknown and will be patched at runtime
    std::vector<int64_t> inferred_shape;
    std::vector<std::pair<size_t, std::string>> dim_symbols;
    int ort_arg_index;
  };

  std::vector<FuncArgMeta> input_metas;
  std::vector<FuncArgMeta> output_metas;
  std::vector<std::pair<int, size_t>> ort_aliased_output_to_func_indices;  // A pair of (Ort dst index, TVM src index)

  struct AllocatorMeta {
    int index;
    bool is_external;
  };

  std::vector<AllocatorMeta> ort_input_allocators;
  std::vector<AllocatorMeta> ort_output_allocators;

  // Note an input can be also an external output.
  // It is due to NodeArg can be used by Nodes in
  // and out of a subgraph at the same time.
  // When it happens, we need to label it as a collided output,
  // and record that external output allocator index.
  std::vector<bool> ort_input_allocator_is_collided_output;

  // initializers meta
  std::vector<const Tensor*> intializers;

  // Note the total arg number == input_count + output_count
  size_t func_input_count;   // input_count == real inputs + initializers
  size_t func_output_count;  // real outputs

  // tvm args (including input and outputs )
  std::vector<int> type_codes;

  // control-flow info for the generated function
  std::unique_ptr<ControlFlowInfo> cf_info;

  size_t ort_input_count;
  size_t ort_output_count;
};

void FillNupharFuncInfo(NupharFuncInfo* func_info,
                        nuphar::OrtSubgraphAllocationInfo* partition_info,
                        const nuphar::NupharSubgraphUnit& subgraph,
                        const NupharCodeGenCtx& codegen_ctx,
                        tvm::Target tvm_target,
                        tvm::runtime::PackedFunc packed_func,
                        const std::string& name);

}  // namespace nuphar
}  // namespace onnxruntime
