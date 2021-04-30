// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/func_info.h"

#include "core/providers/nuphar/runtime/control_flow/scan_exec_ctx.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"
#include <unordered_map>

namespace onnxruntime {
namespace nuphar {

static void FillBasicFuncInfo(NupharFuncInfo* func_info,
                              nuphar::OrtSubgraphAllocationInfo* partition_info,
                              const nuphar::NupharSubgraphUnit& subgraph,
                              const NupharCodeGenCtx& codegen_ctx,
                              tvm::Target tvm_target,
                              tvm::runtime::PackedFunc packed_func,
                              const std::string& name) {
  ORT_ENFORCE(nullptr != func_info);
  ORT_ENFORCE(nullptr != partition_info);

  func_info->name = name;
  func_info->packed_func = packed_func;
  func_info->device_type = static_cast<DLDeviceType>(tvm_target->device_type);

  int tvm_input_idx = 0;
  int def_index = 0;
  // Handle inputs
  func_info->ort_input_count = subgraph.inputs.size();
  // Assign Input meta
  for (auto& def : subgraph.inputs) {
    // fill in allocator info
    NupharFuncInfo::AllocatorMeta input_allocator;
    if (partition_info->inputs.count(def->Name()) > 0) {
      // if an input is from external
      input_allocator.index = partition_info->inputs.at(def->Name());
      input_allocator.is_external = true;
      func_info->ort_input_allocator_is_collided_output.push_back(false);
    } else if (partition_info->outputs.count(def->Name()) > 0) {
      // if an input is from a previous real output
      input_allocator.index = partition_info->outputs.at(def->Name());
      input_allocator.is_external = true;  // a real output is always from external
      func_info->ort_input_allocator_is_collided_output.push_back(true);
    } else {
      // else, an input is from an internal allocator
      input_allocator.index = partition_info->CreateOrGetInternalAllocatorOffset(def->Name());
      input_allocator.is_external = false;
      func_info->ort_input_allocator_is_collided_output.push_back(false);
    }

    func_info->ort_input_allocators.push_back(input_allocator);

    if (codegen_ctx.IsInitializer(def->Name())) {
      ++def_index;
      continue;  // skip initializers
    }

    // fill in func args
    NupharFuncInfo::FuncArgMeta input_meta;
    input_meta.dtype = OrtTypeInfo::ElementTypeFromProto(def->TypeAsProto()->tensor_type().elem_type());
    input_meta.ort_arg_index = def_index;

    // fill in shape info and symobolic info
    for (int dim = 0; dim < gsl::narrow<int>(ShapeRank(def)); ++dim) {
      if (ShapeHasSymbol(def, dim)) {
        input_meta.inferred_shape.push_back(Dimension_Unknown);
        input_meta.dim_symbols.push_back(std::make_pair(gsl::narrow<size_t>(dim), ShapeSymbol(def, dim)));
      } else if (ShapeHasValue(def, dim)) {
        input_meta.inferred_shape.push_back(ShapeValue(def, dim));
      } else {
        input_meta.inferred_shape.push_back(Dimension_Unknown);
      }
    }

    func_info->input_metas.push_back(input_meta);

    ++tvm_input_idx;
    ++def_index;
  }

  // Handle initializers
  // Initializer meta
  std::vector<const Tensor*>& intializers = func_info->intializers;
  // Assign Initializer meta
  for (const auto& item : codegen_ctx.GetWeightLayoutMap()) {
    const WeightLayoutCodegenInfo* layout_info = item.second.get();
    bool is_marshalled = layout_info->is_marshalled;
    const Tensor* t =
        is_marshalled ? layout_info->marshalled_initializer
                      : codegen_ctx.GetOrtInitializerTensor(item.first);

    intializers.push_back(t);
    ++tvm_input_idx;
  }

  // set input_count = the number of inputs + the number of initializers
  func_info->func_input_count = gsl::narrow<size_t>(tvm_input_idx);

  // Handle Outputs

  func_info->ort_output_count = subgraph.outputs.size();
  // Assign Output meta
  int tvm_output_idx = 0;
  std::unordered_map<NodeKey, size_t> visited_output_def_indices;
  def_index = 0;
  for (auto& def : subgraph.outputs) {
    // fill in allocator info
    NupharFuncInfo::AllocatorMeta output_allocator;
    if (partition_info->outputs.count(def->Name()) > 0) {
      // if an output is from external
      output_allocator.index = partition_info->outputs.at(def->Name());
      output_allocator.is_external = true;
    } else {
      // else, an output is from an internal allocator
      output_allocator.index = partition_info->CreateOrGetInternalAllocatorOffset(def->Name());
      output_allocator.is_external = false;
    }

    func_info->ort_output_allocators.push_back(output_allocator);

    // check output alias
    const NodeArg* source_def = Promote<CodeGenUnitStats>(codegen_ctx.GetGraphStats())
                                    ->SourceDefOfOutputAlias(def);

    if (nullptr != source_def) {
      // if def is an alias
      auto key = GetKey(source_def);
      if (visited_output_def_indices.count(key) != 0) {
        // source_def has visisted ==> def is a duplicated output
        // record the pair (dst of ort arg index, src of tvm func index)
        func_info->ort_aliased_output_to_func_indices.emplace_back(def_index,
                                                                   func_info->func_input_count +
                                                                       visited_output_def_indices[key]);

        ++def_index;
        continue;
      }
      // update visited_output_def_indices
      visited_output_def_indices.insert(std::make_pair(key, gsl::narrow_cast<size_t>(tvm_output_idx)));
    } else {
      auto key = GetKey(def);
      if (visited_output_def_indices.count(key) != 0) {
        // def has visisted ==> def is a duplicated output
        // record the pair (dst of ort arg index, src of tvm func index)
        func_info->ort_aliased_output_to_func_indices.emplace_back(def_index,
                                                                   func_info->func_input_count +
                                                                       visited_output_def_indices[key]);

        ++def_index;
        continue;
      }
      visited_output_def_indices.insert(std::make_pair(key, gsl::narrow_cast<size_t>(tvm_output_idx)));
    }

    NupharFuncInfo::FuncArgMeta output_meta;
    output_meta.dtype = OrtTypeInfo::ElementTypeFromProto(def->TypeAsProto()->tensor_type().elem_type());
    output_meta.ort_arg_index = def_index;

    // fill in shape info and symobolic info
    for (int dim = 0; dim < gsl::narrow<int>(ShapeRank(def)); ++dim) {
      if (ShapeHasSymbol(def, dim)) {
        auto p = std::make_pair(gsl::narrow<size_t>(dim), ShapeSymbol(def, dim));
        output_meta.dim_symbols.push_back(p);
        output_meta.inferred_shape.push_back(Dimension_Unknown);
      } else if (ShapeHasValue(def, dim)) {
        output_meta.inferred_shape.push_back(ShapeValue(def, dim));
      } else {
        output_meta.inferred_shape.push_back(Dimension_Unknown);
      }
    }

    func_info->output_metas.push_back(output_meta);
    ++def_index;
    ++tvm_output_idx;
  }

  // set output_count as the real output count
  func_info->func_output_count = gsl::narrow_cast<size_t>(tvm_output_idx);

  // set tvm type_codes
  func_info->type_codes.resize(func_info->func_input_count + func_info->func_output_count, TVMTypeCode::kNDArrayContainer);
}

static void FillScanExecInfo(NupharFuncInfo* func_info,
                             nuphar::OrtSubgraphAllocationInfo* partition_info,
                             const Node& node,
                             const NupharCodeGenCtx& codegen_ctx,
                             tvm::Target tvm_target,
                             tvm::runtime::PackedFunc packed_func,
                             const std::string& name) {
  ORT_ENFORCE(nullptr != func_info);
  ORT_ENFORCE(nullptr != partition_info);

  // create Scan control-flow info
  auto scan_info = std::make_unique<ScanExecInfo>();

  int64_t num_state_variables;
  int64_t num_scan_inputs;
  int64_t num_scan_outputs;

  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  // extract num_scan_inputs
  bool attr_is_ok = attrs.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs).IsOK();
  ORT_UNUSED_PARAMETER(attr_is_ok);
  ORT_ENFORCE_DEBUG(attr_is_ok);

  auto subgraph = GetSubgraph(node);
  ORT_ENFORCE(subgraph != nullptr);
  size_t num_variadic_inputs = subgraph->GetInputs().size();
  size_t num_variadic_outputs = subgraph->GetOutputs().size();

  num_state_variables = gsl::narrow<int64_t>(num_variadic_inputs) - num_scan_inputs;
  num_scan_outputs = gsl::narrow<int64_t>(num_variadic_outputs) - num_state_variables;

  // Set ScanExecInfo's parameter count meta
  scan_info->num_state_variables = num_state_variables;
  scan_info->num_scan_inputs = num_scan_inputs;
  scan_info->num_scan_outputs = num_scan_outputs;
  scan_info->num_scan_implicit_inputs = gsl::narrow_cast<int64_t>(node.ImplicitInputDefs().size());

  // ScanExecInfo's control flow Meta
  std::vector<bool>& scan_input_forwards = scan_info->scan_input_forwards;
  std::vector<bool>& scan_output_forwards = scan_info->scan_output_forwards;
  std::vector<int64_t>& scan_input_axes = scan_info->scan_input_axes;
  std::vector<int64_t>& scan_output_axes = scan_info->scan_output_axes;

  scan_input_forwards.resize(num_scan_inputs);
  scan_output_forwards.resize(num_scan_outputs);

  // extract directions and axes
  std::vector<int64_t> scan_input_directions;
  std::vector<int64_t> scan_output_directions;

  // scan_input_directions
  if (attrs.GetAttrs<int64_t>("scan_input_directions", scan_input_directions).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(scan_input_directions.size()) == num_scan_inputs,
                "Number of entries in 'scan_input_directions ' was ", scan_input_directions.size(),
                ". Must match 'num_scan_inputs' of ", num_scan_inputs);
    ORT_ENFORCE(std::all_of(scan_input_directions.cbegin(), scan_input_directions.cend(),
                            [](int64_t i) { return i == 0 ||
                                                   i == 1; }),
                "Invalid values in 'scan_input_directions'. 0 == forward. 1 == reverse.");
  } else {
    // default to forward
    scan_input_directions = std::vector<int64_t>(num_scan_inputs, 0);
  }

  // scan_input_forwards
  for (size_t i = 0; i < gsl::narrow<size_t>(num_scan_inputs); ++i) {
    scan_input_forwards[i] = scan_input_directions[i] == 0;
  }

  // scan_output_directions
  if (attrs.GetAttrs<int64_t>("scan_output_directions", scan_output_directions).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(scan_output_directions.size()) == num_scan_outputs,
                "Number of entries in 'scan_output_directions ' was ", scan_output_directions.size(),
                ". Must match 'num_scan_outputs' of ", num_scan_outputs);
    ORT_ENFORCE(std::all_of(scan_output_directions.cbegin(), scan_output_directions.cend(),
                            [](int64_t i) { return i == 0 ||
                                                   i == 1; }),
                "Invalid values in 'scan_output_directions'. 0 == forward. 1 == reverse.");
  } else {
    // default to forward
    scan_output_directions = std::vector<int64_t>(num_scan_outputs, 0);
  }

  // scan_output_forwards
  for (size_t i = 0; i < gsl::narrow<size_t>(num_scan_outputs); ++i) {
    scan_output_forwards[i] = scan_output_directions[i] == 0;
  }

  // scan_input_axes
  if (attrs.GetAttrs<int64_t>("scan_input_axes", scan_input_axes).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(scan_input_axes.size()) == num_scan_inputs,
                "Number of entries in 'scan_input_axes ' was ", scan_input_axes.size(),
                ". Must match 'num_scan_inputs' of ", num_scan_inputs);

  } else {
    // default to axis 0
    scan_input_axes = std::vector<int64_t>(num_scan_inputs, 0);
  }

  // scan_output_axes
  if (attrs.GetAttrs<int64_t>("scan_output_axes", scan_output_axes).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(scan_output_axes.size()) == num_scan_outputs,
                "Number of entries in 'scan_output_axes ' was ", scan_output_axes.size(),
                ". Must match 'num_scan_outputs' of ", num_scan_outputs);

  } else {
    // default to axis 0
    scan_output_axes = std::vector<int64_t>(num_scan_outputs, 0);
  }

  // handle NupharFuncInfo
  func_info->name = name;
  func_info->packed_func = packed_func;
  func_info->device_type = static_cast<DLDeviceType>(tvm_target->device_type);

  int tvm_input_idx = 0;
  // Handle state inputs & inputs
  func_info->ort_input_count = num_variadic_inputs;

  // assign state inputs & inputs
  for (size_t ort_input_idx = 0; ort_input_idx < num_variadic_inputs; ++ort_input_idx) {
    // fill in allocator info
    NupharFuncInfo::AllocatorMeta input_allocator;
    const NodeArg* main_graph_def = node.InputDefs()[ort_input_idx];
    ORT_ENFORCE(nullptr != main_graph_def);
    if (partition_info->inputs.count(main_graph_def->Name()) > 0) {
      // if an input is from external
      input_allocator.index = partition_info->inputs.at(main_graph_def->Name());
      input_allocator.is_external = true;
      func_info->ort_input_allocator_is_collided_output.push_back(false);
    } else if (partition_info->outputs.count(main_graph_def->Name()) > 0) {
      // if an input is from a previous real output
      input_allocator.index = partition_info->outputs.at(main_graph_def->Name());
      input_allocator.is_external = true;  // a real output is always from external
      func_info->ort_input_allocator_is_collided_output.push_back(true);
    } else {
      // else, an input is from an internal allocator
      input_allocator.index = partition_info->CreateOrGetInternalAllocatorOffset(main_graph_def->Name());
      input_allocator.is_external = false;
      func_info->ort_input_allocator_is_collided_output.push_back(false);
    }

    func_info->ort_input_allocators.push_back(input_allocator);

    const NodeArg* def = subgraph->GetInputs()[ort_input_idx];
    ORT_ENFORCE(nullptr != def);

    if (ort_input_idx >= gsl::narrow<size_t>(num_state_variables)) {
      // initializer should only happen in real inputs, not in state inputs
      if (codegen_ctx.IsInitializer(def->Name())) {
        continue;  // skip initializers
      }
    }

    NupharFuncInfo::FuncArgMeta input_meta;
    input_meta.dtype = OrtTypeInfo::ElementTypeFromProto(def->TypeAsProto()->tensor_type().elem_type());
    input_meta.ort_arg_index = gsl::narrow_cast<int>(ort_input_idx);

    // fill in shape info and symobolic info
    for (int dim = 0; dim < gsl::narrow<int>(ShapeRank(def)); ++dim) {
      if (ShapeHasSymbol(def, dim)) {
        auto p = std::make_pair(gsl::narrow<size_t>(dim), ShapeSymbol(def, dim));
        input_meta.dim_symbols.push_back(p);
        input_meta.inferred_shape.push_back(Dimension_Unknown);
      } else if (ShapeHasValue(def, dim)) {
        input_meta.inferred_shape.push_back(ShapeValue(def, dim));
      } else {
        input_meta.inferred_shape.push_back(Dimension_Unknown);
      }
    }

    func_info->input_metas.push_back(input_meta);
    ++tvm_input_idx;
  }

  size_t ort_input_idx = num_variadic_inputs;
  // Handle implicit inputs
  for (const NodeArg* def : node.ImplicitInputDefs()) {
    NupharFuncInfo::AllocatorMeta input_allocator;
    if (partition_info->inputs.count(def->Name()) > 0) {
      // if an input is from external
      input_allocator.index = partition_info->inputs.at(def->Name());
      input_allocator.is_external = true;
      func_info->ort_input_allocator_is_collided_output.push_back(false);
    } else if (partition_info->outputs.count(def->Name()) > 0) {
      // if an input is from a previous real output
      input_allocator.index = partition_info->outputs.at(def->Name());
      input_allocator.is_external = true;
      func_info->ort_input_allocator_is_collided_output.push_back(true);
    } else {
      // else, an input is from an internal allocator
      input_allocator.index = partition_info->CreateOrGetInternalAllocatorOffset(def->Name());
      input_allocator.is_external = false;
      func_info->ort_input_allocator_is_collided_output.push_back(false);
    }

    func_info->ort_input_allocators.push_back(input_allocator);

    // skip initializers
    if (codegen_ctx.IsInitializer(def->Name())) {
      ++ort_input_idx;
      continue;  // skip initializers
    }

    NupharFuncInfo::FuncArgMeta input_meta;
    input_meta.dtype = OrtTypeInfo::ElementTypeFromProto(def->TypeAsProto()->tensor_type().elem_type());
    input_meta.ort_arg_index = gsl::narrow_cast<int>(ort_input_idx);

    std::vector<std::pair<size_t, std::string>> symbols;
    for (int dim = 0; dim < gsl::narrow<int>(ShapeRank(def)); ++dim) {
      if (ShapeHasSymbol(def, dim)) {
        auto p = std::make_pair(gsl::narrow<size_t>(dim), ShapeSymbol(def, dim));
        input_meta.dim_symbols.push_back(p);
        input_meta.inferred_shape.push_back(Dimension_Unknown);
      } else if (ShapeHasValue(def, dim)) {
        input_meta.inferred_shape.push_back(ShapeValue(def, dim));
      } else {
        input_meta.inferred_shape.push_back(Dimension_Unknown);
      }
    }
    func_info->input_metas.push_back(input_meta);
    ++tvm_input_idx;
    ++ort_input_idx;
  }

  // Handle initializers
  // Initializer meta
  std::vector<const Tensor*>& intializers = func_info->intializers;

  // Assign Initializer meta
  for (const auto& item : codegen_ctx.GetWeightLayoutMap()) {
    const WeightLayoutCodegenInfo* layout_info = item.second.get();

    bool is_marshalled = layout_info->is_marshalled;
    const Tensor* t =
        is_marshalled ? layout_info->marshalled_initializer
                      : codegen_ctx.GetOrtInitializerTensor(item.first);

    intializers.push_back(t);
    ++tvm_input_idx;
  }

  // set input_count = the number of inputs (real inputs + state inputs) + the number of initializers
  func_info->func_input_count = gsl::narrow<size_t>(tvm_input_idx);

  // Handle State Outputs and Outputs
  func_info->ort_output_count = num_variadic_outputs;

  // Since in Scan, we only allow state using output's memory during Execution, not the other around.
  // When one input and one state are aliased, the kept one can only be the input.
  // Therefore, we do alias detection starting from inputs first.
  std::unordered_map<NodeKey, int> visited_output_def_indices;
  for (size_t ort_output_idx = gsl::narrow<size_t>(num_state_variables); ort_output_idx < num_variadic_outputs; ++ort_output_idx) {
    const NodeArg* def = subgraph->GetOutputs()[ort_output_idx];
    ORT_ENFORCE(nullptr != def);
    const NodeArg* source_def = Promote<CodeGenUnitStats>(codegen_ctx.GetGraphStats())
                                    ->SourceDefOfOutputAlias(def);
    if (nullptr != source_def) {
      auto key = GetKey(source_def);
      ORT_ENFORCE(visited_output_def_indices.count(key) == 0,
                  "Scan has alias btw two inputs. Nuphar only support aliasing btw state and output in Scan");
      visited_output_def_indices.insert(std::make_pair(key, gsl::narrow<int>(ort_output_idx)));
    } else {
      auto key = GetKey(def);
      visited_output_def_indices.insert(std::make_pair(key, gsl::narrow<int>(ort_output_idx)));
    }
  }

  // assign state outputs and outputs
  size_t tvm_output_idx = 0;
  std::unordered_map<NodeKey, int> visited_output_state_func_indices;
  for (size_t ort_output_idx = 0; ort_output_idx < num_variadic_outputs; ++ort_output_idx) {
    // fill in allocator info
    NupharFuncInfo::AllocatorMeta output_allocator;
    const NodeArg* main_graph_def = node.OutputDefs()[ort_output_idx];
    ORT_ENFORCE(nullptr != main_graph_def);
    if (partition_info->outputs.count(main_graph_def->Name()) > 0) {
      output_allocator.index = partition_info->outputs.at(main_graph_def->Name());
      output_allocator.is_external = true;
    } else {
      output_allocator.index = partition_info->CreateOrGetInternalAllocatorOffset(main_graph_def->Name());
      output_allocator.is_external = false;
    }
    func_info->ort_output_allocators.push_back(output_allocator);

    // perform alias analysis
    const NodeArg* def = subgraph->GetOutputs()[ort_output_idx];
    ORT_ENFORCE(nullptr != def);
    const NodeArg* source_def = Promote<CodeGenUnitStats>(codegen_ctx.GetGraphStats())
                                    ->SourceDefOfOutputAlias(def);

    // Determine alias btw output and state output
    auto key = source_def != nullptr ? GetKey(source_def) : GetKey(def);

    int ort_arg_index = gsl::narrow_cast<int>(ort_output_idx);
    if (ort_output_idx < gsl::narrow<size_t>(num_state_variables)) {
      auto key_iter = visited_output_def_indices.find(key);
      // if ort_output_idx is a state output
      if (key_iter != visited_output_def_indices.end()) {
        // If state output is an alias

        auto output_tvm_idx = key_iter->second - gsl::narrow_cast<int>(num_state_variables);

        // also record ort_aliased_output_to_func_indices
        func_info->ort_aliased_output_to_func_indices.push_back(
            std::make_pair(gsl::narrow<int>(ort_output_idx), func_info->func_input_count + output_tvm_idx));

        scan_info->state_to_output_indices.push_back(output_tvm_idx);

        if (visited_output_state_func_indices.count(key) != 0) {
          // We could have multiple states that alias to the same output.
          // We only record the first one and skip the rest.
          continue;
        } else {
          // record i_output for the lookup of the aliased output later
          visited_output_state_func_indices.insert(
              std::make_pair(key, gsl::narrow<int>(func_info->func_input_count + output_tvm_idx)));

          // override ort_arg_index using the output index
          ort_arg_index = visited_output_def_indices[key];
        }
      } else {
        // the state output not aliased(no scan output shares with it)
        scan_info->state_to_output_indices.push_back(NupharFuncInfo::Index_NonAliasedOutput);
      }
    } else {
      // if ort_output_idx is an output
      if (visited_output_state_func_indices.count(key) != 0) {
        // skip a duplicated output, since it was counted in the duplicated state output previously
        continue;
      }
    }

    NupharFuncInfo::FuncArgMeta output_meta;
    output_meta.dtype = OrtTypeInfo::ElementTypeFromProto(def->TypeAsProto()->tensor_type().elem_type());
    output_meta.ort_arg_index = ort_arg_index;

    // shape and symbols
    for (int dim = 0; dim < gsl::narrow<int>(ShapeRank(def)); ++dim) {
      if (ShapeHasSymbol(def, dim)) {
        auto p = std::make_pair(gsl::narrow<size_t>(dim), ShapeSymbol(def, dim));
        output_meta.dim_symbols.push_back(p);
        output_meta.inferred_shape.push_back(Dimension_Unknown);
      } else if (ShapeHasValue(def, dim)) {
        output_meta.inferred_shape.push_back(ShapeValue(def, dim));
      } else {
        output_meta.inferred_shape.push_back(Dimension_Unknown);
      }
    }
    func_info->output_metas.push_back(output_meta);
    ++tvm_output_idx;
  }

  // set output_count as the real output count
  func_info->func_output_count = tvm_output_idx;

  // set tvm type_codes
  func_info->type_codes.resize(func_info->func_input_count + func_info->func_output_count, TVMTypeCode::kNDArrayContainer);

  // set control-flow info
  func_info->cf_info = std::move(scan_info);
}

void FillNupharFuncInfo(NupharFuncInfo* func_info,
                        nuphar::OrtSubgraphAllocationInfo* partition_info,
                        const nuphar::NupharSubgraphUnit& subgraph,
                        const NupharCodeGenCtx& codegen_ctx,
                        tvm::Target tvm_target,
                        tvm::runtime::PackedFunc packed_func,
                        const std::string& name) {
  if (subgraph.nodes.front()->OpType() == "Scan") {
    FillScanExecInfo(func_info, partition_info, *subgraph.nodes.front(), codegen_ctx, tvm_target, packed_func, name);
    return;
  }

  FillBasicFuncInfo(func_info, partition_info, subgraph, codegen_ctx, tvm_target, packed_func, name);
}

}  // namespace nuphar
}  // namespace onnxruntime
