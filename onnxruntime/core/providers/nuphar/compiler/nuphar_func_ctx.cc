// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/nuphar_func_ctx.h"

#include "core/providers/nuphar/runtime/control_flow/scan_exec_ctx.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/subgraph_gen_stats.h"
#include <unordered_map>

// from onnxruntime_typeinf.cc, in global namespace
const onnxruntime::DataTypeImpl* ElementTypeFromProto(int type);

namespace onnxruntime {
namespace tvm_codegen {

static void FillBasicFuncInfo(NupharFuncInfo* func_info,
                              const Node& node,
                              const NupharCodeGenCtx& codegen_ctx,
                              tvm::Target tvm_target,
                              tvm::runtime::PackedFunc packed_func,
                              const std::string& name) {
  ORT_ENFORCE(nullptr != func_info);

  func_info->name = name;
  func_info->packed_func = packed_func;
  func_info->device_type = static_cast<DLDeviceType>(tvm_target->device_type);

  int tvm_input_idx = 0;

  // Handle inputs
  std::vector<int>& ort_input_to_func_indices = func_info->ort_input_to_func_indices;
  // Assign Input meta
  node.ForEachWithIndex(node.InputDefs(),
                        [&](const NodeArg& def, size_t /*i_def*/) {
                          if (codegen_ctx.IsInitializerMarshalled(def.Name())) {
                            ort_input_to_func_indices.push_back(NupharFuncInfo::FuncIndex_Initializer);
                            return Status::OK();  // skip initializers
                          }

                          NupharFuncInfo::FuncArgMeta input_meta;
                          input_meta.dtype = ElementTypeFromProto(def.TypeAsProto()->tensor_type().elem_type());
                          ort_input_to_func_indices.push_back(tvm_input_idx);

                          for (int dim = 0; dim < gsl::narrow<int>(ShapeRank(&def)); ++dim) {
                            if (ShapeHasSymbol(&def, dim)) {
                              input_meta.inferred_shape.push_back(Dimension_Unknown);
                              input_meta.dim_symbols.push_back(std::make_pair(gsl::narrow<size_t>(dim), ShapeSymbol(&def, dim)));
                            } else if (ShapeHasValue(&def, dim)) {
                              input_meta.inferred_shape.push_back(ShapeValue(&def, dim));
                            } else {
                              input_meta.inferred_shape.push_back(Dimension_Unknown);
                            }
                          }
                          func_info->input_metas.push_back(input_meta);
                          ++tvm_input_idx;
                          return Status::OK();
                        });

  // Handle initializers
  // Initializer meta
  std::vector<const Tensor*>& intializers = func_info->intializers;
  // Assign Initializer meta
  for (const auto& item : codegen_ctx.GetInitializerMap()) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      bool is_marshalled = info.layout_info->is_marshalled;
      const Tensor* t =
          is_marshalled ? info.layout_info->marshalled_initializer.get()
                        : info.original_initializer;

      intializers.push_back(t);
      ++tvm_input_idx;
    }
  }

  // set input_count = the number of inputs + the number of initializers
  func_info->input_count = gsl::narrow<size_t>(tvm_input_idx);

  // Handle Outputs
  // Output meta
  std::vector<int>& ort_output_to_func_indices = func_info->ort_output_to_func_indices;
  std::vector<std::pair<int, size_t>>& ort_aliased_output_to_func_indices = func_info->ort_aliased_output_to_func_indices;
  // Assign Output meta
  size_t tvm_output_idx = 0;
  std::unordered_map<NodeKey, size_t> visited_output_defs;
  node.ForEachWithIndex(node.OutputDefs(),
                        [&](const NodeArg& def, size_t i_def) {
                          const NodeArg* source_def = codegen::Promote<codegen::SubGraphStats>(codegen_ctx.GetGraphStats())
                                                          ->SourceDefOfOutputAlias(&def);
                          // Determine output alias
                          if (nullptr != source_def) {
                            auto key = GetKey(source_def);
                            if (visited_output_defs.count(key) != 0) {
                              // source_def has visisted ==> a duplicated output
                              ort_output_to_func_indices.push_back(NupharFuncInfo::FuncIndex_AliasedOutput);
                              ort_aliased_output_to_func_indices.emplace_back(gsl::narrow<int>(i_def),
                                                                              func_info->input_count +
                                                                                  visited_output_defs[key]);
                              return Status::OK();
                            }
                            // update visited_output_defs
                            visited_output_defs.insert(std::make_pair(key, tvm_output_idx));
                          }

                          ort_output_to_func_indices.push_back(gsl::narrow<int>(func_info->input_count + tvm_output_idx));

                          NupharFuncInfo::FuncArgMeta output_meta;
                          output_meta.dtype = ElementTypeFromProto(def.TypeAsProto()->tensor_type().elem_type());

                          // shape and symbols
                          for (int dim = 0; dim < gsl::narrow<int>(ShapeRank(&def)); ++dim) {
                            if (ShapeHasSymbol(&def, dim)) {
                              auto p = std::make_pair(gsl::narrow<size_t>(dim), ShapeSymbol(&def, dim));
                              output_meta.dim_symbols.push_back(p);
                              output_meta.inferred_shape.push_back(Dimension_Unknown);
                            } else if (ShapeHasValue(&def, dim)) {
                              output_meta.inferred_shape.push_back(ShapeValue(&def, dim));
                            } else {
                              output_meta.inferred_shape.push_back(Dimension_Unknown);
                            }
                          }
                          func_info->output_metas.push_back(output_meta);
                          ++tvm_output_idx;
                          return Status::OK();
                        });

  // set output_count as the real output count
  func_info->output_count = tvm_output_idx;

  // set tvm type_codes
  func_info->type_codes.resize(func_info->input_count + func_info->output_count, TVMTypeCode::kNDArrayContainer);
}

static void FillScanExecInfo(NupharFuncInfo* func_info,
                             const Node& node,
                             const NupharCodeGenCtx& codegen_ctx,
                             tvm::Target tvm_target,
                             tvm::runtime::PackedFunc packed_func,
                             const std::string& name) {
  ORT_ENFORCE(nullptr != func_info);

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
  // TODO: please confirm this
  size_t num_variadic_inputs = subgraph->GetInputs().size();
  size_t num_variadic_outputs = subgraph->GetOutputs().size();

  num_state_variables = gsl::narrow<int64_t>(num_variadic_inputs) - num_scan_inputs;
  num_scan_outputs = gsl::narrow<int64_t>(num_variadic_outputs) - num_state_variables;

  // Set ScanExecInfo's parameter count meta
  scan_info->num_state_variables = num_state_variables;
  scan_info->num_scan_inputs = num_scan_inputs;
  scan_info->num_scan_outputs = num_scan_outputs;

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
  // Input meta
  std::vector<int>& ort_input_to_func_indices = func_info->ort_input_to_func_indices;
  // assign state inputs & inputs
  for (size_t ort_input_idx = 0; ort_input_idx < num_variadic_inputs; ++ort_input_idx) {
    const NodeArg* def = subgraph->GetInputs()[ort_input_idx];

    if (ort_input_idx >= gsl::narrow<size_t>(num_state_variables)) {
      // initializer should only happen in real inputs, not in state inputs
      if (codegen_ctx.IsInitializerMarshalled(def->Name())) {
        ort_input_to_func_indices.push_back(NupharFuncInfo::FuncIndex_Initializer);
        continue;  // skip initializers
      }
    }

    NupharFuncInfo::FuncArgMeta input_meta;
    input_meta.dtype = ElementTypeFromProto(def->TypeAsProto()->tensor_type().elem_type());
    ort_input_to_func_indices.push_back(tvm_input_idx);

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
  }

  // Handle initializers
  // Initializer meta
  std::vector<const Tensor*>& intializers = func_info->intializers;
  // Assign Initializer meta
  for (const auto& item : codegen_ctx.GetInitializerMap()) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      bool is_marshalled = info.layout_info->is_marshalled;
      const Tensor* t =
          is_marshalled ? info.layout_info->marshalled_initializer.get()
                        : info.original_initializer;

      intializers.push_back(t);
      ++tvm_input_idx;
    }
  }

  // set input_count = the number of inputs (real inputs + state inputs) + the number of initializers
  func_info->input_count = gsl::narrow<size_t>(tvm_input_idx);

  // Handle State Outputs and Outputs
  // Output meta
  std::vector<int>& ort_output_to_func_indices = func_info->ort_output_to_func_indices;
  std::vector<std::pair<int, size_t>>& ort_aliased_output_to_func_indices = func_info->ort_aliased_output_to_func_indices;
  std::vector<int>& state_to_output_indices = scan_info->state_to_output_indices;

  // Since in Scan, we only allow state using output's memory during Execution, not the other around
  // By doing alias detection for inputs first, state inputs would be replaced.
  std::unordered_map<NodeKey, int> visited_output_defs;
  for (size_t ort_output_idx = gsl::narrow<size_t>(num_state_variables); ort_output_idx < num_variadic_outputs; ++ort_output_idx) {
    const NodeArg* def = subgraph->GetOutputs()[ort_output_idx];
    ORT_ENFORCE(nullptr != def);
    const NodeArg* source_def = codegen::Promote<codegen::SubGraphStats>(codegen_ctx.GetGraphStats())
                                    ->SourceDefOfOutputAlias(def);
    if (nullptr != source_def) {
      auto key = GetKey(source_def);
      ORT_ENFORCE(visited_output_defs.count(key) == 0,
                  "Scan has alias btw two states. Nuphar only support aliasing btw state and output in Scan");
      visited_output_defs.insert(std::make_pair(key, gsl::narrow<int>(ort_output_idx - num_state_variables)));
    }
  }

  // assign state outputs and outputs
  size_t tvm_output_idx = 0;
  std::unordered_map<NodeKey, int> visited_output_state_indices_defs;
  for (size_t ort_output_idx = 0; ort_output_idx < num_variadic_outputs; ++ort_output_idx) {
    const NodeArg* def = subgraph->GetOutputs()[ort_output_idx];
    ORT_ENFORCE(nullptr != def);
    const NodeArg* source_def = codegen::Promote<codegen::SubGraphStats>(codegen_ctx.GetGraphStats())
                                    ->SourceDefOfOutputAlias(def);

    // Determine alias btw output and state output
    auto key = source_def != nullptr ? GetKey(source_def) : "";
    if (ort_output_idx < gsl::narrow<size_t>(num_state_variables)) {
      if (visited_output_defs.count(key) != 0) {
        // If state output is an alias
        // record i_output for the lookup of the aliased output later
        visited_output_state_indices_defs.insert(std::make_pair(key, gsl::narrow<int>(func_info->input_count + tvm_output_idx)));

        // also record ort_aliased_output_to_func_indices
        ort_aliased_output_to_func_indices.push_back(std::make_pair(gsl::narrow<int>(ort_output_idx),
                                                                    func_info->input_count + tvm_output_idx));

        state_to_output_indices.push_back(visited_output_defs[key]);
        ort_output_to_func_indices.push_back(NupharFuncInfo::FuncIndex_AliasedOutput);
      } else {
        // the state output not aliased(no scan output shares with it)
        state_to_output_indices.push_back(NupharFuncInfo::FuncIndex_NonAliasedOutput);
        ort_output_to_func_indices.push_back(gsl::narrow<int>(func_info->input_count + tvm_output_idx));
      }
    } else {
      if (visited_output_state_indices_defs.count(key) != 0) {
        // if an output is alias of a state output
        ort_output_to_func_indices.push_back(visited_output_state_indices_defs[key]);
        continue;
      } else {
        // if an output is not alias of a state output
        ort_output_to_func_indices.push_back(gsl::narrow<int>(func_info->input_count + tvm_output_idx));
      }
    }

    NupharFuncInfo::FuncArgMeta output_meta;
    output_meta.dtype = ElementTypeFromProto(def->TypeAsProto()->tensor_type().elem_type());

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
  func_info->output_count = tvm_output_idx;

  // set tvm type_codes
  func_info->type_codes.resize(func_info->input_count + func_info->output_count, TVMTypeCode::kNDArrayContainer);

  // set control-flow info
  func_info->cf_info = std::move(scan_info);
}

void FillNupharFuncInfo(NupharFuncInfo* func_info,
                        const Node& node,
                        const NupharCodeGenCtx& codegen_ctx,
                        tvm::Target tvm_target,
                        tvm::runtime::PackedFunc packed_func,
                        const std::string& name) {
  if (node.OpType() == "Scan") {
    FillScanExecInfo(func_info, node, codegen_ctx, tvm_target, packed_func, name);
    return;
  }

  FillBasicFuncInfo(func_info, node, codegen_ctx, tvm_target, packed_func, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
