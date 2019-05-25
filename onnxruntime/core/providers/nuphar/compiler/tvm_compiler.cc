// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/tvm_compiler.h"

#include "core/codegen/common/profile.h"
#include "core/codegen/common/settings.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/target/ort_tvm_utils.h"
#include "core/providers/nuphar/common/analysis/subgraph_gen_stats.h"
#include "core/providers/nuphar/common/nuphar_tvm_utils.h"
#include "core/providers/nuphar/compiler/nuphar_handle.h"
#include "core/providers/nuphar/compiler/nuphar_op_ir_builder.h"
#include "core/providers/nuphar/compiler/nuphar_schedule_builder.h"

namespace onnxruntime {
namespace tvm_codegen {

// BuildSubgraph drive a graph traversal that calls CreateInput and CreateOutputs metioned above for a subgraph.
// And collect args among nodes.
// We need another API other than Build, because name mismatching
Status TVMCompiler::BuildSubgraph() {
  tvm_args_ = tvm::Array<tvm::Tensor>();
  tvm_outputs_ = tvm::Array<tvm::Tensor>();

  auto subgraph = GetSubgraph(node_);

  std::vector<std::pair<const NodeArg*, std::size_t>> extras;
  extras.push_back(std::make_pair(node_.OutputDefs()[0], 0));

  ORT_RETURN_IF_ERROR(CreateTVMIR(GraphViewer(*subgraph), context_, /*use_placeholder_for_input*/ true));

  num_initializers_in_graph_inputs_ = 0;
  // fill in all non-initializer inputs
  for (const auto& input : subgraph->GetInputs()) {
    if (context_.IsInitializerMarshalled(input->Name())) {
      ++num_initializers_in_graph_inputs_;
    } else {
      tvm_args_.push_back(context_.GetTVMTensorCtx().Lookup(input));
    }
  }

  // fill in all initializers
  for (const auto& item : context_.GetInitializerMap()) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      tvm_args_.push_back(info.layout_info->marshalled_tensor);
    }
  }

  // find out all outputs
  std::set<NodeKey> visited_alias_def;

  auto add_tvm_arg_and_output = [&](const onnxruntime::NodeArg* def) {
    auto& tvm_tensor = context_.GetTVMTensorCtx().Lookup(def);
    tvm_args_.push_back(tvm_tensor);
    tvm_outputs_.push_back(tvm_tensor);
  };

  for (const auto& def : subgraph->GetOutputs()) {
    auto input_def = codegen::Promote<codegen::SubGraphStats>(context_.GetGraphStats())->SourceDefOfOutputAlias(def);
    if (input_def) {
      auto key = GetKey(input_def);
      if (visited_alias_def.count(key) == 0) {
        visited_alias_def.insert(key);
        add_tvm_arg_and_output(input_def);
      }
    } else {
      add_tvm_arg_and_output(def);
    }
  }

  return Status::OK();
}

TVMCompiler::TVMCompiler(const Node& node,
                         InitializerMap& initializer_lut,
                         const NupharCodeGenHandle* handle)
    : node_(node),
      num_initializers_in_graph_inputs_(0),
      context_(node, initializer_lut, handle) {}

// Collect states from meta
// TODO: rewrite this, and move it out class and to another file
void TVMCompiler::GetStateTensors(tvm::Array<tvm::Tensor>& in_state_tensors,
                                  tvm::Array<tvm::Tensor>& out_state_tensors) {
  for (const auto& loop_state_iter : context_.GetTVMTensorCtx().loop_states) {
    for (const auto& l_state : loop_state_iter.second) {
      in_state_tensors.push_back(l_state.first);
      out_state_tensors.push_back(l_state.second);
    }
  }
}

// Build drive a graph traversal that calls CreateInput and CreateOutputs metioned above.
// And collect args among nodes.
Status TVMCompiler::Build() {
  if (node_.OpType() == "Scan")
    return BuildSubgraph();

  tvm_args_ = tvm::Array<tvm::Tensor>();
  tvm_outputs_ = tvm::Array<tvm::Tensor>();

  // TODO refactor the following
  bool is_fused = (node_.NodeType() == Node::Type::Fused);
  if (is_fused) {
    const onnxruntime::Graph& onnx_func_body = node_.GetFunctionBody()->Body();
    ORT_RETURN_IF_ERROR(CreateTVMIR(GraphViewer(onnx_func_body), context_, /*use_placeholder_for_input*/ true));
  } else {
    ORT_RETURN_IF_ERROR(CreateTVMIR(node_, context_));
  }

  tvm::Array<tvm::Tensor> in_state_tensors;
  tvm::Array<tvm::Tensor> out_state_tensors;

  // TODO: rewrite this, also move it another file
  GetStateTensors(in_state_tensors, out_state_tensors);

  // fill in in_states
  for (auto in_state_tensor : in_state_tensors) {
    tvm_args_.push_back(in_state_tensor);
  }

  // fill in all non-initializer inputs
  num_initializers_in_graph_inputs_ = 0;
  node_.ForEachWithIndex(
      node_.InputDefs(),
      [this](const NodeArg& def, size_t) {
        if (context_.IsInitializerMarshalled(def.Name())) {
          ++num_initializers_in_graph_inputs_;
        } else {
          tvm_args_.push_back(context_.GetTVMTensorCtx().Lookup(&def));
        }
        return Status::OK();
      });

  // fill in all initializers
  for (const auto& item : context_.GetInitializerMap()) {
    const auto& info = item.second;
    if (nullptr != info.layout_info) {
      tvm_args_.push_back(info.layout_info->marshalled_tensor);
    }
  }

  // fill in out_states. Note that out_states are added into tvm_outputs_,
  // for which we will create schedules later.
  for (auto out_state_tensor : out_state_tensors) {
    tvm_args_.push_back(out_state_tensor);
    tvm_outputs_.push_back(out_state_tensor);
  }

  // find out all outputs, and save the output shapes
  std::set<NodeKey> visited_alias_def;
  node_.ForEachWithIndex(
      node_.OutputDefs(),
      [this, &visited_alias_def](const NodeArg& def, size_t) {
        auto input_def = codegen::Promote<codegen::SubGraphStats>(context_.GetGraphStats())->SourceDefOfOutputAlias(&def);
        if (input_def) {
          auto key = GetKey(input_def);
          if (visited_alias_def.count(key) == 0) {
            visited_alias_def.insert(key);

            auto& tvm_tensor = context_.GetTVMTensorCtx().Lookup(input_def);
            tvm_args_.push_back(tvm_tensor);
            tvm_outputs_.push_back(tvm_tensor);
          }
        } else {
          auto& tvm_tensor = context_.GetTVMTensorCtx().Lookup(&def);
          tvm_args_.push_back(tvm_tensor);
          tvm_outputs_.push_back(tvm_tensor);
        }
        return Status::OK();
      });
  return Status::OK();
}

// Lower compiles the tvm::Tensor to a function
Status TVMCompiler::Lower(tvm::Target tvm_target,
                          tvm::Target tvm_host_target,
                          NupharFuncInfo* func_info) {
  // TODO move cache to another file.
  const auto& target_codegen = *context_.GetCodeGenHandle()->codegen_target;
  std::string func_name = NormalizeCppName(node_.OpType() + std::to_string(node_.Index()) + " " + target_codegen.GetTargetName());
  tvm::runtime::PackedFunc cached_func = nuphar_codegen::LoadTVMPackedFuncFromCache(func_name);
  if (cached_func == nullptr) {
    auto tvm_schedule = CreateSchedule(tvm_outputs_, context_);

    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
    auto config = tvm::build_config();
    config->disable_select_rewriting = true;
    // RNN inputs may not be aligned when moving along sequence axis
    bool allow_unaligned_buffers = context_.GetCodeGenHandle()->allow_unaligned_buffers;
    if (node_.OpType() == "Scan" || node_.OpType() == "LSTM" || allow_unaligned_buffers)
      config->data_alignment = 1;
    auto lowered = tvm::lower(tvm_schedule, tvm_args_, func_name, binds, config);

    codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
    if (settings.HasOption(codegen::CodeGenSettings::kCodeGenDumpLower)) {
      if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenDumpLower, "verbose") ||
          settings.OptionMatches(codegen::CodeGenSettings::kCodeGenDumpLower, node_.OpType())) {
        for (const auto& func : lowered)
          LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "[NUPHAR_DUMP_LOWER] Dumping lowered func: " << func << std::endl
                                                   << func->body;
      } else if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenDumpLower, "concise")) {
        LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "[NUPHAR_DUMP_LOWER] OpType: " << node_.OpType() << ", name: " << node_.Name()
                                                 << " #lowered funcs: " << lowered.size() << std::endl;
      }
    }

    auto module = tvm::build(lowered, tvm_target, tvm_host_target, config);
    DumpTVMModuleToFile(func_name, module);
    nuphar_codegen::SaveTVMModuleToCache(func_name, module);
    cached_func = module.GetFunction(func_name);
  }

  FillNupharFuncInfo(func_info, node_, context_, tvm_target, cached_func, func_name);

  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
