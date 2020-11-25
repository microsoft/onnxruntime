// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/nuphar_compiler.h"

#include "core/codegen/common/profile.h"
#include "core/codegen/common/settings.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/common/nuphar_tvm_utils.h"
#include "core/providers/nuphar/compiler/nuphar_handle.h"
#include "core/providers/nuphar/compiler/nuphar_op_ir_builder.h"
#include "core/providers/nuphar/compiler/nuphar_schedule_builder.h"

namespace onnxruntime {
namespace nuphar {

static void HandleAllOutputs(
    const std::vector<const NodeArg*>& outputs,
    tvm::Array<tvm::Tensor>& tvm_args,
    tvm::Array<tvm::Tensor>& tvm_outputs,
    const NupharCodeGenCtx& context) {
  // find out all outputs
  std::set<NodeKey> visited_alias_def;
  auto add_tvm_arg_and_output = [&](const onnxruntime::NodeArg* def) {
    auto& tvm_tensor = context.GetTVMTensorCtx().Lookup(def);
    tvm_args.push_back(tvm_tensor);
    tvm_outputs.push_back(tvm_tensor);
  };

  for (const NodeArg* def : outputs) {
    const NodeArg* input_def = Promote<CodeGenUnitStats>(context.GetGraphStats())->SourceDefOfOutputAlias(def);
    if (input_def) {
      auto key = GetKey(input_def);
      if (visited_alias_def.count(key) == 0) {
        visited_alias_def.insert(key);
        add_tvm_arg_and_output(input_def);
      }
    } else {
      auto key = GetKey(def);
      if (visited_alias_def.count(key) == 0) {
        visited_alias_def.insert(key);
        add_tvm_arg_and_output(def);
      }
    }
  }
}

// Constructor for Node
// This is mainly for single node support
// For multiple subgraph support, please call the next constructor
NupharCompiler::NupharCompiler(const Node& node,
                               const std::map<std::string, const Tensor*>& initializer,
                               std::unordered_map<std::string, std::unique_ptr<Tensor>>& generated_initializers,
                               const NupharCodeGenHandle* handle)
    : num_initializers_in_graph_inputs_(0),
      context_(node, initializer, generated_initializers, handle) {}

NupharCompiler::NupharCompiler(const nuphar::NupharSubgraphUnit& subgraph,
                               std::unordered_map<std::string, std::unique_ptr<Tensor>>& generated_initializers,
                               const NupharCodeGenHandle* handle)
    : num_initializers_in_graph_inputs_(0),
      context_(subgraph, generated_initializers, handle) {}

Status NupharCompiler::Build(const nuphar::NupharSubgraphUnit& subgraph) {
  if (subgraph.nodes.front()->OpType() == "Scan") {
    return BuildSubgraph(*subgraph.nodes.front());
  }

  tvm_args_ = tvm::Array<tvm::Tensor>();
  tvm_outputs_ = tvm::Array<tvm::Tensor>();

  ORT_RETURN_IF_ERROR(CreateTVMIR(subgraph, context_));

  // fill in all non-initializer inputs
  num_initializers_in_graph_inputs_ = 0;
  for (auto& def : subgraph.inputs) {
    if (context_.IsInitializer(def->Name())) {
      ++num_initializers_in_graph_inputs_;
    } else {
      tvm_args_.push_back(context_.GetTVMTensorCtx().Lookup(def));
    }
  }

  // fill in all initializers
  for (const auto& item : context_.GetWeightLayoutMap()) {
    const WeightLayoutCodegenInfo* layout_info = item.second.get();
    tvm_args_.push_back(layout_info->marshalled_tensor);
  }

  // find out all outputs, and save the output shapes
  HandleAllOutputs(subgraph.outputs, tvm_args_, tvm_outputs_, context_);

  return Status::OK();
}

// BuildSubgraph drive a graph traversal that calls CreateInput and CreateOutputs metioned above for a subgraph.
// And collect args among nodes.
// We need another API other than Build, because name mismatching
Status NupharCompiler::BuildSubgraph(const Node& node) {
  tvm_args_ = tvm::Array<tvm::Tensor>();
  tvm_outputs_ = tvm::Array<tvm::Tensor>();

  auto subgraph = GetSubgraph(node);

  ORT_RETURN_IF_ERROR(CreateTVMIR(GraphViewer(*subgraph), context_, /*use_placeholder_for_input*/ true));

  num_initializers_in_graph_inputs_ = 0;
  // fill in all non-initializer inputs

  for (const auto& input : subgraph->GetInputs()) {
    if (context_.IsInitializer(input->Name())) {
      ++num_initializers_in_graph_inputs_;
    } else {
      tvm_args_.push_back(context_.GetTVMTensorCtx().Lookup(input));
    }
  }

  // fill in implicit inputs
  for (const auto& input : node.ImplicitInputDefs()) {
    if (context_.IsInitializer(input->Name())) {
      ++num_initializers_in_graph_inputs_;
    } else {
      tvm_args_.push_back(context_.GetTVMTensorCtx().Lookup(input));
    }
  }

  // fill in all initializers
  for (const auto& item : context_.GetWeightLayoutMap()) {
    const WeightLayoutCodegenInfo* layout_info = item.second.get();
    tvm_args_.push_back(layout_info->marshalled_tensor);
  }

  // find out all outputs
  HandleAllOutputs(subgraph->GetOutputs(), tvm_args_, tvm_outputs_, context_);

  return Status::OK();
}

tvm::runtime::PackedFunc NupharCompiler::GetLoweredPackedFunc(
    const std::string& func_name,
    tvm::Target tvm_target,
    tvm::Target tvm_host_target,
    const tvm::BuildConfig& config,
    const std::string& subgraph_type,
    const std::string& subgraph_name) {
  // TODO: refactor the following logic for both JIT-caching and AOT support
  // JIT-caching and AOT are mutual exclusive.
  // Change it by not always saving a compiled func unless it is in JIT-Caching model.
  // In AOT, there should be another member func explicitly loading
  tvm::runtime::PackedFunc cached_func;
  auto cache_status = nuphar::LoadTVMPackedFuncFromCache(func_name, cached_func);
  if (cache_status != nuphar::CacheStatus::Found) {
    codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();

    if (settings.HasOption(kNupharCacheForceNoJIT)) {
      if (settings.OptionMatches(kNupharCacheForceNoJIT, "on")) {
        ORT_THROW("Force not using JIT code!");
      }
    }

    tvm::Schedule tvm_schedule = CreateSchedule(tvm_outputs_, context_);
    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
    tvm::Array<tvm::LoweredFunc> lowered = tvm::lower(tvm_schedule, tvm_args_, func_name, binds, config);

    if (settings.HasOption(codegen::CodeGenSettings::kCodeGenDumpLower)) {
      if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenDumpLower, "verbose") ||
          settings.OptionMatches(codegen::CodeGenSettings::kCodeGenDumpLower, subgraph_type)) {
        for (const auto& func : lowered)
          LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "[CODEGEN_DUMP_LOWER] Dumping lowered func: " << func << std::endl
                                                   << func->body;
      } else if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenDumpLower, "concise")) {
        LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << "[CODEGEN_DUMP_LOWER] Subgraph Type: "
                                                 << subgraph_type << ", name: " << subgraph_name
                                                 << " #lowered funcs: " << lowered.size() << std::endl;
      }
    }

    tvm::runtime::Module module = tvm::build(lowered, tvm_target, tvm_host_target, config);
    tvm_codegen::DumpTVMModuleToFile(func_name, module);
    if (cache_status == nuphar::CacheStatus::Missing) {
      nuphar::SaveTVMModuleToCache(func_name, module);
    }
    cached_func = module.GetFunction(func_name);
  }

  return cached_func;
}

static tvm::BuildConfig CreateConfig(const Node& node,
                                     bool allow_unaligned_buffers) {
  tvm::BuildConfig config = tvm::build_config();
  config->disable_select_rewriting = true;

  if (allow_unaligned_buffers) {
    config->data_alignment = 1;  // aligned to 1
  } else {
    config->data_alignment = gsl::narrow<int>(MlasGetPreferredBufferAlignment());
  }

  config->restricted_func = true;
  return config;
}

// Lower compiles the tvm::Tensor to a function
Status NupharCompiler::Lower(const nuphar::NupharSubgraphUnit& subgraph,
                             tvm::Target tvm_target,
                             tvm::Target tvm_host_target,
                             NupharFuncInfo* func_info,
                             nuphar::OrtSubgraphAllocationInfo* partition_info) {
  const auto& codegen_handle = context_.GetCodeGenHandle();
  const auto& target_codegen = *codegen_handle->codegen_target;
  std::string func_name = nuphar::GetPackedFuncName(subgraph, target_codegen, codegen_handle->parallel_min_workloads);
  tvm::BuildConfig config = CreateConfig(*subgraph.nodes.front(),
                                         context_.GetCodeGenHandle()->allow_unaligned_buffers);

  // using "subgraph" for type and name for now
  // TODO: change name
  tvm::runtime::PackedFunc cached_func =
      GetLoweredPackedFunc(
          func_name, tvm_target, tvm_host_target,
          config, "subgraph", "subgraph");

  FillNupharFuncInfo(func_info, partition_info, subgraph, context_, tvm_target, cached_func, func_name);

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
