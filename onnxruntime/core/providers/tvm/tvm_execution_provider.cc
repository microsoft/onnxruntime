// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <map>

#include "core/framework/execution_provider.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/platform/env.h"
#include "core/graph/model.h"

#include "tvm_execution_provider.h"
#include "xpu_data_transfer.h"
#include "tvm_allocator.h"
#include "tvm_utils.h"
#include "tvm_api.h"


using namespace ONNX_NAMESPACE;

namespace onnxruntime {

// Information to construct kernel function state.
struct TVMFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  TvmModule* module = nullptr;
  std::function<TvmModule*(std::string func_name,
                const std::vector<std::vector<int64_t>>& input_shapes)> compiler = nullptr;
};

TvmExecutionProvider::TvmExecutionProvider(const TvmEPOptions& options)
    : IExecutionProvider{kTvmExecutionProvider},
      options_{options} {
  AllocatorCreationInfo default_memory_info = {[](int) {
                                                 return std::make_unique<TVMAllocator>();
                                               },
                                               0, false};
  allocator_ = CreateAllocator(default_memory_info);
  InsertAllocator(allocator_);

  // Get environment variables
  const Env& env_instance = Env::Default();

  const std::string dump_subgraphs_env = env_instance.GetEnvironmentVar(tvm::env_vars::kDumpSubgraphs);
  if (!dump_subgraphs_env.empty()) {
    dump_subgraphs_ = std::stoi(dump_subgraphs_env) != 0;
  }
}

TvmExecutionProvider::~TvmExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
TvmExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                     const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();

  std::unordered_set<std::string> required_initializers;
  const std::vector<NodeIndex>& sorted_nodes = graph_viewer.GetNodesInTopologicalOrder();
  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  for (auto& node_idx : sorted_nodes) {
    graph_viewer.GetNode(node_idx)->ForEachDef([&required_initializers, &init_tensors]
                                               (const NodeArg& node_arg, bool is_input) {
              if(is_input && init_tensors.count(node_arg.Name())) {
                  required_initializers.insert(node_arg.Name());
              } }, true);
  }

  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "TVMStandalone";
  meta_def->domain = "StandaloneTest";
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  for (auto& nodeArgPtr : graph_viewer.GetInputs()) {
    inputs.push_back(nodeArgPtr->Name());
  }

  for (auto& name : required_initializers) {
    inputs.push_back(name);
  }

  for (auto& nodeArgPtr : graph_viewer.GetOutputs()) {
    outputs.push_back(nodeArgPtr->Name());
  }
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  sub_graph->SetMetaDef(std::move(meta_def));
  sub_graph->nodes = sorted_nodes;
  result.push_back(
      std::make_unique<ComputeCapability>(std::move(sub_graph)));
  return result;
}

common::Status TvmExecutionProvider::Compile(const std::vector<Node*>& nodes,
                                             std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (auto* fused_node : nodes) {
    auto func_body = fused_node->GetFunctionBody();
    if (!func_body)
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    const std::string func_name = fused_node->Name();
    const Graph& node_graph = func_body->Body();
    Model model(node_graph.Name(), true, ModelMetaData(), PathString(),
                             IOnnxRuntimeOpSchemaRegistryList(), node_graph.DomainToVersionMap(),
                             std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();

    *(model_proto.mutable_graph()) = node_graph.ToGraphProto();
    auto opset = model_proto.add_opset_import();
    opset->set_domain(kOnnxDomain);
    opset->set_version(node_graph.DomainToVersionMap().at(kOnnxDomain));

    std::string string_buf;
    model_proto.SerializeToString(&string_buf);
    buffers_[func_name] = string_buf;
    opsets_[func_name] = int(opset->version());
    model_paths_[func_name] = fused_node->ModelPath().ToPathString();;

    if (dump_subgraphs_) {
        std::fstream dump("/tmp/" + fused_node->Name() + ".onnx",
                          std::ios::out | std::ios::trunc | std::ios::binary);
        model_proto.SerializeToOstream(&dump);
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = std::bind(&TvmExecutionProvider::CreateStateFunc,
                                               this,
                                               std::placeholders::_1,
                                               std::placeholders::_2);

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<TVMFuncState*>(state);
    };
    // TODO(vvchernov): implement ops checking and mechanism of gracefully passing the responsibility to other EPs
    // if the checking fails due to unsupported op(s)
    runners_[func_name] = std::make_shared<Runner>(this, func_name, node_graph);
    compute_info.compute_func = *runners_[func_name].get();

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::unique_ptr<IDataTransfer> TvmExecutionProvider::GetDataTransfer() const {
  //TODO(vvchernov): target or target host?
  if (options_.checkGPUTarget()) {
    return std::make_unique<onnxruntime::XPUDataTransfer>();
  } else if (options_.target.find("llvm") != std::string::npos) {
    return std::make_unique<onnxruntime::TvmCPUDataTransfer>();
  } else {
    ORT_NOT_IMPLEMENTED("TVM GetDataTransfer is not implemented for target ", options_.target);
  }
}

AllocatorPtr TvmExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  return allocator_;
}

int TvmExecutionProvider::CreateStateFunc(ComputeContext* context, FunctionState* state) {
  auto* state_ptr = new TVMFuncState();
  *state_ptr = {context->allocate_func,
                 context->release_func,
                 context->allocator_handle,
                 nullptr,
                 std::bind(&TvmExecutionProvider::CompileFunc,
                           this,
                           std::placeholders::_1,
                           std::placeholders::_2)};
  *state = state_ptr;
  return 0;
}

TvmModule* TvmExecutionProvider::CompileFunc(std::string func_name,
                                             const TVMTensorShapes& input_shapes) {
  if (modules_.count(func_name)) {
    return modules_[func_name].get();
  }

  TvmModule mod_f = tvm::TVMCompile(buffers_[func_name],
                                    model_paths_[func_name],
                                    options_,
                                    opsets_[func_name],
                                    input_shapes);
  auto module_ptr = std::make_shared<TvmModule>();
  *module_ptr = mod_f;
  modules_[func_name] = module_ptr;
  // Release memory after module generation
  buffers_.erase(func_name);
  opsets_.erase(func_name);
  return modules_[func_name].get();
}

}  // namespace onnxruntime
