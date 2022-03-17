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
namespace tvm {

// Information to construct kernel function state.
struct TVMFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  std::shared_ptr<tvm::TVMCompiler> compiler = nullptr;
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
  printOptions();
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

    std::string onnx_model_str;
    model_proto.SerializeToString(&onnx_model_str);
    compilers_[func_name] = std::make_shared<Compiler>(std::move(onnx_model_str),
                              fused_node->ModelPath().ToPathString(),
                              int(opset->version()));
    InputsInfoMap all_input_shapes;
    auto mod = compileModel(func_name, node_graph, all_input_shapes);

    std::vector<DLTensor> output_tensors;
    prepareOutputTensors(mod, output_tensors, node_graph.GetOutputs().size());

    runners_[func_name] = std::make_shared<Runner>(options_, mod, all_input_shapes, output_tensors);

    if (dump_subgraphs_) {
        std::fstream dump("/tmp/" + func_name + ".onnx",
                          std::ios::out | std::ios::trunc | std::ios::binary);
        model_proto.SerializeToOstream(&dump);
    }

    // TODO(vvchernov): implement ops checking and mechanism of gracefully passing the responsibility to other EPs
    // if the checking fails due to unsupported op(s)
    NodeComputeInfo compute_info = prepareComputeInfo(func_name);

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::unique_ptr<IDataTransfer> TvmExecutionProvider::GetDataTransfer() const {
  //TODO(vvchernov): target or target host?
  if (TvmEPOptionsHelper::checkGPUTarget(options_.target)) {
    return std::make_unique<XPUDataTransfer>();
  } else if (TvmEPOptionsHelper::checkCPUTarget(options_.target)) {
    return std::make_unique<TvmCPUDataTransfer>();
  } else {
    ORT_NOT_IMPLEMENTED("TVM GetDataTransfer is not implemented for target ", options_.target);
  }
}

AllocatorPtr TvmExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  return allocator_;
}

void TvmExecutionProvider::printOptions() {
  LOGS(*GetLogger(), INFO) << options_;
}

std::shared_ptr<TvmModule> TvmExecutionProvider::compileModel(const std::string& func_name,
                                                              const Graph& graph,
                                                              InputsInfoMap& all_input_shapes) {
  all_input_shapes.clear();

  TVMTensorShapes input_shapes;
  if (options_.freeze_weights) {
    setInputShapesForFreezedNN(graph, input_shapes, all_input_shapes);
  } else {
    setInputShapesForUnfreezedNN(graph, input_shapes, all_input_shapes);
  }

  std::shared_ptr<TvmModule> mod = compilers_[func_name]->operator()(options_, input_shapes);

  return mod;
}

void TvmExecutionProvider::setInputShapesForFreezedNN(const Graph& graph,
                                                      TVMTensorShapes& input_shapes,
                                                      InputsInfoMap& all_input_shapes) {
  const std::vector<const NodeArg*>& all_nodes = graph.GetInputsIncludingInitializers();

  size_t indx = 0;
  for (const auto* node : all_nodes) {
    if(!graph.IsInitializedTensor(node->Name())) {
      TensorShapeVector shape = getInputShape(node);
      all_input_shapes[indx++] = shape;
      input_shapes.emplace_back(shape);
    }
  }
}

void TvmExecutionProvider::setInputShapesForUnfreezedNN(const Graph& graph,
                                                        TVMTensorShapes& input_shapes,
                                                        InputsInfoMap& all_input_shapes) {
  const std::vector<const NodeArg*>& all_nodes = graph.GetInputsIncludingInitializers();

  size_t indx = 0;
  for (const auto* node : all_nodes) {
    TensorShapeVector shape = getInputShape(node);
    all_input_shapes[indx++] = shape;
    if(!graph.IsInitializedTensor(node->Name())) {
      input_shapes.emplace_back(shape);
    }
  }
}

TensorShapeVector TvmExecutionProvider::getInputShape(const NodeArg* node) {
    TensorShapeVector shape;
    const auto& node_name = node->Name();
    if(!options_.input_shapes.empty() &&
        options_.input_shapes.count(node_name)) {
      shape = options_.input_shapes[node_name];
    } else {
      shape = convertTensorShape(*node->Shape());
    }

    return shape;
}

TensorShapeVector TvmExecutionProvider::convertTensorShape(const TensorShapeProto& shape_proto) {
  TensorShape ort_shape = utils::GetTensorShapeFromTensorShapeProto(shape_proto);
  size_t dims = ort_shape.NumDimensions();

  TensorShapeVector shape(dims);
  for (size_t j = 0; j < dims; ++j) {
    int64_t dim = int64_t(ort_shape[j]);
    ORT_ENFORCE(dim > 0, "Input dimension is not positive value (dim = " + std::to_string(dim) + "). " +
      "Please use provider options to setup input_names and input_shapes");
    shape[j] = dim;
  }

  return shape;
}

void TvmExecutionProvider::prepareOutputTensors(const std::shared_ptr<tvm::TvmModule>& mod,
                                                std::vector<DLTensor>& output_tensors,
                                                size_t num) {
  ORT_ENFORCE(mod != nullptr, "TVM module is not compiled");
  output_tensors.clear();
  options_.output_shapes.clear();
  options_.output_shapes.resize(num);

  if (options_.executor != "vm") {
    tvm::TVMGetOutputShapes(*mod, options_.output_shapes);
  }

  for (auto& output_shape : options_.output_shapes) {
    DLTensor t;
    // Draft for tensor, correct data is defined during inference
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = nullptr;
    if (options_.executor == "vm") {
      t.ndim = 0;
      t.shape = nullptr;
    } else {
      t.ndim = output_shape.size();
      t.shape = output_shape.data();
    }

    output_tensors.push_back(t);
  }
}

NodeComputeInfo TvmExecutionProvider::prepareComputeInfo(const std::string& func_name) {
  NodeComputeInfo compute_info;
  compute_info.create_state_func = std::bind(&TvmExecutionProvider::createStateFunc,
                                              this,
                                              std::placeholders::_1,
                                              std::placeholders::_2);

  compute_info.release_state_func = [](FunctionState state) {
    if (state)
      delete static_cast<TVMFuncState*>(state);
  };

  compute_info.compute_func = *runners_[func_name].get();

  return compute_info;
}

int TvmExecutionProvider::createStateFunc(ComputeContext* context, FunctionState* state) {
  auto* state_ptr = new TVMFuncState();
  *state_ptr = {context->allocate_func,
                context->release_func,
                context->allocator_handle,
                compilers_[context->node_name]};
  // TODO(vvchernov): Who and when release state?
  *state = state_ptr;
  return 0;
}

}  // namespace tvm
}  // namespace onnxruntime
