// Copyright 2020 rock-chips.com Inc.

#include <unistd.h>
#include <limits>
#include <set>
#include <unordered_set>
#include <map>
#include <utility>
#include <functional>
#include "rknpu_execution_provider.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/framework/memcpy.h"
#include "node_attr_helper.h"
#include "rknpu/rknpu_pub.h"
#include "onnx_converter.h"

using std::string;
using std::vector;

namespace onnxruntime {

constexpr const char* RKNPU = "Rknpu";

struct RknpuFuncState {
  std::string uniq_input_shape;

  std::unique_ptr<rk::nn::Exection> exector;
  ONNX_NAMESPACE::ModelProto model_proto;
  std::unordered_map<std::string, int> input_map;
  std::unordered_map<std::string, int> output_map;
  std::vector<int> input_indexes;
  std::vector<int> output_indexes;
};

RknpuExecutionProvider::RknpuExecutionProvider()
    : IExecutionProvider{onnxruntime::kRknpuExecutionProvider} {
}

RknpuExecutionProvider::~RknpuExecutionProvider() {}

std::vector<std::vector<int>> RknpuExecutionProvider::GetSupportedNodes(
    const ONNX_NAMESPACE::ModelProto& model_proto) const {
  rknpu::OnnxConverter converter;
  return converter.GetSupportedNodes(model_proto);
}

std::vector<std::unique_ptr<ComputeCapability>>
RknpuExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                      const IKernelLookup& /*kernel_lookup*/) const {
  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Handle If and Loop operators
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() &&
        tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "Rknpu: Initializers with external data"
                               " location are not currently supported";
      return result;
    }
  }

  // This method is based on that of TRT EP
  // Construct modelproto from graph_viewer
  onnxruntime::Model model(graph_viewer.Name(), true, ModelMetaData(),
                           PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(),
                           graph_viewer.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(),
                           *GetLogger());
  onnxruntime::Graph& graph_build = model.MainGraph();
  const std::vector<NodeIndex>& node_index =
      graph_viewer.GetNodesInTopologicalOrder();
  std::set<NodeArg*> all_node_inputs;
  for (const auto& node : graph_viewer.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (const auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(
          input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
      all_node_inputs.insert(&n_input);
    }
    for (const auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(
          output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(),
                        inputs, outputs, &node.GetAttributes(), node.Domain());
  }
  const auto graph_outputs = graph_viewer.GetOutputs();
  // Add initializer to graph_viewer
  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(graph_build.Resolve().IsOK());
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  const auto supported_nodes_vector = GetSupportedNodes(model_proto);

  int counter = 0;

  for (const auto& group : supported_nodes_vector) {
    if (!group.empty()) {
      std::unordered_set<size_t> node_set;
      node_set.reserve(group.size());
      for (const auto& index : group) {
        node_set.insert(node_index[index]);
      }
      std::unique_ptr<IndexedSubGraph> sub_graph =
          std::make_unique<IndexedSubGraph>();
      // Find inputs and outputs of the subgraph
      std::unordered_map<const NodeArg*, int>
          fused_inputs, fused_outputs, fused_outputs_to_add;
      std::unordered_set<const NodeArg*> erased;
      int input_order = 0;
      int output_order = 0;

      for (const auto& index : group) {
        sub_graph->nodes.push_back(node_index[index]);
        const auto& node = graph_viewer.GetNode(node_index[index]);

        for (const auto& input : node->InputDefs()) {
          const auto& it = fused_outputs.find(input);

          if (it != fused_outputs.end()) {
            fused_outputs.erase(it);
            erased.insert(input);
          }
          // only when input is neither in output list nor erased list, add the
          // input to input list
          else if (erased.find(input) == erased.end()) {
            fused_inputs[input] = input_order++;
          }
        }

        // For output searching, there is a special case:
        // If node's OutputEdges are more than its outputs, meaning certain
        // output is used more than once,
        // if the output is connected to nodes that don't belong to the
        // subgraph, the output need to be added to the output list
        if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
          for (auto it = node->OutputEdgesBegin(),
                    end = node->OutputEdgesEnd();
               it != end; ++it) {
            const auto& node_idx = it->GetNode().Index();
            const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];

            if (node_set.find(node_idx) != node_set.end()) {
              const auto& iter = fused_inputs.find(output);

              if (iter != fused_inputs.end()) {
                fused_inputs.erase(iter);
                erased.insert(output);
              } else if (erased.find(output) == erased.end()) {
                fused_outputs[output] = output_order++;
              }
            } else {
              fused_outputs_to_add[output] = output_order++;
            }
          }
        } else {
          for (const auto& output : node->OutputDefs()) {
            const auto& it = fused_inputs.find(output);

            if (it != fused_inputs.end()) {
              fused_inputs.erase(it);
              erased.insert(output);
            }
            // only when output is neither in input list nor erased list,
            // add the output to output list
            else if (erased.find(output) == erased.end()) {
              fused_outputs[output] = output_order++;
            }
          }
        }
      }

      fused_outputs.insert(
          fused_outputs_to_add.begin(), fused_outputs_to_add.end());

      // Sort inputs and outputs by the order they were added
      std::map<int, const NodeArg*> inputs, outputs;

      for (auto it = fused_inputs.begin(),
                end = fused_inputs.end();
           it != end; ++it) {
        inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }

      for (auto it = fused_outputs.begin(),
                end = fused_outputs.end();
           it != end; ++it) {
        for (const auto& x : all_node_inputs) {
          if (x->Name() == it->first->Name()) {
            outputs.insert(
                std::pair<int, const NodeArg*>(it->second, it->first));
            break;
          }
        }
        if (std::find(graph_outputs.begin(),
                      graph_outputs.end(), it->first) != graph_outputs.end()) {
          outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
        }
      }

      // Assign inputs and outputs to subgraph's meta_def
      auto meta_def =
          std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
      meta_def->name = "RKNPU_" + std::to_string(counter++);
      meta_def->domain = kMSDomain;

      for (const auto& input : inputs) {
        meta_def->inputs.push_back(input.second->Name());
      }

      for (const auto& output : outputs) {
        meta_def->outputs.push_back(output.second->Name());
      }

      meta_def->since_version = 1;
      sub_graph->SetMetaDef(std::move(meta_def));

      result.push_back(
          std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

common::Status RknpuExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;
    onnxruntime::Model model(graph_body_viewer.Name(), true, ModelMetaData(),
                             PathString(),
                             IOnnxRuntimeOpSchemaRegistryList(),
                             graph_body_viewer.DomainToVersionMap(),
                             std::vector<ONNX_NAMESPACE::FunctionProto>(),
                             *GetLogger());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
    graph_body_viewer.ToProto(*model_proto->mutable_graph(), true, true);
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

    // Build map from input name to its index in input definitions
    std::unordered_map<std::string, int> input_map;
    const auto& input_defs = fused_node.InputDefs();
    input_map.reserve(input_defs.size());
    for (int i = 0, end = input_defs.size(); i < end; ++i) {
      input_map[input_defs[i]->Name()] = i;
    }

    // Build map from output name to its index in output definitions
    std::unordered_map<std::string, int> output_map;
    const auto& output_defs = fused_node.OutputDefs();
    output_map.reserve(output_defs.size());
    for (int i = 0, end = output_defs.size(); i < end; ++i) {
      output_map[output_defs[i]->Name()] = i;
    }

    model_proto_[fused_node.Name()] = model_proto;
    input_info_[fused_node.Name()] = input_map;
    output_info_[fused_node.Name()] = output_map;

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context,
                                         FunctionState* state) {
      std::unique_ptr<RknpuFuncState> p =
          std::make_unique<RknpuFuncState>();
      rk::nn::Graph* graph = new rk::nn::Graph();
      *p = {"", std::unique_ptr<rk::nn::Exection>(new rk::nn::Exection(graph)),
            model_proto_[context->node_name], input_info_[context->node_name],
            output_info_[context->node_name],
            std::vector<int>{}, std::vector<int>{}};
      *state = p.release();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        RknpuFuncState* p = static_cast<RknpuFuncState*>(state);
        rk::nn::Graph* graph = p->exector->GetGraph();
        delete graph;
        delete p;
      }
    };

    compute_info.compute_func = [](FunctionState state,
                                   const OrtApi* /*api*/,
                                   OrtKernelContext* context) {
      Ort::KernelContext ctx(context);
      RknpuFuncState* rk_state = reinterpret_cast<RknpuFuncState*>(state);
      const size_t n_inputs = ctx.GetInputCount();
      const size_t n_outputs = ctx.GetOutputCount();

      std::string input_shape;
      input_shape.reserve(4 * sizeof(int64_t) * n_inputs + n_inputs);
      for (size_t i = 0; i < n_inputs; i++) {
        auto input_tensor = ctx.GetInput(i);
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        auto tensor_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

        const auto ndim = tensor_shape.size();
        input_shape.append(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        input_shape.append(reinterpret_cast<const char*>(tensor_shape.data()),
                           ndim * sizeof(int64_t));
      }

      bool rebuild = false;
      std::vector<const void*> input_bufs(n_inputs);
      if (rk_state->uniq_input_shape == "") {
        auto graph_proto = rk_state->model_proto.mutable_graph();
        for (size_t i = 0; i < n_inputs; i++) {
          auto input_tensor = ctx.GetInput(i);
          input_bufs[i] = const_cast<const void*>(
              input_tensor.GetTensorRawData());
          auto tensor_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

          auto g_in_shape =
              graph_proto->mutable_input((int)i)->mutable_type()->mutable_tensor_type()->mutable_shape();
          g_in_shape->clear_dim();
          for (size_t dim = 0; dim < tensor_shape.size(); dim++) {
            g_in_shape->add_dim()->set_dim_value(tensor_shape[dim]);
          }
        }
        rk_state->uniq_input_shape = input_shape;
        rebuild = true;
      } else if (rk_state->uniq_input_shape != input_shape) {
        // TODO
        throw std::invalid_argument(
            "The input_shape is not match"
            " the rk_state->uniq_input_shape!");
      } else {
        LOGS_DEFAULT(INFO) << "input_shape equal to rk_state->uniq_input_shape,"
                              " skip rebuild!";
      }

      rk::nn::Graph* graph = rk_state->exector->GetGraph();
      if (rebuild) {
        rknpu::OnnxConverter converter;
        converter.Convert(rk_state->model_proto, graph, input_bufs, rk_state->input_map);

        rk_state->exector->Build();

        auto input_map = rk_state->input_map;
        auto output_map = rk_state->output_map;
        for (auto it = output_map.begin(); it != output_map.end(); it++) {
          if (converter.m(it->first) != it->first)
            output_map[converter.m(it->first)] = output_map[it->first];
        }

        int n_inputs = graph->GetInputs().size();
        rk_state->input_indexes.resize(n_inputs);
        for (int i = 0; i < n_inputs; ++i) {
          const auto input = graph->GetInputs()[i];
          const std::string& name = input->GetName();
          auto iter = input_map.find(name);
          if (iter != input_map.end()) {
            rk_state->input_indexes[i] = iter->second;
          }
        }

        int n_outputs = graph->GetOutputs().size();
        rk_state->output_indexes.resize(n_outputs);
        for (int i = 0; i < n_outputs; ++i) {
          const auto output = graph->GetOutputs()[i];
          const std::string& name = output->GetName();
          auto iter = output_map.find(name);
          if (iter != output_map.end()) {
            rk_state->output_indexes[i] = iter->second;
          }
        }
      }

      ORT_ENFORCE(graph->GetInputs().size() <= n_inputs,
                  "Inconsistent input sizes");
      ORT_ENFORCE(graph->GetOutputs().size() == n_outputs,
                  "Inconsistent output sizes");

      std::vector<rk::nn::InputInfo> inputs;
      inputs.resize(graph->GetInputs().size());
      for (size_t i = 0; i < graph->GetInputs().size(); i++) {
        auto input_tensor =
            ctx.GetInput(rk_state->input_indexes[i]);
        float* input_buf =
            const_cast<float*>(input_tensor.GetTensorData<float>());

        const auto input_element_count = input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

        auto type = graph->GetInputs()[i]->GetPrecision();
        size_t type_size = 4;
        switch (type) {
          case rk::nn::PrecisionType::FLOAT32:
            type_size = 4;
            break;
          case rk::nn::PrecisionType::UINT8:
            type_size = 1;
            break;
          case rk::nn::PrecisionType::INT32:
            type_size = 4;
            break;
          case rk::nn::PrecisionType::INT64:
            type_size = 8;
            break;
          default:
            // TODO
            throw std::invalid_argument(
                "compute_func: unknow input data type!");
            break;
        }

        inputs[i].index = i;
        inputs[i].buf = (void*)input_buf;
        inputs[i].size = input_element_count * type_size;
        inputs[i].pass_through = false;
        inputs[i].type = type;
        inputs[i].layout = rk::nn::DataLayoutType::NCHW;
      }

      std::vector<rk::nn::OutputInfo> outputs;
      outputs.resize(n_outputs);
      for (size_t i = 0; i < n_outputs; i++) {
        const auto output = graph->GetOutputs()[i];
        const auto output_shape = output->GetDims();
        std::vector<int64_t>
            int64_output_shape(output_shape.begin(), output_shape.end());
        const auto* output_tensor = ctx.GetOutput(
            rk_state->output_indexes[i],
            int64_output_shape.data(),
            int64_output_shape.size());
        float* output_buf = output_tensor.GetTensorMutableData<float>();

        const auto type = output->GetPrecision();
        size_t type_size = 4;
        switch (type) {
          case rk::nn::PrecisionType::FLOAT32:
            type_size = 4;
            break;
          case rk::nn::PrecisionType::UINT8:
            type_size = 1;
            break;
          case rk::nn::PrecisionType::INT32:
            type_size = 4;
            break;
          case rk::nn::PrecisionType::INT64:
            type_size = 8;
            break;
          default:
            // TODO
            throw std::invalid_argument(
                "compute_func: unknow output data type!");
            break;
        }

        outputs[i].index = i;
        outputs[i].buf = (void*)output_buf;
        outputs[i].size = accumulate(output_shape.begin(),
                                     output_shape.end(), 1,
                                     std::multiplies<uint32_t>()) *
                          type_size;
        outputs[i].type = type;
        outputs[i].layout = rk::nn::DataLayoutType::NCHW;
        outputs[i].want_float = false;
      }

      rk_state->exector->SetInputs(inputs);
      rk_state->exector->Run();
      rk_state->exector->GetOutputs(outputs);

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kRknpuExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kRknpuExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kRknpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kRknpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static void RegisterRknpuKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRknpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRknpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetRknpuKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry =
      std::make_shared<KernelRegistry>();
  RegisterRknpuKernels(*kernel_registry);

  return kernel_registry;
}

std::shared_ptr<KernelRegistry>
RknpuExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry =
      onnxruntime::GetRknpuKernelRegistry();
  return kernel_registry;
}

}  // namespace onnxruntime
