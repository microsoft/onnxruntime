// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "coreml_execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "model/model.h"
#include "model/host_utils.h"
#include "builders/helper.h"
#include "builders/model_builder.h"

namespace onnxruntime {

constexpr const char* COREML = "CoreML";

CoreMLExecutionProvider::CoreMLExecutionProvider(uint32_t coreml_flags)
    : IExecutionProvider{onnxruntime::kCoreMLExecutionProvider, true},
      coreml_flags_(coreml_flags) {
  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(COREML, OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));

  AllocatorCreationInfo cpu_memory_info(
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(COREML, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      });

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

CoreMLExecutionProvider::~CoreMLExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
CoreMLExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // We do not run CoreML EP on subgraph, instead we cover this in the control flow nodes
  // TODO investigate whether we want to support subgraph using CoreML EP
  if (graph_viewer.IsSubgraph() && !(coreml_flags_ & COREML_FLAG_ENABLE_ON_SUBGRAPH)) {
    return result;
  }

  /*
  Very basic search for groups of nodes that can be handled by the EP.
  This doesn't work perfectly if you have a scenario like the following where A and D could be handled by the EP
  but B is between them in the topological sort as you'll get two single node capabilities. However if can also
  be advantageous if C and E could be handled by the EP as they would be combined with D even though not connected.
  Not sure how often each of these scenarios happens.

    A  B  C
    | /   |
    D     E
    |     |

  Would probably be better to walk the edges for each node the EP can handle as they are iterated in topological order,
  accumulating nodes (and saving which ones have been taken) until you run out. This would guarantee all
  connected nodes that can be handled are grouped together.
  */

  const auto& logger = *GetLogger();

  bool has_neural_engine = coreml::HasNeuralEngine(logger);
  if ((coreml_flags_ & COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE) && !has_neural_engine) {
    LOGS(logger, VERBOSE) << "The current system does not have Apple Neural Engine";
    return result;
  }

  const auto node_groups = coreml::GetSupportedNodes(graph_viewer, logger);

  if (node_groups.empty()) {
    return result;
  }

  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  size_t num_of_supported_nodes = 0;
  for (const auto& group : node_groups) {
    if (group.empty())
      continue;

    num_of_supported_nodes += group.size();
    LOGS(logger, VERBOSE) << "CoreMLExecutionProvider::GetCapability, current supported node group size: "
                          << group.size();

    std::unordered_set<NodeIndex> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(index);
    }

    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

    std::unordered_set<const NodeArg*> node_outputs;
    std::unordered_set<const NodeArg*> subgraph_inputs;
    std::unordered_set<const NodeArg*> subgraph_outputs;
    std::vector<const NodeArg*> ordered_subgraph_inputs;
    std::vector<const NodeArg*> ordered_subgraph_outputs;

    for (const auto& index : group) {
      sub_graph->nodes.push_back(index);
      const auto* node = graph_viewer.GetNode(index);

      for (const auto* input : node->InputDefs()) {
        // if the node input was not produced by this subgraph, add it to the subgraph inputs.
        if (node_outputs.count(input) == 0) {
          if (subgraph_inputs.count(input) == 0) {
            subgraph_inputs.insert(input);
            ordered_subgraph_inputs.push_back(input);
          }
        }
      }

      const auto& output_defs = node->OutputDefs();
      for (const auto* output_def : output_defs) {
        node_outputs.insert(output_def);
        // if output is overall graph output we need to produce it.
        if (graph_outputs.count(output_def) != 0) {
          ordered_subgraph_outputs.push_back(output_def);
        }
      }

      // if output connects to a node not in this subgraph we need to produce it
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        if (node_set.count(it->GetNode().Index()) == 0) {
          const auto* output_def = output_defs[it->GetSrcArgIndex()];
          if (subgraph_outputs.count(output_def) == 0) {
            subgraph_outputs.insert(output_def);
            ordered_subgraph_outputs.push_back(output_def);
          }
        }
      }
    }

    // Assign inputs and outputs to subgraph's meta_def
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = "COREML_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    meta_def->domain = kMSDomain;
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

    for (const auto& input : ordered_subgraph_inputs) {
      meta_def->inputs.push_back(input->Name());
    }

    for (const auto& output : ordered_subgraph_outputs) {
      meta_def->outputs.push_back(output->Name());
    }

    sub_graph->SetMetaDef(std::move(meta_def));

    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  LOGS(logger, INFO) << "CoreMLExecutionProvider::GetCapability,"
                     << " number of partitions supported by CoreML: " << result.size()
                     << " number of nodes in the graph: " << graph_viewer.NumberOfNodes()
                     << " number of nodes supported by CoreML: " << num_of_supported_nodes;

  return result;
}

common::Status CoreMLExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    coreml::ModelBuilder builder(graph_viewer, *GetLogger(), coreml_flags_);
    std::unique_ptr<coreml::Model> coreml_model;
    const std::string coreml_model_file_path = coreml::util::GetTemporaryFilePath();
    ORT_RETURN_IF_ERROR(builder.Compile(coreml_model, coreml_model_file_path));

    {
      const auto& input_defs = fused_node.InputDefs();
      std::vector<std::string> onnx_input_names(input_defs.size());
      for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
        onnx_input_names[i] = input_defs[i]->Name();
      }
      coreml_model->SetInputs(std::move(onnx_input_names));
    }

    {
      const auto& output_defs = fused_node.OutputDefs();
      std::vector<std::string> onnx_output_names(output_defs.size());
      for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
        onnx_output_names[i] = output_defs[i]->Name();
      }
      coreml_model->SetOutputs(std::move(onnx_output_names));
    }

    coreml_models_.emplace(fused_node.Name(), std::move(coreml_model));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = coreml_models_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      // the `state` is a coreml::model managed by unique_ptr
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      coreml::Model* model = reinterpret_cast<coreml::Model*>(state);
      const size_t num_inputs = ort.KernelContext_GetInputCount(context);
      const size_t num_outputs = ort.KernelContext_GetOutputCount(context);
      const auto& model_inputs = model->GetInputs();
      const auto& model_outputs = model->GetOutputs();

      ORT_RETURN_IF_NOT(model_inputs.size() <= num_inputs, "Inconsistent input sizes");
      ORT_RETURN_IF_NOT(model_outputs.size() == num_outputs, "Inconsistent output sizes");

      std::unordered_map<std::string, coreml::OnnxTensorData> inputs;
      inputs.reserve(model_inputs.size());
      for (size_t i = 0; i < model_inputs.size(); i++) {
        const auto& input_name = model_inputs[i];
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
        auto* tensor_info = ort.GetTensorTypeAndShape(input_tensor);

        auto shape = ort.GetTensorShape(tensor_info);
        // If we have an empty shape, this is a scalar input,
        // Since all the input output of CoreML EP is MultiArray, we will make the scalar input as a {1} MultiArray
        if (shape.empty())
          shape.push_back(1);

        const void* inputBuffer = ort.GetTensorData<void>(input_tensor);
        inputs.emplace(
            input_name,
            coreml::OnnxTensorData{
                coreml::OnnxTensorInfo{ort.GetTensorElementType(tensor_info), shape},
                // CoreML MLMultiArray API expect input to be non-const
                // https://developer.apple.com/documentation/coreml/mlmultiarray/2881219-initwithdatapointer?language=objc
                const_cast<void*>(inputBuffer),
            });

        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }

      // From this point we will need to take the exclusive lock on the model until the Predict is
      // performed, to block other threads to perform Predict on the same model
      // TODO, investigate concurrent runs for different executions from the same model
      {
        std::unique_lock<OrtMutex> lock(model->GetMutex());
        std::unordered_map<std::string, coreml::OnnxTensorData> outputs;
        outputs.reserve(model_outputs.size());
        for (size_t i = 0; i < model_outputs.size(); i++) {
          const auto& output_name = model_outputs[i];
          const auto& output_info = model->GetInputOutputInfo(output_name);
          auto output_shape = output_info.shape;
          auto output_type = output_info.data_type;

          // Since CoreML EP use {1} MLMultiArray as scalar, if the model output should have empty shape
          // We are going to replace the {1} shape of the output back to {}
          if (model->IsScalarOutput(output_name))
            output_shape.clear();

          auto* output_tensor =
              ort.KernelContext_GetOutput(context, i, output_shape.data(), output_shape.size());

          void* output_buffer;
          switch (output_type) {
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
              output_buffer = ort.GetTensorMutableData<float>(output_tensor);
              break;
            default:
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                     "Unsupported type: ", output_type, " for output: ", output_name);
              break;
          }

          outputs.emplace(output_name,
                          coreml::OnnxTensorData{
                              coreml::OnnxTensorInfo{output_type, output_shape},
                              output_buffer,
                          });
        }

        return model->Predict(inputs, outputs);
      }
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

}  // namespace onnxruntime
