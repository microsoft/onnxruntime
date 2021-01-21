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

CoreMLExecutionProvider::CoreMLExecutionProvider()
    : IExecutionProvider{onnxruntime::kCoreMLExecutionProvider} {
  AllocatorCreationInfo device_info(
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(COREML, OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));

  AllocatorCreationInfo cpu_memory_info(
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
            OrtMemoryInfo(COREML, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      });

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

CoreMLExecutionProvider::~CoreMLExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
CoreMLExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_view,
                                       const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // We do not run CoreML EP on subgraph, instead we cover this in the control flow nodes
  if (graph_view.IsSubgraph()) {
    return result;
  }

  std::unordered_set<std::string> all_node_inputs;
  for (const auto& node : graph_view.Nodes()) {
    for (auto* input : node.InputDefs()) {
      all_node_inputs.insert(input->Name());
    }
  }

  const auto supported_nodes_vector = coreml::GetSupportedNodes(graph_view);

  // Find inputs, initializers and outputs for each supported subgraph
  size_t num_of_supported_nodes = 0;
  const std::vector<NodeIndex>& node_index = graph_view.GetNodesInTopologicalOrder();
  const auto& graph_outputs = graph_view.GetOutputs();
  for (const auto& group : supported_nodes_vector) {
    if (group.empty())
      continue;

    num_of_supported_nodes += group.size();
    LOGS_DEFAULT(VERBOSE) << "CoreMLExecutionProvider::GetCapability, current supported node group size: "
                          << group.size();

    std::unordered_set<size_t> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(node_index[index]);
    }

    std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
    // Find inputs and outputs of the subgraph
    std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add;
    std::unordered_set<const NodeArg*> erased;
    int input_order = 0;
    int output_order = 0;

    for (const auto& index : group) {
      sub_graph->nodes.push_back(node_index[index]);
      const auto* node = graph_view.GetNode(node_index[index]);

      for (const auto* input : node->InputDefs()) {
        const auto it = fused_outputs.find(input);
        if (it != fused_outputs.end()) {
          fused_outputs.erase(it);
          erased.insert(input);
        }
        //only when input is neither in output list nor erased list, add the input to input list
        else if (erased.find(input) == erased.end()) {
          fused_inputs[input] = input_order++;
        }
      }

      // For output searching, there is a special case:
      // If certain output is used more than once,
      // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
      // to the output list

      std::unordered_set<const NodeArg*> processed_outputs;
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        const auto node_idx = it->GetNode().Index();
        const auto* output = node->OutputDefs()[it->GetSrcArgIndex()];

        if (node_set.find(node_idx) != node_set.end()) {
          const auto iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            fused_outputs[output] = output_order++;
          }
        } else {
          fused_outputs_to_add[output] = output_order++;
        }

        processed_outputs.insert(output);
      }

      for (const auto* output : node->OutputDefs()) {
        if (processed_outputs.find(output) != processed_outputs.end())
          continue;

        const auto iter = fused_inputs.find(output);
        if (iter != fused_inputs.end()) {
          fused_inputs.erase(iter);
          erased.insert(output);
        }
        // only when output is neither in input list nor erased list, add the output to output list
        else if (erased.find(output) == erased.end() && output->Exists()) {
          fused_outputs[output] = output_order++;
        }
      }
    }

    fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
    // Sort inputs and outputs by the order they were added
    std::multimap<int, const NodeArg*> inputs, outputs;

    for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
      inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
    }

    for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
      if (all_node_inputs.find(it->first->Name()) != all_node_inputs.end()) {
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      } else if (std::find(graph_outputs.begin(), graph_outputs.end(), it->first) != graph_outputs.end()) {
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }
    }

    // Assign inputs and outputs to subgraph's meta_def
    auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = "COREML_" + std::to_string(metadef_id_++);
    meta_def->domain = kMSDomain;

    for (const auto& input : inputs) {
      meta_def->inputs.push_back(input.second->Name());
    }

    for (const auto& output : outputs) {
      meta_def->outputs.push_back(output.second->Name());
    }

    // meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    meta_def->since_version = 1;
    sub_graph->SetMetaDef(std::move(meta_def));

    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  LOGS_DEFAULT(INFO) << "CoreMLExecutionProvider::GetCapability,"
                     << " number of partitions supported by CoreML: " << result.size()
                     << " number of nodes in the graph: " << graph_view.NumberOfNodes()
                     << " number of nodes supported by CoreML: " << num_of_supported_nodes;

  return result;
}

common::Status CoreMLExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    coreml::ModelBuilder builder(graph_viewer);
    std::unique_ptr<coreml::Model> coreml_model;
    const std::string coreml_model_file_path = coreml::util::GetTemporaryFilePath();
    ORT_RETURN_IF_ERROR(builder.Compile(coreml_model, coreml_model_file_path));
    ORT_RETURN_IF_ERROR(coreml_model->LoadModel());

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
