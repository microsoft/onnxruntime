// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/coreml_execution_provider.h"
#include "core/providers/coreml/coreml_provider_factory.h"  // defines flags

#include <algorithm>

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/providers/coreml/model/model.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/graph/model.h"

namespace onnxruntime {

constexpr const char* COREML = "CoreML";

CoreMLExecutionProvider::CoreMLExecutionProvider(const CoreMLOptions& options)
    : IExecutionProvider{onnxruntime::kCoreMLExecutionProvider},
      coreml_options_(options),
      coreml_version_(coreml::util::CoreMLVersion()) {
  LOGS_DEFAULT(VERBOSE) << "CoreML version: " << coreml_version_;
  if (coreml_version_ < MINIMUM_COREML_VERSION) {
    ORT_THROW("CoreML EP is not supported on this platform.");
  }
}

CoreMLExecutionProvider::~CoreMLExecutionProvider() {}

// Create a helper function for type checking
bool CoreMLExecutionProvider::IsSupportedType(int32_t type) const {
  return supported_input_output_types_.find(type) != supported_input_output_types_.end();
}

// After an input node is removed from the partition, we need to update the inputs of the partition.
// We evaluate the outputs of the input nodes
// For each output, if its consumer node is part of the partition, we add this output to the inputs of the partition
void UpdateInputsSetAfterNodeRemoval(std::unordered_set<NodeIndex>& partition_nodes_set,
                                     const Node* node,
                                     std::unordered_set<std::string>& newInputs,
                                     const onnxruntime::GraphViewer& graph_viewer) {
  for (const auto& output : node->OutputDefs()) {
    for (const auto& consumer_node : graph_viewer.GetConsumerNodes(output->Name())) {
      NodeIndex consumer_node_index = consumer_node->Index();
      if (partition_nodes_set.find(consumer_node_index) != partition_nodes_set.end() &&
          newInputs.find(output->Name()) == newInputs.end()) {
        newInputs.insert(output->Name());
      }
    }
  }
}

// we want to add the inputs of the removed node to the outputs of the partition IF:
// 1. the input is an output of a node that is part of the partition
// 2. the input is not already in the outputs of the partition
void UpdateOutputsSetAfterNodeRemoval(std::unordered_set<NodeIndex>& partition_nodes_set,
                                      const Node* node,
                                      std::unordered_set<std::string>& newOutputs,
                                      const onnxruntime::GraphViewer& graph_viewer) {
  for (const auto& input : node->InputDefs()) {
    const Node* producer_node_for_input = graph_viewer.GetProducerNode(input->Name());
    NodeIndex producer_index = producer_node_for_input ? producer_node_for_input->Index() : NodeIndex(-1);
    if (partition_nodes_set.find(producer_index) != partition_nodes_set.end() &&
        newOutputs.find(input->Name()) == newOutputs.end()) {
      newOutputs.insert(input->Name());
    }
  }
}

bool CoreMLExecutionProvider::ProcessIncompatibleNodes(const onnxruntime::GraphViewer& graph_viewer,
                                                       std::unordered_set<NodeIndex>& partition_nodes,
                                                       IndexedSubGraph::MetaDef* meta_def,
                                                       const bool is_input,
                                                       const logging::Logger& logger) const {
  std::unordered_set<std::string> new_names;
  bool update_meta_def = false;
  const size_t original_partition_size = partition_nodes.size();

  // Get the appropriate vector (inputs or outputs)
  auto& names = is_input ? meta_def->inputs : meta_def->outputs;

  for (const auto& name : names) {
    // Get node arg type
    const auto* node_arg = graph_viewer.GetNodeArg(name);
    int32_t type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
    if (!GetType(*node_arg, type, logger)) {
      LOGS(logger, ERROR) << "NodeArg " << name << " has no type information.";
      continue;
    }

    // Get relevant nodes (producers or consumers)
    std::vector<const Node*> nodes;
    if (is_input) {
      nodes = graph_viewer.GetConsumerNodes(name);
    } else {
      const Node* producer = graph_viewer.GetProducerNode(name);
      if (producer) nodes.push_back(producer);
    }

    // Process nodes based on type compatibility
    if (!IsSupportedType(type)) {
      for (const Node* node : nodes) {
        if (node && partition_nodes.find(node->Index()) != partition_nodes.end()) {
          update_meta_def = true;
          partition_nodes.erase(node->Index());

          // Update the appropriate set after node removal
          if (is_input) {
            UpdateInputsSetAfterNodeRemoval(partition_nodes, node, new_names, graph_viewer);
          } else {
            UpdateOutputsSetAfterNodeRemoval(partition_nodes, node, new_names, graph_viewer);
          }

          LOGS(logger, VERBOSE) << "Removing node " << node->Name()
                                << " from CoreML partition due to unsupported type: "
                                << ONNX_NAMESPACE::TensorProto_DataType_Name(
                                       (ONNX_NAMESPACE::TensorProto_DataType)type);
        }
      }
    } else {
      // Type is supported, check if nodes are in partition
      for (const Node* node : nodes) {
        if (node && partition_nodes.find(node->Index()) != partition_nodes.end()) {
          new_names.insert(name);
        } else {
          update_meta_def = true;
        }
      }
    }
  }

  // Update metadata if needed
  if (update_meta_def) {
    names = std::vector<std::string>(new_names.begin(), new_names.end());
  }

  return original_partition_size != partition_nodes.size();
}

void CoreMLExecutionProvider::FilterIncompatibleEdgeNodesFromPartition(IndexedSubGraph& partition,
                                                                       const onnxruntime::GraphViewer& graph_viewer,
                                                                       const logging::Logger& logger) const {
  IndexedSubGraph::MetaDef* meta_def = partition.GetMutableMetaDef();
  if (!meta_def) return;

  std::unordered_set<NodeIndex> partition_nodes(partition.nodes.begin(), partition.nodes.end());
  bool modified;

  do {
    bool inputs_modified = ProcessIncompatibleNodes(graph_viewer, partition_nodes, meta_def, true, logger);
    bool outputs_modified = ProcessIncompatibleNodes(graph_viewer, partition_nodes, meta_def, false, logger);

    modified = inputs_modified || outputs_modified;
    if (modified) {
      partition.nodes = std::vector<NodeIndex>(partition_nodes.begin(), partition_nodes.end());
    }
  } while (modified);
}

// CoreML only supports a limited set of inputs and outputs (int32 and float32), so we remove all edge nodes with incompatible types
std::vector<std::unique_ptr<ComputeCapability>> CoreMLExecutionProvider::FilterIncompatibleEdgeNodesFromPartitions(
    std::vector<std::unique_ptr<ComputeCapability>>&& capabilities,
    const onnxruntime::GraphViewer& graph_viewer,
    const logging::Logger& logger) const {
  std::vector<std::unique_ptr<ComputeCapability>> filtered_capabilities;

  for (auto& capability : capabilities) {
    if (capability && capability->sub_graph) {
      FilterIncompatibleEdgeNodesFromPartition(*capability->sub_graph, graph_viewer, logger);
      if (!capability->sub_graph->nodes.empty()) {
        filtered_capabilities.push_back(std::move(capability));
      }
    }
  }
  return filtered_capabilities;
}

std::vector<std::unique_ptr<ComputeCapability>>
CoreMLExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/,
                                       const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                       IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  const auto& logger = *GetLogger();

  // We do not run CoreML EP on subgraph, instead we cover this in the control flow nodes
  // TODO investigate whether we want to support subgraph using CoreML EP. May simply require processing the
  // implicit inputs of the control flow node that contains the subgraph as inputs to the CoreML model we generate.
  if (graph_viewer.IsSubgraph() && !coreml_options_.EnableOnSubgraph()) {
    return result;
  }

  const auto builder_params = coreml::MakeOpBuilderParams(graph_viewer, coreml_version_,
                                                          coreml_options_.RequireStaticShape(), coreml_options_.CreateMLProgram());
  const auto supported_nodes = coreml::GetSupportedNodes(graph_viewer, builder_params, logger);
  const Graph* main_graph = &graph_viewer.GetGraph();
  while (main_graph->IsSubgraph()) {
    main_graph = main_graph->ParentGraph();
  }
  const auto& metadata = main_graph->GetModel().MetaData();

  std::string user_provided_key = metadata.count(kCOREML_CACHE_KEY) > 0
                                      ? metadata.at(kCOREML_CACHE_KEY)
                                      : "";
  if (user_provided_key.size() > 64 ||
      std::any_of(user_provided_key.begin(), user_provided_key.end(),
                  [](unsigned char c) { return !std::isalnum(c); })) {
    LOGS(logger, ERROR) << "[" << kCOREML_CACHE_KEY << ":" << user_provided_key << "] is not a valid cache key."
                        << " It should be alphanumeric and less than 64 characters.";
    user_provided_key = "";
  }
  const auto gen_metadef_name =
      [&]() {
        HashValue model_hash;
        int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
        // use model_hash as the key if user doesn't provide one
        if (user_provided_key.empty()) {
          // user passed a empty string
          // model_hash is a 64-bit hash value of model_path if model_path is not empty,
          // otherwise it hashes the graph input names and all the node output names.
          // it can't guarantee the uniqueness of the key, so user should manager the key for the best.
          user_provided_key = std::to_string(model_hash);
        }
        // The string format is used by onnxruntime/core/providers/coreml/builders/model_builder.cc::GetModelOutputPath
        // If the format changes, the function should be updated accordingly.
        return MakeString(user_provided_key, "_", COREML, "_", model_hash, "_", metadef_id);
      };

  result = utils::CreateSupportedPartitions(graph_viewer, supported_nodes, {},
                                            gen_metadef_name, COREML, kCoreMLExecutionProvider,
                                            nullptr,
                                            /*drop_constant_initializers*/ true);

  std::vector<std::unique_ptr<ComputeCapability>> filtered_result =
      FilterIncompatibleEdgeNodesFromPartitions(std::move(result), graph_viewer, logger);

  const auto num_of_partitions = filtered_result.size();
  const auto num_of_supported_nodes = std::transform_reduce(
      filtered_result.begin(), filtered_result.end(),
      size_t{0}, std::plus<>{},
      [](const auto& partition) -> size_t {
        return partition && partition->sub_graph ? partition->sub_graph->nodes.size() : 0;
      });

  const auto summary_msg = MakeString(
      "CoreMLExecutionProvider::GetCapability,",
      " number of partitions supported by CoreML: ", num_of_partitions,
      " number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
      " number of nodes supported by CoreML: ", num_of_supported_nodes);

  // If the graph is partitioned in multiple subgraphs, and this may impact performance,
  // we want to give users a summary message at warning level.
  if (num_of_partitions > 1) {
    LOGS(logger, WARNING) << summary_msg;
  } else {
    LOGS(logger, INFO) << summary_msg;
  }

  return filtered_result;
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status CoreMLExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;

    std::unique_ptr<coreml::Model> coreml_model;
    {
      auto get_names = [](const ConstPointerContainer<std::vector<NodeArg*>>& args) -> std::vector<std::string> {
        std::vector<std::string> names;
        names.reserve(args.size());

        for (const NodeArg* def : args) {
          names.push_back(def->Name());
        }

        return names;
      };

      std::vector<std::string> onnx_input_names = get_names(fused_node.InputDefs());
      std::vector<std::string> onnx_output_names = get_names(fused_node.OutputDefs());

      const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
      ORT_RETURN_IF_ERROR(coreml::ModelBuilder::Build(graph_viewer, *GetLogger(), coreml_version_, coreml_options_,
                                                      std::move(onnx_input_names), std::move(onnx_output_names),
                                                      coreml_model));
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

    compute_info.compute_func = [](FunctionState state, const OrtApi* /* api */, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);
      const size_t num_inputs = ctx.GetInputCount();
      const size_t num_outputs = ctx.GetOutputCount();

      coreml::Model* model = reinterpret_cast<coreml::Model*>(state);

      // input/output names used by the CoreML model in the order that matches the fused_node InputDefs/OutputDefs
      const auto& model_inputs = model->GetOrderedInputs();
      const auto& model_outputs = model->GetOrderedOutputs();

      ORT_RETURN_IF_NOT(model_inputs.size() <= num_inputs, "Inconsistent input sizes");
      ORT_RETURN_IF_NOT(model_outputs.size() == num_outputs, "Inconsistent output sizes");

      std::unordered_map<std::string, coreml::OnnxTensorData> inputs;
      inputs.reserve(model_inputs.size());
      for (size_t i = 0; i < model_inputs.size(); i++) {
        const auto& input_name = model_inputs[i];
        const auto* input_info = model->TryGetInputOutputInfo(input_name);
        if (input_info == nullptr) {
          // The CoreML model may not have an actual input that corresponds to this one.
          // E.g., when the input is an initializer that already got copied to the CoreML model.
          // If there's no CoreML model input, we don't need to provide this input to CoreML.
          continue;
        }

        auto input_tensor = ctx.GetInput(i);
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        // Disallow inputs with dynamic shape which actually have zero elements.
        // CoreML doesn't consistently handle this well (e.g., there may be runtime errors).
        const auto& inferred_shape = input_info->shape;
        ORT_RETURN_IF(!coreml::IsStaticShape(inferred_shape) && coreml::DoesShapeSpecifyZeroElements(shape),
                      "Input (", input_name, ") has a dynamic shape (", coreml::Shape2String(inferred_shape),
                      ") but the runtime shape (", coreml::Shape2String(shape),
                      ") has zero elements. This is not supported by the CoreML EP.");

        // If we have an empty shape, this is a scalar input,
        // Since all the input output of CoreML EP is MultiArray, we will make the scalar input as a {1} MultiArray
        if (shape.empty()) {
          shape.push_back(1);
        }

        // CoreML MLMultiArray API expect input to be non-const
        // https://developer.apple.com/documentation/coreml/mlmultiarray/2881219-initwithdatapointer?language=objc
        void* inputBuffer = const_cast<void*>(input_tensor.GetTensorRawData());
        inputs.emplace(input_name, coreml::OnnxTensorData{
                                       coreml::OnnxTensorInfo{tensor_info.GetElementType(), shape},
                                       inputBuffer,
                                   });
      }

      // From this point we will need to take the exclusive lock on the model until the Predict is
      // performed, to block other threads to perform Predict on the same model
      // TODO, investigate concurrent runs for different executions from the same model
      {
        std::unique_lock<std::mutex> lock(model->GetMutex());
        std::unordered_map<std::string, coreml::OnnxTensorInfo> outputs;
        outputs.reserve(model_outputs.size());

        coreml::GetOutputTensorMutableRawDataFn get_output_tensor_mutable_raw_data_fn =
            [&ctx, &model_outputs](const std::string& name,
                                   int32_t requested_onnx_tensor_element_type,
                                   gsl::span<const int64_t> static_shape) -> void* {
          const auto model_output_it = std::find(model_outputs.begin(), model_outputs.end(), name);
          ORT_ENFORCE(model_output_it != model_outputs.end(), "Failed to find CoreML model output name: ", name);

          const auto output_idx = gsl::narrow_cast<size_t>(std::distance(model_outputs.begin(), model_output_it));
          auto output_tensor = ctx.GetOutput(output_idx, static_shape.data(), static_shape.size());

          const auto type_and_shape_info = output_tensor.GetTensorTypeAndShapeInfo();
          const auto actual_element_type = type_and_shape_info.GetElementType();
          ORT_ENFORCE(utils::CApiElementTypeFromProtoType(requested_onnx_tensor_element_type) == actual_element_type,
                      "Requested and actual output tensor element types do not match. Requested: ",
                      utils::CApiElementTypeFromProtoType(requested_onnx_tensor_element_type),
                      ", actual: ", actual_element_type);

          return output_tensor.GetTensorMutableRawData();
        };

        for (size_t i = 0; i < model_outputs.size(); i++) {
          const auto& output_name = model_outputs[i];
          const auto& output_info = model->GetInputOutputInfo(output_name);
          auto output_shape = output_info.shape;
          auto output_type = output_info.data_type;

          // Since CoreML EP use {1} MLMultiArray as scalar, if the model output should have empty shape
          // We are going to replace the {1} shape of the output back to {}
          if (model->IsScalarOutput(output_name)) {
            output_shape.clear();
          }

          // Since CoreML EP only accepts int32 output type and onnx requires int64 output,
          // We are going to set the model output (from int32) ->int64
          if (model->IsInt64Output(output_name)) {
            output_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
          }

          outputs.emplace(output_name, coreml::OnnxTensorInfo{output_type, output_shape});
        }

        return model->Predict(inputs, outputs, get_output_tensor_mutable_raw_data_fn);
      }
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

}  // namespace onnxruntime
