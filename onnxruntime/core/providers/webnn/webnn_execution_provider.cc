// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "webnn_execution_provider.h"

#include "core/framework/compute_capability.h"
#include "core/framework/memcpy.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/safeint.h"

#include "builders/model.h"
#include "builders/helper.h"
#include "builders/model_builder.h"

namespace onnxruntime {

WebNNExecutionProvider::WebNNExecutionProvider(
    const std::string& webnn_device_flags, const std::string& webnn_power_flags)
    : IExecutionProvider{onnxruntime::kWebNNExecutionProvider, true} {
  // Create WebNN context and graph builder.
  const emscripten::val ml = emscripten::val::global("navigator")["ml"];
  if (!ml.as<bool>()) {
    ORT_THROW("Failed to get ml from navigator.");
  }
  emscripten::val context_options = emscripten::val::object();
  // Currently WebNN implementation in Chromium temporarily reuses the MLContextOptions
  // defined in Model Loader API, which uses MLDevicePreference instead of MLDeviceType
  // defined in WebNN. Because there's an ongoing spec discussion to simplify this API at
  // https://github.com/webmachinelearning/webnn/issues/302.
  context_options.set("devicePreference", emscripten::val(webnn_device_flags));
  // WebNN EP uses NHWC layout for CPU XNNPACK backend and NCHW for GPU DML backend.
  if (webnn_device_flags.compare("cpu") == 0) {
    preferred_layout_ = DataLayout::NHWC;
    wnn_device_type_ = webnn::WebnnDeviceType::CPU;
  } else {
    preferred_layout_ = DataLayout::NCHW;
    wnn_device_type_ = webnn::WebnnDeviceType::GPU;
  }
  if (webnn_power_flags.compare("default") != 0) {
    context_options.set("powerPreference", emscripten::val(webnn_power_flags));
  }
  wnn_context_ = ml.call<emscripten::val>("createContextSync", context_options);
  if (!wnn_context_.as<bool>()) {
    ORT_THROW("Failed to create WebNN context.");
  }
  wnn_builder_ = emscripten::val::global("MLGraphBuilder").new_(wnn_context_);
  if (!wnn_builder_.as<bool>()) {
    ORT_THROW("Failed to create WebNN builder.");
  }
}

WebNNExecutionProvider::~WebNNExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
WebNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                      const IKernelLookup& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // We do not run WebNN EP on subgraph, instead we cover this in the control flow nodes.
  // TODO investigate whether we want to support subgraph using WebNN EP.
  if (graph_viewer.IsSubgraph()) {
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

  const auto node_groups = webnn::GetSupportedNodes(graph_viewer, wnn_builder_, wnn_device_type_, logger);

  if (node_groups.empty()) {
    return result;
  }

  const auto& graph_output_list = graph_viewer.GetOutputs();
  InlinedHashSet<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  size_t num_of_supported_nodes = 0;
  for (const auto& group : node_groups) {
    if (group.empty())
      continue;

    num_of_supported_nodes += group.size();
    LOGS(logger, VERBOSE) << "WebNNExecutionProvider::GetCapability, current supported node group size: "
                          << group.size();

    InlinedHashSet<NodeIndex> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(index);
    }

    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

    InlinedHashSet<const NodeArg*> node_outputs;
    InlinedHashSet<const NodeArg*> subgraph_inputs;
    InlinedHashSet<const NodeArg*> subgraph_outputs;
    std::vector<const NodeArg*> ordered_subgraph_inputs;
    // Output should be unique. It may be produced as graph output and subgraph output.
    InlinedHashSet<const NodeArg*> ordered_subgraph_outputs;

    for (const auto& index : group) {
      sub_graph->nodes.push_back(index);
      const auto* node = graph_viewer.GetNode(index);

      for (const auto* input : node->InputDefs()) {
        if (!input->Exists()) {
          // skip the placeholder inputs.
          continue;
        }
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
          ordered_subgraph_outputs.insert(output_def);
        }
      }

      // if output connects to a node not in this subgraph we need to produce it.
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        if (node_set.count(it->GetNode().Index()) == 0) {
          const auto* output_def = output_defs[it->GetSrcArgIndex()];
          if (subgraph_outputs.count(output_def) == 0) {
            subgraph_outputs.insert(output_def);
            ordered_subgraph_outputs.insert(output_def);
          }
        }
      }
    }

    // Assign inputs and outputs to subgraph's meta_def.
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = "WEBNN_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
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

  auto num_of_partitions = result.size();
  const auto summary_msg = MakeString(
      "WebNNExecutionProvider::GetCapability,",
      " number of partitions supported by WebNN: ", num_of_partitions,
      " number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
      " number of nodes supported by WebNN: ", num_of_supported_nodes);

  // If the graph is partitioned in multiple subgraphs, and this may impact performance,
  // we want to give users a summary message at warning level.
  if (num_of_partitions > 1) {
    LOGS(logger, WARNING) << summary_msg;
  } else {
    LOGS(logger, INFO) << summary_msg;
  }

  return result;
}

common::Status WebNNExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    webnn::ModelBuilder builder(graph_viewer, *GetLogger(), wnn_context_,
                                wnn_builder_, preferred_layout_, wnn_device_type_);
    std::unique_ptr<webnn::Model> model;
    ORT_RETURN_IF_ERROR(builder.Compile(model));
    // Build map from input name to its index in input definitions.
    {
      InlinedHashMap<std::string, size_t> input_map;
      const auto& input_defs = fused_node.InputDefs();
      input_map.reserve(input_defs.size());
      for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
        input_map[input_defs[i]->Name()] = i;
      }
      model->SetInputMap(std::move(input_map));
    }
    // Build map from output name to its index in output definitions.
    {
      InlinedHashMap<std::string, size_t> output_map;
      const auto& output_defs = fused_node.OutputDefs();
      output_map.reserve(output_defs.size());
      for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
        output_map[output_defs[i]->Name()] = i;
      }
      model->SetOutputMap(std::move(output_map));
    }
    models_.emplace(fused_node.Name(), std::move(model));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = models_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      // The `state` is a webnn::model managed by unique_ptr.
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);

      const size_t num_inputs = ctx.GetInputCount();
      const size_t num_outputs = ctx.GetOutputCount();

      webnn::Model* model = reinterpret_cast<webnn::Model*>(state);

      const auto& model_inputs = model->GetInputs();
      const auto& model_outputs = model->GetOutputs();

      ORT_RETURN_IF_NOT(model_inputs.size() <= num_inputs, "Inconsistent input sizes");
      ORT_RETURN_IF_NOT(model_outputs.size() == num_outputs, "Inconsistent output sizes");

      InlinedHashMap<std::string, webnn::OnnxTensorData> inputs;
      inputs.reserve(model_inputs.size());
      for (size_t i = 0; i < model_inputs.size(); i++) {
        const auto& input_name = model_inputs[i];
        auto input_idx = model->GetMappedInputIdx(input_name);
        auto input_tensor = ctx.GetInput(input_idx);
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        // If we have an empty shape, this is a scalar input,
        // Since all the input output of WebNN EP is MultiArray, we will make the scalar input as a {1} MultiArray.
        if (shape.empty())
          shape.push_back(1);
        std::vector<int> temp(shape.size());
        transform(shape.begin(), shape.end(), temp.begin(),
                  [](int64_t dim) -> uint32_t { return SafeInt<int32_t>(dim); });
        const void* inputBuffer = const_cast<void*>(input_tensor.GetTensorRawData());
        inputs.emplace(
            input_name,
            webnn::OnnxTensorData{
                webnn::OnnxTensorInfo{tensor_info.GetElementType(), shape},
                const_cast<void*>(inputBuffer),
            });
      }

      // From this point we will need to take the exclusive lock on the model until the Predict is
      // performed, to block other threads to perform Predict on the same model.
      // TODO, investigate concurrent runs for different executions from the same model.
      {
        std::unique_lock<OrtMutex> lock(model->GetMutex());
        InlinedHashMap<std::string, webnn::OnnxTensorData> outputs;
        outputs.reserve(model_outputs.size());
        for (size_t i = 0; i < model_outputs.size(); i++) {
          const auto& output_name = model_outputs[i];
          const auto& output_info = model->GetInputOutputInfo(output_name);
          auto output_shape = output_info.shape;
          auto output_type = output_info.data_type;

          // Since WebNN EP use {1} tensor as scalar, if the model output should have empty shape.
          // We are going to replace the {1} shape of the output back to {}.
          if (model->IsScalarOutput(output_name))
            output_shape.clear();

          auto output_tensor =
              ctx.GetOutput(i, output_shape.data(), output_shape.size());

          void* output_buffer;
          switch (output_type) {
            case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
            case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
            case ONNX_NAMESPACE::TensorProto_DataType_INT32:
            case ONNX_NAMESPACE::TensorProto_DataType_INT64:
            case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
            case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
              output_buffer = output_tensor.GetTensorMutableRawData();
              break;
            default:
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                     "Unsupported type: ", output_type, " for output: ", output_name);
              break;
          }

          outputs.emplace(output_name,
                          webnn::OnnxTensorData{
                              webnn::OnnxTensorInfo{output_type, output_shape},
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

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kWebNNExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kWebNNExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static void RegisterWebNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebNNExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_ENFORCE(kernel_registry.Register(function_table_entry()).IsOK());
  }
}

std::shared_ptr<KernelRegistry> GetWebNNKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry =
      std::make_shared<KernelRegistry>();
  RegisterWebNNKernels(*kernel_registry);

  return kernel_registry;
}

std::shared_ptr<KernelRegistry>
WebNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry =
      onnxruntime::GetWebNNKernelRegistry();
  return kernel_registry;
}

}  // namespace onnxruntime
