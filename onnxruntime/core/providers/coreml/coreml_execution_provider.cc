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

namespace onnxruntime {

constexpr const char* COREML = "CoreML";

CoreMLExecutionProvider::CoreMLExecutionProvider(uint32_t coreml_flags)
    : IExecutionProvider{onnxruntime::kCoreMLExecutionProvider},
      coreml_flags_(coreml_flags),
      coreml_version_(coreml::util::CoreMLVersion()) {
  if (coreml_version_ < MINIMUM_COREML_VERSION) {
    LOGS_DEFAULT(ERROR) << "CoreML EP is not supported on this platform.";
  }

#if defined(COREML_ENABLE_MLPROGRAM)
  if (coreml_version_ < MINIMUM_COREML_MLPROGRAM_VERSION &&
      (coreml_flags_ & COREML_FLAG_CREATE_MLPROGRAM) != 0) {
    LOGS_DEFAULT(WARNING) << "ML Program is not supported on this OS version. Falling back to NeuralNetwork.";
    coreml_flags_ ^= COREML_FLAG_CREATE_MLPROGRAM;
  }
#else
  if ((coreml_flags_ & COREML_FLAG_CREATE_MLPROGRAM) != 0) {
    LOGS_DEFAULT(WARNING) << "ML Program is not supported in this build. Falling back to NeuralNetwork.";
    coreml_flags_ ^= COREML_FLAG_CREATE_MLPROGRAM;
  }
#endif
}

CoreMLExecutionProvider::~CoreMLExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
CoreMLExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (coreml_version_ < MINIMUM_COREML_VERSION) {
    return result;
  }

  const auto& logger = *GetLogger();

  // We do not run CoreML EP on subgraph, instead we cover this in the control flow nodes
  // TODO investigate whether we want to support subgraph using CoreML EP. May simply require processing the
  // implicit inputs of the control flow node that contains the subgraph as inputs to the CoreML model we generate.
  if (graph_viewer.IsSubgraph() && !(coreml_flags_ & COREML_FLAG_ENABLE_ON_SUBGRAPH)) {
    return result;
  }

  const bool has_neural_engine = coreml::HasNeuralEngine(logger);
  if ((coreml_flags_ & COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE) && !has_neural_engine) {
    LOGS(logger, WARNING) << "The current system does not have Apple Neural Engine. CoreML EP will not be used.";
    return result;
  }

  const auto builder_params = coreml::MakeOpBuilderParams(graph_viewer, coreml_version_, coreml_flags_);
  const auto supported_nodes = coreml::GetSupportedNodes(graph_viewer, builder_params, logger);

  const auto gen_metadef_name =
      [&]() {
        HashValue model_hash;
        int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
        return MakeString(COREML, "_", model_hash, "_", metadef_id);
      };

  result = utils::CreateSupportedPartitions(graph_viewer, supported_nodes, {},
                                            gen_metadef_name, COREML, kCoreMLExecutionProvider);

  const auto num_of_partitions = result.size();
  const auto num_of_supported_nodes = std::transform_reduce(
      result.begin(), result.end(),
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

  return result;
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
      ORT_RETURN_IF_ERROR(coreml::ModelBuilder::Build(graph_viewer, *GetLogger(), coreml_version_, coreml_flags_,
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
        std::unique_lock<OrtMutex> lock(model->GetMutex());
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
