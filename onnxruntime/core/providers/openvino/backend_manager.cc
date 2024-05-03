// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <array>
#include <cassert>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {

GlobalContext& BackendManager::GetGlobalContext() {
  return global_context_;
}

BackendManager::BackendManager(const GlobalContext& global_context,
                               const onnxruntime::Node& fused_node,
                               const onnxruntime::GraphViewer& subgraph,
                               const logging::Logger& logger,
                               EPCtxHandler& ctx_handle) {
  global_context_ = global_context;
  ep_ctx_handle_ = ctx_handle;

  openvino_sdk_version_ = std::to_string(global_context_.OpenVINO_Version.at(0)) + "." +
                          std::to_string(global_context_.OpenVINO_Version.at(1));
  if (ep_ctx_handle_.CheckForOVEPCtxNode(subgraph, openvino_sdk_version_)) {
    if (ep_ctx_handle_.ImportBlobFromEPCtxModel(subgraph) != Status::OK())
      ORT_THROW("Import blob from model failed");
  }

  auto prec_str = GetGlobalContext().precision_str;

  // Save the indexes of graph inputs among fused_node's inputDefs
  // (which also contains initializers).
  auto node_input_defs = fused_node.InputDefs();
  int i = 0;
  for (auto idef : node_input_defs) {
    subgraph_context_.input_names.insert({idef->Name(), i});
    i++;
  }

  auto graph_inputs = subgraph.GetInputs();
  for (auto input : graph_inputs) {
    auto it = subgraph_context_.input_names.find(input->Name());
    if (it == subgraph_context_.input_names.end()) {
      ORT_THROW("Input not found in the input defs list");
    }
    int index = it->second;
    subgraph_context_.input_indexes.push_back(index);
  }

  auto graph_outputs_defs = fused_node.OutputDefs();
  i = 0;
  for (auto output_def : graph_outputs_defs) {
    subgraph_context_.output_names.insert({output_def->Name(), i});
    i++;
  }
  subgraph_context_.subgraph_name = fused_node.Name();
  model_proto_ = GetModelProtoFromFusedNode(fused_node, subgraph, logger);
  std::string device_type = openvino_ep::BackendManager::GetGlobalContext().device_type;

  if (ModelHasSymbolicInputDims(subgraph)) {
    subgraph_context_.has_dynamic_input_shape = true;
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has symbolic input dims";
    if (GetGlobalContext().device_type.find("CPU") != std::string::npos ||
        GetGlobalContext().device_type.find("GPU") != std::string::npos) {
      if (!GetGlobalContext().disable_dynamic_shapes) {
        LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Starting backend initialization. "
                           << "Creating backend Dynamic Shapes";
        try {
          concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                          GetGlobalContext(),
                                                          subgraph_context_,
                                                          ep_ctx_handle_);
        } catch (std::string const& msg) {
          ORT_THROW(msg);
        }
        LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                           << "Backend created for graph " << subgraph_context_.subgraph_name;
      }
    }
  } else {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model has concrete input dims. "
                       << "Initializing backend for graph "
                       << subgraph_context_.subgraph_name;

    subgraph_context_.has_dynamic_input_shape = false;

    // OV NPU plugin is supported with fallback to OV CPU upon compilation failures.
    try {
      concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                      GetGlobalContext(),
                                                      subgraph_context_,
                                                      ep_ctx_handle_);
    } catch (const OnnxRuntimeException& ex) {
      if (device_type.find("NPU") != std::string::npos) {
        LOGS_DEFAULT(WARNING) << ex.what();
        LOGS_DEFAULT(WARNING) << "Model compilation failed at OV NPU."
                              << "Falling back to OV CPU for execution";
        GetGlobalContext().device_type = "CPU";
        GetGlobalContext().precision_str = "FP32";
        try {
          concrete_backend_ = BackendFactory::MakeBackend(*model_proto_,
                                                          GetGlobalContext(),
                                                          subgraph_context_,
                                                          ep_ctx_handle_);
        } catch (std::string const& msg) {
          ORT_THROW(msg);
        }
      } else {
        ORT_THROW(ex.what());
      }
    }
  }
}

// Call EPContext model exporter here if the provider option for exporting
// precompiled blob is set. If that's the case:
// By default, create model in embed mode where the blob stream is exported as data within
// the EPContext node.
Status BackendManager::ExportCompiledBlobAsEPCtxNode(const onnxruntime::GraphViewer& graph_body_viewer,
                                                     const logging::Logger& logger) {
  if (GetGlobalContext().disable_dynamic_shapes && subgraph_context_.has_dynamic_input_shape) {
    std::string exception_str =
        "Exporting dynamically compiled models at runtime is not supported. "
        "Cannot export blobs of dynamic models that request static shape inference. "
        "To export this model, set disable_dynamic_shapes to False";
    ORT_THROW(exception_str);
  }

  std::string model_blob_str;
  auto compiled_model = concrete_backend_->GetOVCompiledModel();
  auto graph_name = global_context_.onnx_model_path_name;
  // Remove extension so we can append suffix to form the complete name of output graph
  graph_name = [&]() {
    size_t dot = graph_name.find_last_of(".");
    if (dot == std::string::npos) return graph_name;
    return graph_name.substr(0, dot);
  }();
  // If embed_mode, then pass on the serialized blob
  // If not embed_mode, dump the blob here and only pass on the path to the blob
  if (global_context_.ep_context_embed_mode) {
    std::ostringstream model_blob_stream;
    compiled_model.export_model(model_blob_stream);
    model_blob_str = model_blob_stream.str();
    ORT_ENFORCE(model_blob_str.size() != 0);
  } else {
    std::ofstream f(graph_name + ".blob", std::ios::out | std::ios::trunc | std::ios::binary);
    compiled_model.export_model(f);
    model_blob_str = graph_name + ".blob";
  }

  ORT_RETURN_IF_ERROR(ep_ctx_handle_.ExportEPCtxModel(graph_body_viewer,
                                                      graph_name,
                                                      logger,
                                                      global_context_.ep_context_embed_mode,
                                                      model_blob_str,
                                                      openvino_sdk_version_,
                                                      GetGlobalContext().device_type));

  return Status::OK();
}

bool BackendManager::ModelHasBatchedInputs(const ONNX_NAMESPACE::ModelProto& model_proto) const {
  bool has_batched_inputs = true;

  for (int i = 0; i < static_cast<int>(subgraph_context_.input_indexes.size()); i++) {
    auto& input = model_proto.graph().input(subgraph_context_.input_indexes[i]);

    // Batch-process only raw image inputs (NCHW or NHWC layouts)
    auto& shape = input.type().tensor_type().shape();
    if (shape.dim_size() != 4) {
      has_batched_inputs = false;
      break;
    }

    if (shape.dim(0).value_case() == shape.dim(0).kDimValue) {
      has_batched_inputs = false;
      break;
    }

    for (int index = 1; index < 4; index++) {
      if (shape.dim(index).value_case() != shape.dim(0).kDimValue) {
        has_batched_inputs = false;
        break;
      }
    }
    if (!has_batched_inputs) {
      break;
    }
  }
  return has_batched_inputs;
}

bool BackendManager::ModelHasSymbolicInputDims(const onnxruntime::GraphViewer& subgraph) const {
  bool has_sym_dims = false;
  auto graph_inputs = subgraph.GetInputs();
  for (auto input : graph_inputs) {
    if (input->Shape() == nullptr) {
      has_sym_dims = true;
      break;
    }
    for (auto& dim : input->Shape()->dim()) {
      if (dim.value_case() != dim.kDimValue) {
        has_sym_dims = true;
        break;
      }
    }
    if (has_sym_dims) {
      break;
    }
  }
  return has_sym_dims;
}

// Creates a new NodeArg from an input (or output) of a QDQ node unit. If the input/output is quantized,
// this function modifies the tensor type to the specified float type.
static NodeArg& ProcessNodeUnitIO(onnxruntime::Graph& dst_graph, const onnxruntime::GraphViewer& src_graph,
                                  std::unordered_map<std::string, NodeUnitIODef>& initializers_to_dequant,
                                  const NodeUnitIODef& io_def, int float_type) {
  const std::string& name = io_def.node_arg.Name();
  const ONNX_NAMESPACE::TypeProto* orig_type_proto = io_def.node_arg.TypeAsProto();

  // Handle quantized input or output. Convert to float type.
  if (io_def.quant_param.has_value()) {
    // Copy the original quantized type proto, but update the type to float.
    std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = ONNX_NAMESPACE::TypeProto::Create();
    type_proto->copy_from(orig_type_proto);
    type_proto->mutable_tensor_type()->set_elem_type(float_type);

    // Handle initializer inputs.
    // By default QDQ models store quantized weights that are dequantized.
    // Ex: weight(int8) -> DequantizeLinear (to float) ->
    // Keep track of these initializers so the EP can dequantize them later.
    if (src_graph.GetAllInitializedTensors().count(name)) {
      initializers_to_dequant.insert({name, io_def});
    }

    return dst_graph.GetOrCreateNodeArg(name, type_proto.get());
  }

  // Unquantized input or output. Just copy.
  return dst_graph.GetOrCreateNodeArg(name, orig_type_proto);
}

// Handles adding a standalone node unit (i.e., one not wrapped with DQ/Q ops) to the dst graph.
static void AddStandaloneNodeUnit(onnxruntime::Graph& dst_graph, const onnxruntime::GraphViewer& src_graph,
                                  const NodeUnit& node_unit, int32_t float_type,
                                  std::unordered_map<std::string, NodeUnitIODef>& initializers_to_dequant,
                                  const logging::Logger& /* logger */) {
  assert(node_unit.UnitType() == NodeUnit::Type::SingleNode);

  if (node_unit.OpType() == "QuantizeLinear" || node_unit.OpType() == "DequantizeLinear") {
    // Standalone Q (or DQ) operator. Typically seen at the input (or output) of the graph.
    // Can replace with an Identity op for this demo. Should probably do something better.
    std::array<NodeArg*, 1> input_args = {&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                                             node_unit.Inputs()[0], float_type)};
    std::array<NodeArg*, 1> output_args = {&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                                              node_unit.Outputs()[0], float_type)};
    dst_graph.AddNode(node_unit.Name(),
                      "Identity",
                      "",
                      input_args,
                      output_args,
                      nullptr,
                      kOnnxDomain);

    // TODO: Another scenario to consider is a conversion between quantized types (e.g., int16 to int8)
    // in mixed-precision QDQ models. The pattern to detect is a standalone DQ followed by a standalone Q:
    // Ex: DQ(int16 to float) -> QuantizeLinear(float to int8) ->
    // Replacing both with identity ops (as we do here) may just work.
  } else {
    dst_graph.AddNode(node_unit.GetNode());  // Copy standalone node unit.
  }
}

static const std::vector<std::string> supported_qdq_ops = {"Conv", "Add", "Div", "MatMul"};
enum class SkipReason {
  Uint16QDQ,
  DuplicateDQ,
  Other
};
static bool CheckDQRuleSet(const NodeUnit& node_unit,
                           const Node* dq_node,
                           SkipReason& reason) {
  const auto& dq_input_defs = dq_node->InputDefs();
  const auto& target_input_defs = node_unit.GetNode().InputDefs();

  // zero point is the third input
  auto zero_point_dt = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
      dq_input_defs.at(2)->TypeAsProto()->tensor_type().elem_type());
  auto op_type = node_unit.OpType();

  // #1 If UInt16 DQ, don't keep it
  if (zero_point_dt == ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    reason = SkipReason::Uint16QDQ;
    return false;
  }

  // #2 Reverse DQ duplication
  if (dq_node->Name().find("/duplicated") != std::string::npos) {
    reason = SkipReason::DuplicateDQ;
    return false;
  }

  // Keep DQ for all supported Ops according to the following rules
  if (op_type == "Conv") {
    // This is the DQ we want to keep, so just return as it's not Uint16
    return true;
  } else if (op_type == "Add") {
    if ((target_input_defs.at(0)->Name().find("DequantizeLinear_Output") != std::string::npos) &&
        (target_input_defs.at(1)->Name().find("DequantizeLinear_Output") != std::string::npos)) {
      if (target_input_defs.at(0)->Name().find("bias") != std::string::npos) {
        reason = SkipReason::Other;
        return false;
      } else {
        // keeps both DQ inputs for this Add
        return true;
      }
    }
  } else if (op_type == "Div") {
    return true;
  } else if (op_type == "MatMul") {
    if ((target_input_defs.at(0)->Name().find("Softmax") == std::string::npos) &&
        (target_input_defs.at(1)->Name().find("Softmax") == std::string::npos)) {
      ORT_ENFORCE(target_input_defs.size() == 2);
      ORT_ENFORCE((target_input_defs.at(0)->Name().find("DequantizeLinear_Output") != std::string::npos) &&
                  (target_input_defs.at(1)->Name().find("DequantizeLinear_Output") != std::string::npos));
      return true;  // Keep both DQ inputs for this MatMul
    }
  } else {
    // For unsupported ops, check if connected input NodeUnit is one of supported ops, then keep the DQ
    for (Node::NodeConstIterator dq_in = dq_node->InputNodesBegin(); dq_in != dq_node->InputNodesEnd(); ++dq_in) {
      const auto& connected_q_op = *dq_in;

      // Node preceding a DQ should be a Q
      ORT_ENFORCE(connected_q_op.OpType() == "QuantizeLinear");

      for (Node::NodeConstIterator previous_target = connected_q_op.InputNodesBegin();
           previous_target != connected_q_op.InputNodesEnd(); ++previous_target) {
        const auto& previous_target_op = *previous_target;

        if (std::find(supported_qdq_ops.begin(),
                      supported_qdq_ops.end(), previous_target_op.OpType()) != supported_qdq_ops.end()) {
          return true;
        } else {
          reason = SkipReason::Other;
          return false;
        }
      }
    }
  }
  reason = SkipReason::Other;
  return false;
}

static bool CheckQRuleSet(const NodeUnit& node_unit,
                          const Node* q_node,
                          SkipReason& reason) {
  // If the target node of the NodeUnit following this one is one of the supported Op types, then keep this Q
  // This Q should also be uint8

  const auto& q_input_defs = q_node->InputDefs();
  const auto& target_input_defs = node_unit.GetNode().InputDefs();
  const auto& target_output_defs = node_unit.GetNode().OutputDefs();

  // zero point is the third input
  auto zero_point_dt = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
      q_input_defs.at(2)->TypeAsProto()->tensor_type().elem_type());
  auto op_type = node_unit.OpType();

  // If UInt16 Q, don't keep it
  if (zero_point_dt == ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    reason = SkipReason::Uint16QDQ;
    return false;
  }

  // Keep Q for all supported Ops according to the following rules
  if (op_type == "Conv") {
    // This is the Q we want to keep, so just return as it's not Uint16
    return true;
  } else if (op_type == "Add") {
    if ((target_input_defs.at(0)->Name().find("DequantizeLinear_Output") != std::string::npos) &&
        (target_input_defs.at(1)->Name().find("DequantizeLinear_Output") != std::string::npos)) {
      ORT_ENFORCE(target_output_defs.size() == 1);
      return true;
    }
  } else if (op_type == "Div") {
    return true;
  } else if (op_type == "MatMul") {
    if (target_input_defs.at(0)->Name().find("Softmax") == std::string::npos &&
        target_input_defs.at(1)->Name().find("Softmax") == std::string::npos) {
      ORT_ENFORCE(target_output_defs.size() == 1);
      return true;
    }
  } else {
    // If connected output is one of supported ops, keep the Q
    for (Node::NodeConstIterator q_out = q_node->OutputNodesBegin(); q_out != q_node->OutputNodesEnd(); ++q_out) {
      const auto& connected_dq_op = *q_out;

      // Node following a Q should be a DQ
      ORT_ENFORCE(connected_dq_op.OpType() == "DequantizeLinear");

      for (Node::NodeConstIterator next_target = connected_dq_op.OutputNodesBegin();
           next_target != connected_dq_op.OutputNodesEnd(); ++next_target) {
        const auto& next_target_op = *next_target;

        if (std::find(supported_qdq_ops.begin(),
                      supported_qdq_ops.end(), next_target_op.OpType()) != supported_qdq_ops.end()) {
          return true;
        } else {
          return false;
        }
      }
    }
  }
  return false;
}

// Handles adding a QDQ node unit (e.g., DQ -> Add -> Q) to the dst graph.
// Only adds the QDQ node unit's float-precision target node.
static void AddQDQNodeUnit(onnxruntime::Graph& dst_graph,
                           const onnxruntime::GraphViewer& src_graph,
                           const NodeUnit& node_unit,
                           int32_t float_type,
                           std::unordered_map<std::string, NodeUnitIODef>& initializers_to_dequant,
                           const logging::Logger& /* logger */) {
  assert(node_unit.UnitType() == NodeUnit::Type::QDQGroup);

  // LOGS_DEFAULT(INFO) << "\nNode in QDQ Group: " << node_unit.Name() << " " << node_unit.OpType() << " ";

  // Collect inputs coming into the node unit.
  const auto& node_unit_inputs = node_unit.Inputs();
  std::vector<NodeArg*> input_args;
  input_args.reserve(node_unit_inputs.size());

  std::unordered_map<std::string, NodeArg*> dq_node_args_to_keep;  // These DQ nodes will be retained in the graph

  for (auto dq_node : node_unit.GetDQNodes()) {
    const auto& input_defs = dq_node->InputDefs();
    ORT_ENFORCE(input_defs.size() == 3);

    SkipReason reason;
    bool keep_dq = CheckDQRuleSet(node_unit, dq_node, reason);

    if (keep_dq) {
      dst_graph.AddNode(*dq_node);  // Add the node to the graph
      dq_node_args_to_keep.insert({input_defs.at(0)->Name(),
                                   &dst_graph.GetOrCreateNodeArg(dq_node->OutputDefs().at(0)->Name(),
                                                                 dq_node->OutputDefs().at(0)->TypeAsProto())});
    } else {

      // If it's a duplicate DQ AND the previous node unit is not of type SingleNode (i.e, input/output units)
      if (reason == SkipReason::DuplicateDQ) {
        // Skips the DQ, but keep the route to the target node
        // Add the output of the other original DQ as input arg of this target node

        std::vector<const Node*> dst_nodes_present = dst_graph.Nodes();

        // Check if the connected Q was already kept in the dst graph. If it's not found, don't reconnect
        if (auto it = std::find_if(dst_nodes_present.begin(), dst_nodes_present.end(),
                                   [&](const Node* n) {
                                     // search for the connected Q in the dst graph
                                     return (n->Name() == dq_node->InputNodesBegin()->Name() &&
                                             n->OpType() == "QuantizeLinear");
                                   });
            it == std::end(dst_nodes_present)) continue;  // Skip connecting this duplicate DQ

        std::string target_in_arg_name = dq_node->OutputDefs().at(0)->Name();
        std::string duplicate_str = "/duplicated";
        target_in_arg_name.erase(target_in_arg_name.end() - duplicate_str.length(), target_in_arg_name.end());
        dq_node_args_to_keep.insert({input_defs.at(0)->Name(),
                                     &dst_graph.GetOrCreateNodeArg(target_in_arg_name,
                                                                   dq_node->OutputDefs().at(0)->TypeAsProto())});
      }
    }
  }

  // Add Node args for inputs
  for (const auto& node_unit_input : node_unit_inputs) {
    auto node_arg_name = node_unit_input.node_arg.Name();
    if (auto dq_node_arg = dq_node_args_to_keep.find(node_arg_name); dq_node_arg != dq_node_args_to_keep.end()) {
      // Add supported DQ as an input arg for the target node
      input_args.push_back(dq_node_arg->second);
    } else {
      // Otherwise, convert to float
      NodeArg& input_arg = ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                             node_unit_input, float_type);
      input_args.push_back(&input_arg);
    }
  }

  const Node& target_node = node_unit.GetNode();

  // Collect outputs coming out of the node unit.
  const auto& node_unit_outputs = node_unit.Outputs();
  std::vector<NodeArg*> output_args;
  output_args.reserve(node_unit_outputs.size());

  if (!node_unit.GetQNodes().empty()) {
    ORT_ENFORCE(node_unit.GetQNodes().size() == 1);
    const auto& q_node = node_unit.GetQNodes().at(0);

    SkipReason reason;

    bool keep_q = CheckQRuleSet(node_unit, q_node, reason);

    // LOGS_DEFAULT(INFO) << q_node->Name() << " keep " << keep_q;

    if (keep_q) {
      dst_graph.AddNode(*q_node);  // Add the node to the graph
      // if keep_q, then output defs of the target node doesn't change
      output_args.push_back(&dst_graph.GetOrCreateNodeArg(target_node.OutputDefs().at(0)->Name(),
                                                          target_node.OutputDefs().at(0)->TypeAsProto()));
    } else {
      // convert this Q to float
      output_args.push_back(&ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                               node_unit_outputs.at(0), float_type));
    }
  } else {
    for (const auto& node_unit_output : node_unit_outputs) {
      // convert non-qdq outputs to float
      NodeArg& output_arg = ProcessNodeUnitIO(dst_graph, src_graph, initializers_to_dequant,
                                              node_unit_output, float_type);
      output_args.push_back(&output_arg);
    }
  }

  // Add the target node in the node unit to the graph.
  dst_graph.AddNode(target_node.Name(),
                    target_node.OpType(),
                    target_node.Description(),
                    input_args,
                    output_args,
                    &target_node.GetAttributes(),
                    target_node.Domain());
}

// Creates a new model without the DQ/Q operators in the src graph.
static Status CreateModel(const GraphViewer& src_graph,
                          const logging::Logger& logger,
                          int32_t float_type,
                          /*out*/ std::unique_ptr<onnxruntime::Model>& model) {
  // NOTE: This function is a re-implementation of GraphViewerToProto() in core/graph/graph_proto_serializer.cc
  // with the following differences:
  //   - Uses onnxruntime::Graph APIs instead of onnx::GraphProto APIs.
  //   - Traverses the src graph using QDQ node units.
  //   - Filters out DQ/Q ops that wrap full-precision nodes.
  //   - Dequantizes quantized initializers.

  // Constructs model from scratch using the metadata in src_graph
  model = src_graph.CreateModel(logger);

  //
  // Initialize model/graph metadata.
  //

  // TODO: add Model::SetModelVersion() to provider api
  // model->SetModelVersion(ONNX_NAMESPACE::Version::IR_VERSION);

  auto& dst_graph = model->MainGraph();

  // TODO: add Graph::SetName() provider api
  // dst_graph.SetName(src_graph.Name());

  // TODO: add Graph::SetDescription() and GraphViewer::Description() to provider api
  // dst_graph.SetDescription(src_graph.Description());

  // Mark outer scope NodeArgs
  for (const auto& name : src_graph.GetOuterScopeNodeArgNames()) {
    auto* node_arg = src_graph.GetNodeArg(name);
    ORT_RETURN_IF_NOT(node_arg != nullptr, "Outer scope node arg name '" + name + "'was added but does not exist. ");
    dst_graph.AddOuterScopeNodeArg(name);
  }

  //
  // Add nodes (without their DQ/Q ops) to dst graph.
  //

  // Keep track of all the initializers we need to dequantize to float.
  std::unordered_map<std::string, NodeUnitIODef> initializers_to_dequant;

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(&src_graph);

  std::unordered_set<const NodeUnit*> seen_node_units;
  const auto& node_indices = src_graph.GetNodesInTopologicalOrder();

  // Process node units in topological order. Filter out Q/DQ ops.
  for (size_t i = 0; i < node_indices.size(); i++) {
    gsl::not_null<const onnxruntime::Node*> node(src_graph.GetNode(node_indices[i]));

    // Get the node_unit associated with the node.
    gsl::not_null<const NodeUnit*> node_unit = node_unit_map.at(node);

    // Visiting 'nodes' in topological order does not guarantee that 'node_units' are
    // also visited in topological order. Skip this node if it is not the node_unit's target node
    // to ensure actual 'node_units' are visited in topological order.
    if (node != &node_unit->GetNode()) {
      continue;
    }

    if (seen_node_units.count(node_unit) != 0) {
      continue;  // Already handled this node unit
    }

    if (node_unit->UnitType() == NodeUnit::Type::SingleNode) {
      AddStandaloneNodeUnit(dst_graph, src_graph, *node_unit, float_type, initializers_to_dequant, logger);
    } else {
      AddQDQNodeUnit(dst_graph, src_graph, *node_unit, float_type, initializers_to_dequant, logger);
    }

    seen_node_units.insert(node_unit);
  }

  //
  // Copy initializers to dst graph. Dequantize initializers that were originally quantized.
  //

  std::unordered_set<std::string> current_scope_initializer_set;

  auto& initializers = src_graph.GetAllInitializedTensors();

  // Sort initializers to maintain consistency in model proto created across inference requests
  std::vector<std::string> const_inits;
  for (auto& it : initializers) {
    const_inits.push_back(it.first);
  }
  std::sort(const_inits.begin(), const_inits.end());

  for (auto& it : const_inits) {
    if (initializers_to_dequant.count(it)) {
      // TODO: Dequantize initializer, create a new TensorProto, and add it to dst_graph.
      // See reference implementation in core/providers/cpu/quantization/quantize_linear.cc DequantizeLinearApply()
    } else {
      dst_graph.AddInitializedTensor(*(initializers.at(it)));
    }
    current_scope_initializer_set.insert(it);
  }

  // handle outer scope value which is a constant initializer
  for (auto& node_idx : src_graph.GetNodesInTopologicalOrder()) {
    const auto& node = src_graph.GetNode(node_idx);
    for (const auto& input : node->InputDefs()) {
      if (current_scope_initializer_set.find(input->Name()) != current_scope_initializer_set.end()) {
        continue;
      }
      if (src_graph.IsConstantInitializer(input->Name(), true)) {
        if (initializers_to_dequant.count(input->Name())) {
          // TODO: Dequantize initializer, create a new TensorProto, and add it to dst_graph.
          // See reference implementation in core/providers/cpu/quantization/quantize_linear.cc DequantizeLinearApply()
        } else {
          dst_graph.AddInitializedTensor(*(src_graph.GetConstantInitializer(input->Name(), true)));
        }
        current_scope_initializer_set.insert(input->Name());
      }
    }
  }

  // Validate graph, remove unnecessary initializers, and run type/shape inference.
  ORT_RETURN_IF_ERROR(dst_graph.Resolve());
  return Status::OK();
}

std::unique_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::GetModelProtoFromFusedNode(const onnxruntime::Node& fused_node,
                                           const onnxruntime::GraphViewer& subgraph,
                                           const logging::Logger& logger) const {
  int32_t float_type = global_context_.precision_str == "FP32" ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT
                                                               : ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  std::unique_ptr<onnxruntime::Model> model;
  Status status = CreateModel(subgraph, logger, float_type, model);
  auto model_proto = model->ToProto();

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    const std::string& name = fused_node.Name();
    std::fstream dump(name + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }
#else
  ORT_UNUSED_PARAMETER(fused_node);
#endif
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());

  return model_proto;
}

std::vector<std::vector<int64_t>> GetInputTensorShapes(const Ort::KernelContext& context) {
  const auto input_count = context.GetInputCount();
  std::vector<std::vector<int64_t>> input_shapes;
  input_shapes.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto input_tensor = context.GetInput(i);
    auto tensor_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    input_shapes.push_back(std::move(tensor_shape));
  }
  return input_shapes;
}

std::string MakeMapKeyString(const std::vector<std::vector<int64_t>>& shapes,
                             const std::string& device_type) {
  std::string key;
  key += device_type;
  key += "|";  // separator
  for (auto shape : shapes) {
    for (auto dim : shape) {
      std::ostringstream o;
      o << dim;
      key += o.str();
      key += ",";
    }
    key += "|";
  }
  return key;
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteInputShapeInfo(const ONNX_NAMESPACE::ModelProto& model_proto,
                                      const std::vector<std::vector<int64_t>>& input_shapes) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (size_t i = 0, limit = input_shapes.size(); i < limit; i++) {
    auto g_in_shape = graph_proto->mutable_input(static_cast<int>(i))
                          ->mutable_type()
                          ->mutable_tensor_type()
                          ->mutable_shape();
    g_in_shape->clear_dim();
    const auto& shape = input_shapes[i];
    for (size_t dim = 0, end = shape.size(); dim < end; dim++) {
      g_in_shape->add_dim()->set_dim_value(shape[dim]);
    }
  }
  return model_copy;
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto>
BackendManager::ReWriteBatchDimWithOne(const ONNX_NAMESPACE::ModelProto& model_proto) {
  auto model_copy = std::shared_ptr<ONNX_NAMESPACE::ModelProto>(ONNX_NAMESPACE::ModelProto::Create());
  std::string proto_str;
  model_proto.SerializeToString(proto_str);
  model_copy->ParseFromString(proto_str);
  auto graph_proto = model_copy->mutable_graph();

  for (int i = 0; i < graph_proto->input_size(); i++) {
    ONNX_NAMESPACE::TensorShapeProto* g_in_shape =
        graph_proto->mutable_input(static_cast<int>(i))
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape();
    g_in_shape->mutable_dim(0)->clear_dim_value();
    g_in_shape->mutable_dim(0)->set_dim_value(1);
  }
  return model_copy;
}

void BackendManager::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  std::chrono::high_resolution_clock::time_point start_compute, end_compute;
#ifdef OPENVINO_FIL_ENABLED
  static bool fil_enabled = true;
  if (fil_enabled) {
    start_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "Start Compute";
  }
#endif
  // OV NPU doesn't support dynamic shaped model inference.
  // if disable_dynamic_shapes is set to true then execution of dynamic model is done
  // by rewriting the model to static shaped model at runtime based on input shape.
  // disable_dynamic_shapes is always set to true for OV NPU plugin.
  bool use_dynamic_backend = true;
  if (subgraph_context_.has_dynamic_input_shape &&
      !GetGlobalContext().disable_dynamic_shapes &&
      (GetGlobalContext().device_type.find("CPU") != std::string::npos ||
       GetGlobalContext().device_type.find("GPU") != std::string::npos)) {
    concrete_backend_->Infer(context);
    use_dynamic_backend = false;
  } else if (use_dynamic_backend && subgraph_context_.has_dynamic_input_shape) {
    std::vector<std::vector<int64_t>> tensor_shapes = GetInputTensorShapes(ctx);
    auto key = MakeMapKeyString(tensor_shapes, GetGlobalContext().device_type);
    std::shared_ptr<IBackend> dynamic_backend;
    auto search = backend_map_.find(key);
    if (search == backend_map_.end()) {
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Creating dynamic backend for key: " << key;
      LOGS_DEFAULT(INFO) << "[OpenVINO-EP] "
                         << "Backend created for graph " << subgraph_context_.subgraph_name;
      auto modelproto_with_concrete_shapes = ReWriteInputShapeInfo(*model_proto_, tensor_shapes);
      try {
        dynamic_backend = BackendFactory::MakeBackend(*modelproto_with_concrete_shapes,
                                                      GetGlobalContext(),
                                                      subgraph_context_,
                                                      ep_ctx_handle_);
      } catch (const OnnxRuntimeException& ex) {
        if (GetGlobalContext().device_type.find("NPU") != std::string::npos) {
          LOGS_DEFAULT(WARNING) << ex.what();
          LOGS_DEFAULT(WARNING) << "Model compilation failed at OV NPU."
                                << "Falling back to OV CPU for execution";
          GetGlobalContext().device_type = "CPU";
          GetGlobalContext().precision_str = "FP32";
          key = MakeMapKeyString(tensor_shapes, GetGlobalContext().device_type);
          try {
            dynamic_backend = BackendFactory::MakeBackend(*modelproto_with_concrete_shapes,
                                                          GetGlobalContext(),
                                                          subgraph_context_,
                                                          ep_ctx_handle_);
          } catch (std::string const& msg) {
            ORT_THROW(msg);
          }
        }
      }
      backend_map_.insert({key, dynamic_backend});
    } else {
      dynamic_backend = search->second;
    }

    dynamic_backend->Infer(context);
  } else {
    concrete_backend_->Infer(context);
  }
#ifdef OPENVINO_FIL_ENABLED
  if (fil_enabled) {
    end_compute = std::chrono::high_resolution_clock::now();
    LOGS_DEFAULT(INFO) << "End Compute";
    std::chrono::duration<double> compute_time = end_compute - start_compute;
    std::cout << "Compute Time: " << compute_time.count() << " s" << std::endl;
    fil_enabled = false;  // calculating compute time for first run only
  }
#endif
}

void BackendManager::ShutdownBackendManager() {
}

}  // namespace openvino_ep
}  // namespace onnxruntime
