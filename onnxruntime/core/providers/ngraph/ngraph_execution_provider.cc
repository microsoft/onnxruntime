// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "ngraph_execution_provider.h"
#include "ngraph_custom_op.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ngraph/ngraph.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>
#pragma GCC diagnostic pop

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, MIN(destsz, srcsz))

namespace onnxruntime {

constexpr const char* NGRAPH = "nGraph";

NGRAPHExecutionProvider::NGRAPHExecutionProvider(const NGRAPHExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kNGraphExecutionProvider} {
  DeviceAllocatorRegistrationInfo default_allocator_info({OrtMemTypeDefault,
                                                          [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(NGRAPH, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault)); },
                                                          std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(default_allocator_info));

  DeviceAllocatorRegistrationInfo cpu_allocator_info({OrtMemTypeCPUOutput,
                                                      [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(NGRAPH, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeCPUOutput)); },
                                                      std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(cpu_allocator_info));

  ORT_ENFORCE(info.ng_backend_type == "CPU", "nGraph Execution Provider for onnxruntime currently is only supported for CPU backend.");

  try {
    ng_backend_ = ngraph::runtime::Backend::create(info.ng_backend_type);
  } catch (const std::exception& exp) {
    LOGS_DEFAULT(FATAL) << "Exception while creating nGraph " << info.ng_backend_type << " Backend: " << std::string(exp.what());
  } catch (...) {
    LOGS_DEFAULT(FATAL) << "Unknown exception while while creating nGraph " << info.ng_backend_type << " Backend";
    throw;
  }
}

std::shared_ptr<KernelRegistry> NGRAPHExecutionProvider::GetKernelRegistry() const {
  return std::make_shared<KernelRegistry>();
}

/**
 * Checks if a tensor represented by srcLocation can be copied into the dstLocation tensor
 * @param src_location result of Location().name call on the source tensor
 * @param dst_location result of Location().name call on the destination tensor
 * @return true if src and dest locations combination allows copying
 */
bool TensorCopyPossible(const std::string& src_location, const std::string& dst_location) {
  // contains allowed combinations of source and destination locations for tensors copying purposes
  // the first element of a pair denotes a source, the second - destination
  static const std::map<std::string, std::string> allowed_copy_directions = {
      {NGRAPH, CPU}, {NGRAPH, NGRAPH}, {CPU, NGRAPH}};

  // copying of tensors is allowed only if the params match any of the allowed combinations
  return std::any_of(allowed_copy_directions.begin(),
                     allowed_copy_directions.end(), [&](const auto& copy_direction) {
                       return src_location == copy_direction.first && dst_location == copy_direction.second;
                     });
}

Status NGRAPHExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  const size_t src_bytes = src.DataType()->Size() * src.Shape().Size();
  const size_t dst_bytes = dst.DataType()->Size() * dst.Shape().Size();
  if (src_bytes != dst_bytes) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "nGraph: Source and Destination data sizes are not equal - cannot copy tensors");
  }

  if (!TensorCopyPossible(src.Location().name, dst.Location().name)) {
    ORT_NOT_IMPLEMENTED("Copying tensors between '", src.Location().name, "' and '", dst.Location().name,
                        "' is not implemented in NGRAPHExecutionProvider");
  }

  MEMCPY_S(dst.MutableDataRaw(), src.DataRaw(), dst_bytes, src_bytes);

  return Status::OK();
}

// Returns true only if op is in a mode that is not currently supported
static bool IsUnsupportedOpMode(const Node* node, const onnxruntime::GraphViewer& graph_viewer) {
  const auto& optype = node->OpType();
  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  if (optype == "Reshape") {
    //nGraph Reshape op currently requires shape info available in advance.
    const auto& shape_arg = node->InputDefs()[1];
    return initializers.find(shape_arg->Name()) == initializers.end();
  } else if (optype == "MaxPool") {
    //MaxPool "indices" output is not currently supported.
    if (node->OutputDefs().size() > 1) {
      return true;
    }

    // ceil_mode and dilations attrs are not supported in nGraph
    const auto& attributes = node->GetAttributes();
    const auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() && ceil_attr->second.i() != 0) {
      return true;
    }

    if (attributes.find("dilations") != attributes.end()) {
      return true;
    }
  } else if (optype == "OneHot") {
    //nGraph OneHot op currently requires depth info available in advance.
    const auto& depth_arg = node->InputDefs()[1];
    return initializers.find(depth_arg->Name()) == initializers.end();
  } else if (optype == "TopK") {
    //TopK opset 10 is currently not supported.
    //K as input is currently not suppported.
    return node->InputDefs().size() > 1;
  } else if (optype == "LSTM") {
    const auto& attributes = node->GetAttributes();
    const auto direction_attr = attributes.find("direction");
    if (direction_attr != attributes.end()) {
      return direction_attr->second.s() == "reverse";
    }
  } else if (optype == "MatMul") {
    //All matmuls except float have computation missmatch
    const bool A_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
    const bool B_is_float = node->InputDefs()[1]->Type()->find("float") != std::string::npos;
    return (A_is_float && B_is_float) ? false : true;
  } else if (optype == "Pad") {
    //3D pad with negative padding have computation missmatch
    const auto& attributes = node->GetAttributes();
    const auto pad_attr = attributes.find("pads");
    if (pad_attr != attributes.end() && (pad_attr->second.ints().size() > 4 || pad_attr->second.ints().size() == 3)) {
      for (const auto& val : pad_attr->second.ints()) {
        if (val < 0)
          return true;
      }
    }
  } else if (optype == "Cast") {
    //support of casting to bool in nGraph is in progress
    const auto& attributes = node->GetAttributes();
    const auto to_attr = attributes.find("to");
    return to_attr->second.i() == ONNX_NAMESPACE::TensorProto::BOOL;
  } else if (optype == "Slice") {
    //Slice in opset 10 is currently not supported.
    //unsupported inputs: starts, ends, axes, steps
    if (node->InputDefs().size() > 1) {
      return true;
    }
    //nGraph does not properly handle the situation where any value of the "starts" attribute
    //is higher than a corresponding value in the "ends"
    const auto& attributes = node->GetAttributes();
    if (attributes.count("starts") == 0 || attributes.count("ends") == 0) {
      return true;
    }

    const auto& starts = attributes.find("starts")->second.ints();
    const auto& ends = attributes.find("ends")->second.ints();
    for (int i = 0; i < starts.size(); ++i) {
      if (starts.Get(i) > ends.Get(i)) {
        return true;
      }
    }
  } else if (optype == "AveragePool") {
    // ceil_mode attribute is not supported in nGraph
    const auto& attributes = node->GetAttributes();
    const auto ceil_attr = attributes.find("ceil_mode");
    // default value of ceil_mode (0) is supported.
    if (ceil_attr != attributes.end() && ceil_attr->second.i() != 0) {
      return true;
    }
  }

  //Op doesn't fall into known any of unsupported modes.
  return false;
}

static bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
      return true;
    default:
      return false;
  }
}

static bool IsNodeSupported(const std::map<std::string, std::set<std::string>>& op_map,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeIndex node_idx) {
  const auto& node = graph_viewer.GetNode(node_idx);
  const auto& optype = node->OpType();
  const auto& domain = node->Domain();

  /*
  1. Check input and output data types are supported.
  2. Check Op is supported
	 2a. Check if Op is of known unsupported modes (edge cases). If yes return false right away.
	 2b. If above is not true, check if the op is available in nGraph.
  */

  //Check 1
  bool are_types_supported = true;

  node->ForEachDef([&are_types_supported](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
    are_types_supported &= IsTypeSupported(&node_arg);
  });

  if (!are_types_supported) {
    return false;
  }

  //Check 2a
  if (domain == kOnnxDomain && IsUnsupportedOpMode(node, graph_viewer)) {
    return false;
  }

  //Check 2b
  const auto opset = op_map.find(domain);
  if (opset == op_map.end() || opset->second.find(optype) == opset->second.end()) {
    return false;
  } else {
    return true;
  }
}

static void AppendClusterToSubGraph(const std::vector<NodeIndex>& nodes,
                                    const onnxruntime::GraphViewer& graph_viewer,
                                    const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& outputs,
                                    const std::unordered_set<std::string>& ng_required_initializers,
                                    std::vector<std::unique_ptr<ComputeCapability>>& result) {
  static size_t op_counter = 0;

  // Create ng_required_initializers attribute of NGraphCustomOp
  ONNX_NAMESPACE::AttributeProto initializers;
  initializers.set_name("initializers");
  initializers.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS);
  for (const auto& init : ng_required_initializers) {
    auto tensor = initializers.add_tensors();
    *tensor = *(graph_viewer.GetAllInitializedTensors().at(init));
  }

  auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->attributes["initializers"] = initializers;
  meta_def->name = "NGRAPHCustomOp_" + std::to_string(++op_counter);
  meta_def->domain = kNGraphDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  sub_graph->nodes = nodes;
  sub_graph->SetMetaDef(meta_def);
  result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
}

static int GetOnnxOpSet(const GraphViewer& graph_viewer) {
  const auto& dm_to_ver = graph_viewer.DomainToVersionMap();
  return dm_to_ver.at(kOnnxDomain);
}

static std::map<std::string, std::set<std::string>> GetNgSupportedOps(const int onnx_opset) {
  std::map<std::string, std::set<std::string>> ng_supported_ops;
  ng_supported_ops.emplace(kOnnxDomain, ngraph::onnx_import::get_supported_operators(onnx_opset, kOnnxDomain));

  const std::set<std::string> ng_disabled_ops = {};  //Place-holder for ops not supported.

  for (const auto& disabled_op : ng_disabled_ops) {
    ng_supported_ops.at(kOnnxDomain).erase(disabled_op);
  }

  return ng_supported_ops;
}

static std::vector<NodeIndex> GetUnsupportedNodeIndices(const GraphViewer& graph_viewer, /*out*/ std::unordered_set<std::string>& ng_required_initializers) {
  const auto ng_supported_ops = GetNgSupportedOps(GetOnnxOpSet(graph_viewer));

  std::vector<NodeIndex> unsupported_nodes_idx;

  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    if (IsNodeSupported(ng_supported_ops, graph_viewer, node_idx)) {
      // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&ng_required_initializers, &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
              if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
                ng_required_initializers.insert(node_arg.Name());
              } }, true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }

  return unsupported_nodes_idx;
}

/* Returns a vector clusters(or node_idx). For each unsupported node, the graph is split into 3 parts.
   supported_cluster + (UNsupported_node + rest_of_the_graph). This functions returns vector of all supported_clusters by nGraph 
*/
static std::vector<std::vector<NodeIndex>> GetPartitionedClusters(const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> ng_clusters;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx) and append it to return list.
    std::vector<NodeIndex> this_cluster{prev, it};
    if (!this_cluster.empty()) {
      ng_clusters.push_back(std::move(this_cluster));
    }
    // Point prev to node idx past this unsuported node.
    prev = ++it;
  }

  //Tail
  std::vector<NodeIndex> this_cluster{prev, topological_order.end()};
  if (!this_cluster.empty()) {
    ng_clusters.push_back(std::move(this_cluster));
  }

  return ng_clusters;
}

static void GetInputsOutputsOfCluster(const GraphViewer& graph_viewer,
                                      const std::vector<NodeIndex>& cluster,
                                      const std::unordered_set<std::string>& ng_required_initializers,
                                      /*out*/ std::vector<std::string>& cluster_inputs,
                                      /*out*/ std::vector<std::string>& cluster_outputs) {
  std::unordered_set<std::string> input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;

  for (const auto& node_idx : cluster) {
    const auto& node = graph_viewer.GetNode(node_idx);

    // Collect all inputs and outputs
    node->ForEachDef(
        [&input_args, &output_args](const NodeArg& node_arg, bool is_input) {
          is_input ? input_args.insert(node_arg.Name()) : output_args.insert(node_arg.Name());
        },
        true);

    // Check if output of this node is used by nodes outside this_cluster. If yes add this to cluster outputs
    for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
      const auto& ext_node = graph_viewer.GetNode((*it).Index());

      if (std::find(cluster.begin(), cluster.end(), ext_node->Index()) == cluster.end()) {
        // Node is external to this_cluster. Search through its inputs to find the output that is generated by this_cluster.
        std::set<std::string> ext_node_inputs;
        ext_node->ForEachDef(
            [&ext_node_inputs](const onnxruntime::NodeArg& arg, bool is_input) {
              if (is_input) {
                ext_node_inputs.insert(arg.Name());
              }
            },
            true);

        for (const auto& out_def : node->OutputDefs()) {
          if (ext_node_inputs.find(out_def->Name()) != ext_node_inputs.end()) {
            external_output_args.insert(out_def->Name());
          }
        }
      }
    }
  }

  std::vector<std::string> cluster_initializers;

  //Extract initializers used by this_cluster.
  std::unordered_set<std::string> original_graph_inputs;
  for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    original_graph_inputs.insert(node_arg->Name());
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  for (const auto& in_arg : input_args) {
    if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        ng_required_initializers.count(in_arg)) {
      cluster_initializers.push_back(in_arg);
    } else if (!output_args.count(in_arg)) {
      cluster_inputs.push_back(in_arg);
    }
  }

  std::copy(external_output_args.begin(), external_output_args.end(), std::back_inserter(cluster_outputs));
  for (const auto& node_arg : graph_viewer.GetOutputs()) {
    const auto& name = node_arg->Name();
    if (output_args.count(name) && !external_output_args.count(name)) {
      cluster_outputs.push_back(name);
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
NGRAPHExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const std::vector<const KernelRegistry*>& kernel_registries) const {
  ORT_UNUSED_PARAMETER(kernel_registries);

  std::vector<std::unique_ptr<ComputeCapability>> result;

  //TODO:(nivas) Handle If and Loop operators
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() && tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "nGraph EP: Initializers with external data location are not currently supported";
      return result;
    }
  }

  /* This is a list of initializers that nGraph considers as constants. Example weights, reshape shape etc.
     TODO: Support overridable initializers */
  std::unordered_set<std::string> ng_required_initializers;

  const auto unsupported_nodes = GetUnsupportedNodeIndices(graph_viewer, ng_required_initializers);

  //If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    //Fill inputs with names
    std::for_each(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                  [&inputs](const NodeArg* node_arg) { inputs.push_back(node_arg->Name()); });

    /* In scenarios, when there are no inputs or all inputs being initializers, 
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
    if (inputs.empty()) {
      return result;
    }

    //Fill outputs with names
    std::for_each(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(),
                  [&outputs](const NodeArg* node_arg) { outputs.push_back(node_arg->Name()); });

    // Remove initializers from inputs if they are in ng_required_initializers
    inputs.erase(std::remove_if(inputs.begin(), inputs.end(), [&ng_required_initializers](const std::string& name) -> bool {
                   return ng_required_initializers.count(name);
                 }),
                 inputs.end());

    // Create and add this graph to result.
    AppendClusterToSubGraph(graph_viewer.GetNodesInTopologicalOrder(), graph_viewer, inputs, outputs, ng_required_initializers, result);

  } else {  // unsupported_nodes_idx.empty()
    const auto ng_clusters = GetPartitionedClusters(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

    for (const auto& this_cluster : ng_clusters) {
      std::vector<std::string> cluster_inputs, cluster_outputs;
      GetInputsOutputsOfCluster(graph_viewer, this_cluster, ng_required_initializers, cluster_inputs, cluster_outputs);

      if (!cluster_inputs.empty()) {
        AppendClusterToSubGraph(this_cluster, graph_viewer, cluster_inputs, cluster_outputs, ng_required_initializers, result);
      }
    }
  }

  return result;
}

static ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::Node* fused_node) {
  const auto& attributes = fused_node->GetAttributes();
  const auto& initializers = attributes.at("initializers").tensors();

  ONNX_NAMESPACE::ModelProto model_proto;
  auto graph_proto = model_proto.mutable_graph();
  const auto& fused_graph = fused_node->GetFunctionBody()->Body();

  for (const auto& node : fused_graph.Nodes()) {
    node.ToProto(*(graph_proto->add_node()));
  }

  for (const auto& input : fused_node->InputDefs()) {
    auto valueInfoProto = graph_proto->add_input();
    *valueInfoProto = input->ToProto();
  }

  for (const auto& output : fused_node->OutputDefs()) {
    auto valueInfoProto = graph_proto->add_output();
    *valueInfoProto = output->ToProto();
  }

  for (const auto& initializer : initializers) {
    graph_proto->add_initializer()->CopyFrom(initializer);
  }

  auto opset = model_proto.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(fused_graph.DomainToVersionMap().at(kOnnxDomain));
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  return model_proto;
}

Status NGRAPHExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node : fused_nodes) {
    auto model_proto = GetModelProtoFromFusedNode(fused_node);

    NodeComputeInfo compute_info;

    // Local copy of backend since, class members cannot be captured.
    auto ngraph_backend = ng_backend_;
    compute_info.create_state_func = [model_proto, ngraph_backend](ComputeContext* context, FunctionState* state) {
      auto* p = new onnxruntime::ngraph_ep::NGRAPHCustomOp(context, model_proto, ngraph_backend);
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete reinterpret_cast<onnxruntime::ngraph_ep::NGRAPHCustomOp*>(state);
    };

    compute_info.compute_func = [](FunctionState state, ONNXRunTimeTensor* input_tensors, size_t num_inputs, ONNXRunTimeTensor* output_tensors, size_t num_outputs) {
      onnxruntime::ngraph_ep::NGRAPHCustomOp* ng_custom_op = reinterpret_cast<onnxruntime::ngraph_ep::NGRAPHCustomOp*>(state);

      const Status compute_status = ng_custom_op->Compute(input_tensors, num_inputs, output_tensors, num_outputs);

      return compute_status == Status::OK() ? 0 : 1;
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

}  // namespace onnxruntime
