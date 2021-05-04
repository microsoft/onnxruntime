// Copyright (c) Xilinx Inc. All rights reserved.
// Licensed under the MIT License.

#include <istream>
#include <fstream>

#include <pyxir/pyxir.hpp>
#include <pyxir/frontend/onnx.hpp>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "vitisai_execution_provider.h"
#include "vitisai_custom_op.h"

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

constexpr const char* VITISAI = "VITISAI";

typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;
typedef std::shared_ptr<pyxir::graph::XLayer> XLayerHolder;

VitisAIExecutionProvider::VitisAIExecutionProvider(const VitisAIExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kVitisAIExecutionProvider}, backend_type_(info.backend_type), device_id_(info.device_id) {
  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(VITISAI, OrtAllocatorType::OrtDeviceAllocator));
      }};

  InsertAllocator(CreateAllocator(default_memory_info));
}

/**
 * Returns a vector of clusters (or node_idx) that are supported by the given
 * backend type
 */
static std::vector<std::vector<NodeIndex>>
GetSupportedNodeClusters(const XGraphHolder& xg, const std::string& backend_type,
                         const GraphViewer& graph_viewer,
                         /*out*/ std::unordered_set<std::string>& required_initializers) {
  std::vector<std::vector<NodeIndex>> clusters;

  // Retrieve supported tensor names and corresponding subgraphs they belong to
  int cur_idx = 0;
  std::unordered_map<std::string, std::string> supported_tensors;
  std::unordered_map<std::string, int> cluster_idx;
  for (auto& xl_name : xg->get_layer_names()) {
    XLayerHolder xl = xg->get(xl_name);
    if (xl->target == backend_type) {
      supported_tensors[xl->get_attr("onnx_id").get_string()] = xl->subgraph;
      if (cluster_idx.find(xl->subgraph) == cluster_idx.end()) {
        cluster_idx[xl->subgraph] = cur_idx;
        std::vector<NodeIndex> new_cluster;
        clusters.push_back(new_cluster);
      }
    }
  }

  for (const auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    ConstPointerContainer<std::vector<NodeArg*>> node_args = graph_viewer.GetNode(node_idx)->OutputDefs();

    int cluster_id = -1;
    bool is_node_supported = false;
    for (ConstPointerContainer<std::vector<NodeArg*>>::ConstIterator it =
             node_args.begin();
         it != node_args.end(); ++it) {
      if (supported_tensors.find((*it)->Name()) != supported_tensors.end()) {
        is_node_supported = true;
        int found_cluster_id = cluster_idx[supported_tensors[(*it)->Name()]];
        if (cluster_id != -1 && found_cluster_id != cluster_id) {
          //Output tensors belong to different clusters
          LOGS_DEFAULT(FATAL) << "VITIS-AI EP: Found node which belongs to "
                              << "multiple clusters. This is an invalid case";
        }
        cluster_id = found_cluster_id;
      } else if (is_node_supported) {
        // Some output tensors are supported but not others,
        //  should not happen
        LOGS_DEFAULT(FATAL) << "VITIS-AI EP: Found node output tensor "
                            << (*it)->Name() << " which is partially supported by "
                            << " DPU accelerator. This is an invalid case";
      }
    }

    if (is_node_supported) {
      // Collect inputs that are initializers
      graph_viewer.GetNode(node_idx)->ForEachDef([&required_initializers, &graph_viewer](const onnxruntime::NodeArg& node_arg, bool is_input) {
        if(is_input && graph_viewer.GetAllInitializedTensors().count(node_arg.Name())) {
          required_initializers.insert(node_arg.Name());
        } }, true);
      clusters[cluster_id].push_back(node_idx);
    }
  }

  return clusters;
}

static void GetInputsOutputsOfCluster(const GraphViewer& graph_viewer,
                                      const std::vector<NodeIndex>& cluster,
                                      const std::unordered_set<std::string>& ng_required_initializers,
                                      /*out*/ std::vector<std::string>& cluster_inputs,
                                      /*out*/ std::vector<std::string>& cluster_outputs) {
  std::unordered_set<std::string> input_args;
  std::vector<std::string> ordered_input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;

  for (const auto& node_idx : cluster) {
    const auto& node = graph_viewer.GetNode(node_idx);

    // Collect all inputs and outputs
    node->ForEachDef(
        [&input_args, &ordered_input_args, &output_args](const NodeArg& node_arg, bool is_input) {
          if (is_input) {
            if (!input_args.count(node_arg.Name())) {
              ordered_input_args.push_back(node_arg.Name());
            }
            input_args.insert(node_arg.Name());
          } else {
            output_args.insert(node_arg.Name());
          }
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

  //Extract initializers used by this_cluster.
  std::unordered_set<std::string> original_graph_inputs;
  for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    original_graph_inputs.insert(node_arg->Name());
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  std::vector<std::string> const_inputs;
  for (const auto& in_arg : ordered_input_args) {
    if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        ng_required_initializers.count(in_arg)) {
      const_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : ordered_input_args) {
    if (!output_args.count(in_arg) &&
        !((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
          ng_required_initializers.count(in_arg))) {
      cluster_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : const_inputs) {
    cluster_inputs.push_back(in_arg);
  }

  std::copy(external_output_args.begin(), external_output_args.end(), std::back_inserter(cluster_outputs));
  for (const auto& node_arg : graph_viewer.GetOutputs()) {
    const auto& name = node_arg->Name();
    if (output_args.count(name) && !external_output_args.count(name)) {
      cluster_outputs.push_back(name);
    }
  }
}

static void AppendClusterToSubGraph(const std::vector<NodeIndex>& nodes,
                                    const std::vector<std::string>& inputs,
                                    const std::vector<std::string>& outputs,
                                    std::vector<std::unique_ptr<ComputeCapability>>& result) {
  static size_t op_counter = 0;

  auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
  meta_def->name = "VitisAICustomOp_" + std::to_string(++op_counter);
  meta_def->domain = kVitisAIDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;

  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  sub_graph->nodes = nodes;
  sub_graph->SetMetaDef(std::move(meta_def));
  result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
}

std::vector<std::unique_ptr<ComputeCapability>>
VitisAIExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                        const std::vector<const KernelRegistry*>& kernel_registries) const {
  ORT_UNUSED_PARAMETER(kernel_registries);

  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Dump model Proto to file to pass it to pyxir
  auto logger = *GetLogger();

  const Graph& node_graph = graph.GetGraph();
  const std::string& name_ = node_graph.Name();
  onnxruntime::Model model{name_, true, ModelMetaData{}, PathString{},
                           IOnnxRuntimeOpSchemaRegistryList{},
                           node_graph.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(),
                           logger};

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  *(model_proto.mutable_graph()) = node_graph.ToGraphProto();

  std::istringstream model_stream{model_proto.SerializeAsString()};

  // Transform ONNX into Pyxir XGraph data structure
  XGraphHolder xg = pyxir::onnx::import_onnx_model(model_stream);

  // Annotate the subgraph layers in the XGraph that can be executed on the
  //  specified `backend_type_` target
  pyxir::partition(xg, std::vector<std::string>{backend_type_}, "");

  // Next stuff
  if (graph.IsSubgraph()) {
    return result;
  }

  // Need access to model_path_
  for (const auto& tensor : graph.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() && tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "VITIS-AI EP: Initializers with external data location are not currently supported";
      return result;
    }
  }

  std::unordered_set<std::string> required_initializers;
  const auto clusters = GetSupportedNodeClusters(xg, backend_type_, graph, required_initializers);

  for (const auto& this_cluster : clusters) {
    std::vector<std::string> cluster_inputs, cluster_outputs;
    GetInputsOutputsOfCluster(graph, this_cluster, required_initializers, cluster_inputs, cluster_outputs);

    if (!cluster_inputs.empty()) {
      AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
    }
  }

  return result;
}

common::Status VitisAIExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node : fused_nodes) {
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [this, fused_node, logger = GetLogger()](ComputeContext* context, FunctionState* state) {
      auto* p = new vitisai_ep::VitisAICustomOp(context, fused_node, backend_type_, logger);
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete reinterpret_cast<onnxruntime::vitisai_ep::VitisAICustomOp*>(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      onnxruntime::vitisai_ep::VitisAICustomOp* custom_op = reinterpret_cast<onnxruntime::vitisai_ep::VitisAICustomOp*>(state);
      return custom_op->Compute(api, context);
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime
