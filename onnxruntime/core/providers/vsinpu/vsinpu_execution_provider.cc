/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include <unordered_map>
#include <string>
#include <unordered_set>
#include "core/framework/compute_capability.h"
#include "core/providers/vsinpu/vsinpu_execution_provider.h"
#include "core/providers/vsinpu/vsinpu_ep_graph.h"
#include "core/providers/vsinpu/builders/op_builder.h"
#include "core/providers/vsinpu/builders/op_builder_factory.h"
#include "core/providers/vsinpu/vsinpu_util.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/node_unit.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/partitioning_utils.h"

namespace onnxruntime {
VSINPUExecutionProvider::VSINPUExecutionProvider(const VSINPUExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kVSINPUExecutionProvider},
      device_id_(info.device_id) {
  AllocatorCreationInfo default_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo("VSINPU", OrtAllocatorType::OrtDeviceAllocator));
      }};

  CreateAllocator(default_memory_info);

  AllocatorCreationInfo cpu_memory_info{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo("VSINPU", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }};

  CreateAllocator(cpu_memory_info);
}

VSINPUExecutionProvider::~VSINPUExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
VSINPUExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/) const {
  const auto& logger = *GetLogger();
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location()) {
      LOGS_DEFAULT(VERBOSE) << "location:" << tensor.second->data_location();
      if (tensor.second->data_location() ==
          ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
        LOGS_DEFAULT(WARNING) << "VSINPU: Initializers with external data location are not "
                                 "currently supported";
        return result;
      }
    }
  }
  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer, logger);

  // This holds the result of whether a NodeUnit is supported or not,
  // to prevent nodes in a NodeUnit to be checked for multiple times
  std::unordered_map<const NodeUnit*, bool> node_unit_supported_result;
  node_unit_supported_result.reserve(node_unit_holder.size());
  std::unordered_set<std::string> node_outputs_in_current_group{};

  const auto is_node_supported = [&](const Node& node) -> bool {
    const NodeUnit* node_unit = node_unit_map.at(&node);
    bool supported = false;

    // If we have visited one of the nodes in the node_unit, use the result directly
    const auto it = node_unit_supported_result.find(node_unit);
    if (it != node_unit_supported_result.cend()) {
      supported = it->second;
    } else {
      // We only check the target node of the node unit
      supported = vsi::npu::GraphEP::IsNodeSupportedInGroup(*node_unit, graph_viewer);
      node_unit_supported_result[node_unit] = supported;
    }

    LOGS_DEFAULT(VERBOSE) << "Node supported: [" << supported
                          << "] Operator type: [" << node.OpType()
                          << "] index: [" << node.Index()
                          << "] name: [" << node.Name()
                          << "] as part of the NodeUnit type: [" << node_unit->OpType()
                          << "] index: [" << node_unit->Index()
                          << "] name: [" << node_unit->Name()
                          << "]";

    if (supported) {
      // We want to save all the output names of nodes in the current group for easy query
      for (const auto* output : node.OutputDefs()) {
        node_outputs_in_current_group.insert(output->Name());
      }
    }
    return supported;
  };

  const auto on_group_closed = [&](const std::vector<const Node*>& group) -> bool {
    // reset per-partition node group tracking
    node_outputs_in_current_group.clear();
    return true;
  };

  const auto gen_metadef_name = [&]() {
    static size_t group_counter = 0;
    return "VSINPU_" + std::to_string(++group_counter);
  };
  result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported, on_group_closed,
                                            gen_metadef_name, "VSINPU", kVSINPUExecutionProvider, &node_unit_map);
  std::for_each(result.begin(), result.end(), [&graph_viewer](auto& capability) {
    if (capability && capability->sub_graph && capability->sub_graph->GetMetaDef()) {
      const auto* meta_def = capability->sub_graph->GetMetaDef();
      bool has_any_non_constant_inputs = std::any_of(meta_def->inputs.begin(),
                                                     meta_def->inputs.end(), [&graph_viewer](const auto& input) {
                                                       return !graph_viewer.IsConstantInitializer(input, true);
                                                     });

      // ALL inputs are constant
      if (!has_any_non_constant_inputs) {
        capability.reset();
      }
    }
  });

  const auto num_of_partitions = result.size();
  const auto num_of_supported_nodes = std::accumulate(
      result.begin(), result.end(), size_t{0},
      [](const auto& acc, const auto& partition) -> size_t {
        return acc + (partition && partition->sub_graph ? partition->sub_graph->nodes.size() : 0);
      });

  const auto summary_msg = MakeString(
      "VSINPUExecutionProvider::GetCapability,",
      " number of partitions supported by VSINPU: ", num_of_partitions,
      "; number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
      "; number of nodes supported by VSINPU: ", num_of_supported_nodes);

  // If the graph is partitioned in multiple subgraphs, and this may impact performance,
  // we want to give users a summary message at warning level.
  if (num_of_partitions > 1) {
    LOGS_DEFAULT(WARNING) << summary_msg;
  } else {
    LOGS_DEFAULT(INFO) << summary_msg;
  }

  return result;
}

Status ComputeStateFunc(vsi::npu::GraphEP* graph_ep,
                        OrtKernelContext* context,
                        const logging::Logger& logger) {
  Ort::KernelContext ctx(context);
  size_t num_in = ctx.GetInputCount();
  const size_t num_inputs = graph_ep->GetGraphInputs().size();

  for (size_t i = 0, j = 0; i < num_inputs; i++) {
    if (!graph_ep->GetGraphInputs()[i]->is_initializer) {
      const auto onnx_input_tensor = ctx.GetInput(i);
      const auto tensor_info = onnx_input_tensor.GetTensorTypeAndShapeInfo();

      auto origin_tensor = graph_ep->GetGraphInputs()[i]->tensor;
      origin_tensor->CopyDataToTensor(onnx_input_tensor.GetTensorRawData(),
                                      vsi::npu::util::GetTensorBytes(tensor_info));
      j++;
    }
  }

  if (!graph_ep->GetGraph()->Run()) {
    LOGS(logger, ERROR) << "Failed to run graph.";
  }
  for (size_t i = 0; i < ctx.GetOutputCount(); i++) {
    auto timvx_tensor = graph_ep->GetGraphOutputs()[i]->tensor;
    auto out_shape = graph_ep->GetGraphOutputs()[i]->shape.GetDims();
    auto onnx_output_tensor =
        ctx.GetOutput(i, out_shape.data(), out_shape.size());
    timvx_tensor->CopyDataFromTensor(const_cast<void*>(onnx_output_tensor.GetTensorRawData()));
  }

  return Status::OK();
}

Status VSINPUExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  const auto& logger = *GetLogger();

  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
    std::shared_ptr<vsi::npu::GraphEP> graph_ep = std::make_shared<vsi::npu::GraphEP>(graph_viewer, logger);

    for (auto tensor : graph_viewer.GetInputsIncludingInitializers()) {
      LOGS(logger, VERBOSE) << "subgraph input init:" << vsi::npu::util::PrintNode(*tensor) << "#"
                            << graph_viewer.IsInitializedTensor(tensor->Name());
      auto input = std::make_shared<vsi::npu::GraphIOInfo>();
      input->name = tensor->Name();
      input->is_initializer = graph_viewer.IsConstantInitializer(tensor->Name(), true);
      graph_ep->GetGraphInputs().push_back(input);
    }
    for (auto tensor : graph_viewer.GetOutputs()) {
      LOGS(logger, VERBOSE) << "subgraph output:" << vsi::npu::util::PrintNode(*tensor);
      auto output = std::make_shared<vsi::npu::GraphIOInfo>();
      output->name = tensor->Name();
      output->is_initializer = false;
      graph_ep->GetGraphOutputs().push_back(output);
    }

    auto node_indices = graph_viewer.GetNodesInTopologicalOrder();
    for (const auto& node_index : node_indices) {
      const auto node = graph_viewer.GetNode(node_index);
      const NodeUnit& node_unit = graph_ep->GetNodeUnit(node);

      // Only add op when we hit the target node
      if (node != &node_unit.GetNode()) {
        continue;
      }
      LOGS(logger, VERBOSE) << "Adding node: [" << node->OpType() << "]";
      vsi::npu::SupportedBuiltinOps().at(node->OpType())->BuildOp(graph_ep.get(), graph_viewer, node_unit);
    }

    LOGS(logger, INFO) << "Verifying graph";
    graph_ep->GetCompiled() = graph_ep->GetGraph()->Compile();
    if (!graph_ep->GetCompiled()) {
      LOGS(logger, ERROR) << "Failed to verify graph.";
    } else {
      LOGS(logger, INFO) << "Graph has been verified successfully.";
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [graph_ep](ComputeContext* /*context*/,
                                                FunctionState* state) {
      *state = graph_ep.get();
      return 0;
    };

    compute_info.compute_func =
        [graph_ep, this](FunctionState /*state*/, const OrtApi* /* api */,
                         OrtKernelContext* context) {
          std::lock_guard<std::mutex> lock(this->GetMutex());
          Status res = ComputeStateFunc(graph_ep.get(), context, *GetLogger());
          return res;
        };

    compute_info.release_state_func = [](FunctionState /*state*/) {};

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> VSINPUExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  return kernel_registry;
}

}  // namespace onnxruntime
