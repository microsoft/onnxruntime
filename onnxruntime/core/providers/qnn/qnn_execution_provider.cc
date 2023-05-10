// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_execution_provider.h"

#include "core/providers/common.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/partitioning_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/partitioning_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {

constexpr const char* QNN = "QNN";

QNNExecutionProvider::QNNExecutionProvider(const ProviderOptions& provider_options_map)
    : IExecutionProvider{onnxruntime::kQnnExecutionProvider, true}, runtime_options_(provider_options_map) {
  static const std::string BACKEND_PATH = "backend_path";
  auto backend_path_pos = runtime_options_.find(BACKEND_PATH);

  if (backend_path_pos != runtime_options_.end()) {
    backend_path_ = backend_path_pos->second;
    LOGS_DEFAULT(INFO) << "Backend path: " << backend_path_;
  } else {
    LOGS_DEFAULT(ERROR) << "No backend path provided.";
  }

  profiling_level_ = qnn::ProfilingLevel::OFF;
  static const std::string PROFILING_LEVEL = "profiling_level";
  auto profiling_level_pos = runtime_options_.find(PROFILING_LEVEL);
  if (profiling_level_pos != runtime_options_.end()) {
    ParseProfilingLevel(profiling_level_pos->second);
    LOGS_DEFAULT(INFO) << "profiling_level: " << static_cast<uint8_t>(profiling_level_);
  }

  rpc_control_latency_ = 10;
  static const std::string RPC_CONTROL_LANTENCY = "rpc_control_latency";
  auto latency_pos = runtime_options_.find(RPC_CONTROL_LANTENCY);
  if (latency_pos != runtime_options_.end()) {
    rpc_control_latency_ = static_cast<uint32_t>(std::stoul(latency_pos->second));
    LOGS_DEFAULT(INFO) << "rpc_control_latency: " << rpc_control_latency_;
  }

  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(QNN, OrtAllocatorType::OrtDeviceAllocator));
      });

  cpu_allocator_ = CreateAllocator(device_info);
  InsertAllocator(cpu_allocator_);

  qnn_backend_manager_ = std::make_unique<qnn::QnnBackendManager>(backend_path_,
                                                                  profiling_level_,
                                                                  rpc_control_latency_);
}

bool QNNExecutionProvider::IsNodeSupported(qnn::QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                           std::unordered_map<const NodeUnit*, bool>& node_unit_supported_result,
                                           const logging::Logger& logger) const {
  bool is_npu_backend = qnn_backend_manager_->IsNpuBackend();
  // If we have visited one of the nodes in the node_unit, use the result directly
  const auto it = node_unit_supported_result.find(&node_unit);
  if (it != node_unit_supported_result.cend()) {
    return it->second;
  } else {
    // quantized required, filter out the non-quantized nodes, filter in the QDQ nodes
    auto IsQdqNode = [](const NodeUnit& node_unit) {
      if ("QuantizeLinear" == node_unit.OpType() || "DequantizeLinear" == node_unit.OpType()) {
        return true;
      } else {
        return false;
      }
    };

    // Is NPU backend, is single node, case by case
    // Q/DQ nodes -- supported
    // Transpose nodes -- supported
    // Cast nodes -- need to call CastOpBuilder::IsOpSupported
    if (is_npu_backend && NodeUnit::Type::SingleNode == node_unit.UnitType()) {
      if (IsQdqNode(node_unit)) {  // Qnn has Quantize & Dequantize Op
        LOGS(logger, VERBOSE) << "Single Q/DQ node is supported for NPU backend. Node name: " << node_unit.Name();
        return true;
      }

      // Tranpose only changes the data layout. NPU still supports it.
      if ("Transpose" == node_unit.OpType()) {
        LOGS(logger, VERBOSE) << "Single Transpose node is supported for NPU backend. Node name: " << node_unit.Name();
        return true;
      }

      // For Cast, need to call IsOpSupported (below) to validate input and output types.
      // For other single non-qdq nodes, immediately return not supported.
      if (node_unit.OpType() != "Cast") {
        LOGS(logger, VERBOSE) << "Non-QDQ single node is not supported for NPU backend. Node name: " << node_unit.Name()
                              << " Op type: " << node_unit.OpType();
        return false;
      }
    }

    // Non-NPU backend, quantized model not supported, but a QDQ node encountered
    if (!is_npu_backend && IsQdqNode(node_unit)) {
      LOGS(logger, ERROR) << "There's no reason to run a QDQ model on non HTP/DSP backend!";
      return false;
    }

    bool supported = false;
    const auto* op_builder = qnn::GetOpBuilder(node_unit.OpType());
    if (op_builder == nullptr) {
      LOGS(logger, VERBOSE) << "Op not implemented in QNN EP. Op type: " << node_unit.OpType();
    } else {
      auto status = op_builder->IsOpSupported(qnn_model_wrapper,
                                              node_unit, logger,
                                              is_npu_backend);
      if (Status::OK() != status) {
        LOGS(logger, VERBOSE) << "Op type: " << node_unit.OpType()
                              << ", not supported: " << status.ErrorMessage();
      }
      supported = (Status::OK() == status);
    }
    node_unit_supported_result[&node_unit] = supported;
    return supported;
  }
}

std::unordered_set<const Node*>
QNNExecutionProvider::GetSupportedNodes(const GraphViewer& graph_viewer,
                                        const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                        const size_t node_unit_size,
                                        const logging::Logger& logger) const {
  std::unordered_set<const Node*> supported_nodes{};
  // This holds the result of whether a NodeUnit is supported or not,
  // to prevent nodes in a NodeUnit to be checked for multiple times
  std::unordered_map<const NodeUnit*, bool> node_unit_supported_result;
  node_unit_supported_result.reserve(node_unit_size);

  std::unordered_set<std::string> initializer_input_lookup;
  auto graph_initializers = graph_viewer.GetAllInitializedTensors();
  for (auto graph_ini : graph_initializers) {
    initializer_input_lookup.emplace(graph_ini.first);
  }

  std::unordered_map<std::string, size_t> model_input_index_map;
  std::unordered_map<std::string, size_t> model_output_index_map;
  std::unordered_map<std::string, qnn::OnnxTensorInfo> inputs_info;
  std::unordered_map<std::string, qnn::OnnxTensorInfo> outputs_info;
  auto qnn_model_wrapper = qnn::QnnModelWrapper(graph_viewer, logger,
                                                qnn_backend_manager_->GetQnnInterface(),
                                                qnn_backend_manager_->GetQnnBackendHandle(),
                                                model_input_index_map,
                                                model_output_index_map,
                                                initializer_input_lookup, cpu_allocator_);

  for (const auto& node : graph_viewer.Nodes()) {
    const NodeUnit* node_unit = node_unit_map.at(&node);
    const bool supported = IsNodeSupported(qnn_model_wrapper,
                                           *node_unit,
                                           node_unit_supported_result,
                                           logger);
    LOGS(logger, VERBOSE) << "Node supported: [" << supported
                          << "] index: [" << node.Index()
                          << "] name: [" << node.Name()
                          << "] Operator type: [" << node.OpType()
                          << "] as part of the NodeUnit type: [" << node_unit->OpType()
                          << "] index: [" << node_unit->Index()
                          << "] name: [" << node_unit->Name()
                          << "]";
    if (supported) {
      supported_nodes.insert(&node);
    }
  }

  return supported_nodes;
}

std::vector<std::unique_ptr<ComputeCapability>>
QNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                    const IKernelLookup& /*kernel_lookup*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  const auto& logger = *GetLogger();

  auto rt = qnn_backend_manager_->SetupBackend(logger);
  if (Status::OK() != rt) {
    LOGS(logger, ERROR) << "QNN SetupBackend failed " << rt.ErrorMessage();
    return result;
  }

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph_viewer);

  const auto supported_nodes = GetSupportedNodes(graph_viewer, node_unit_map, node_unit_holder.size(), logger);

  if (supported_nodes.empty()) {
    LOGS(logger, INFO) << "Number of partitions supported by QNN EP: 0";
    return result;
  } else if (supported_nodes.size() == 1) {
    const auto* node = *supported_nodes.begin();
    if (node->OpType() == "QuantizeLinear" || node->OpType() == "DequantizeLinear") {
      LOGS(logger, INFO) << "It doesn't make sense just run a Q/DQ node on HTP.";
      LOGS(logger, INFO) << "Number of partitions supported by QNN EP: 0";
      return result;
    }
  }

  const auto gen_metadef_name = [&]() {
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    return MakeString(QNN, "_", model_hash, "_", metadef_id);
  };

  result = utils::CreateSupportedPartitions(graph_viewer, supported_nodes, {},
                                            gen_metadef_name, QNN, kQnnExecutionProvider, true);

  const auto num_of_partitions = result.size();
  const auto num_of_supported_nodes = std::transform_reduce(
      result.begin(), result.end(),
      size_t{0}, std::plus<>{},
      [](const auto& partition) -> size_t {
        return partition && partition->sub_graph ? partition->sub_graph->nodes.size() : 0;
      });

  const auto summary_msg = MakeString("Number of partitions supported by QNN EP: ", num_of_partitions,
                                      ", number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
                                      ", number of nodes supported by QNN: ", num_of_supported_nodes);

  // If the graph is partitioned in multiple subgraphs, and this may impact performance,
  // we want to give users a summary message at warning level.
  if (num_of_partitions > 1) {
    LOGS(logger, WARNING) << summary_msg;
  } else {
    LOGS(logger, INFO) << summary_msg;
  }

  return result;
}

DataLayout QNNExecutionProvider::GetPreferredLayout() const {
  return DataLayout::NHWC;
}

Status QNNExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                     std::vector<NodeComputeInfo>& node_compute_funcs) {
  const auto& logger = *GetLogger();
  bool is_npu_backend = qnn_backend_manager_->IsNpuBackend();

  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    std::unique_ptr<qnn::QnnModel> qnn_model = std::make_unique<qnn::QnnModel>(logger,
                                                                               qnn_backend_manager_.get(),
                                                                               cpu_allocator_,
                                                                               is_npu_backend);

    ORT_RETURN_IF_ERROR(qnn_model->ComposeGraph(graph_viewer, fused_node));
    ORT_RETURN_IF_ERROR(qnn_model->FinalizeGraphs());
    ORT_RETURN_IF_ERROR(qnn_model->SetupQnnInputOutput());

    LOGS(logger, VERBOSE) << "fused node name: " << fused_node.Name();
    qnn_models_.emplace(fused_node.Name(), std::move(qnn_model));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      LOGS(logger, VERBOSE) << "compute_info.create_state_func context->node_name: " << context->node_name;
      *state = qnn_models_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      // the 'state' is a qnn::QnnModel managed by unique_ptr
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtApi*, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);
      qnn::QnnModel* model = reinterpret_cast<qnn::QnnModel*>(state);
      Status result = model->ExecuteGraph(ctx);
      return result;
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime
