// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_execution_provider.h"

#include <filesystem>
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/kernel_registry.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "core/providers/partitioning_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/partitioning_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/framework/run_options.h"

namespace onnxruntime {

constexpr const char* QNN = "QNN";

static std::unique_ptr<std::vector<std::function<void()>>> s_run_on_unload_;

void RunOnUnload(std::function<void()> function) {
  OrtMutex mutex;
  std::lock_guard<OrtMutex> guard(mutex);
  if (!s_run_on_unload_) {
    s_run_on_unload_ = std::make_unique<std::vector<std::function<void()>>>();
  }
  s_run_on_unload_->push_back(std::move(function));
}

struct OnUnload {
  ~OnUnload() {
    if (!s_run_on_unload_)
      return;

    for (auto& function : *s_run_on_unload_)
      function();

    s_run_on_unload_.reset();
  }

} g_on_unload;

static void ParseProfilingLevel(std::string profiling_level_string,
                                qnn::ProfilingLevel& profiling_level) {
  std::transform(profiling_level_string.begin(),
                 profiling_level_string.end(),
                 profiling_level_string.begin(),
                 [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
  LOGS_DEFAULT(INFO) << "profiling_level: " << profiling_level_string;
  if (profiling_level_string == "off") {
    profiling_level = qnn::ProfilingLevel::OFF;
  } else if (profiling_level_string == "basic") {
    profiling_level = qnn::ProfilingLevel::BASIC;
  } else if (profiling_level_string == "detailed") {
    profiling_level = qnn::ProfilingLevel::DETAILED;
  } else {
    LOGS_DEFAULT(WARNING) << "Profiling level not valid.";
  }
}

static void ParseHtpPerformanceMode(std::string htp_performance_mode_string,
                                    qnn::HtpPerformanceMode& htp_performance_mode) {
  std::transform(htp_performance_mode_string.begin(),
                 htp_performance_mode_string.end(),
                 htp_performance_mode_string.begin(),
                 [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
  LOGS_DEFAULT(VERBOSE) << "Htp performance mode: " << htp_performance_mode_string;
  if (htp_performance_mode_string == "burst") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpBurst;
  } else if (htp_performance_mode_string == "balanced") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpBalanced;
  } else if (htp_performance_mode_string == "default") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  } else if (htp_performance_mode_string == "high_performance") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpHighPerformance;
  } else if (htp_performance_mode_string == "high_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpHighPowerSaver;
  } else if (htp_performance_mode_string == "low_balanced") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpLowBalanced;
  } else if (htp_performance_mode_string == "low_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpLowPowerSaver;
  } else if (htp_performance_mode_string == "power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpPowerSaver;
  } else if (htp_performance_mode_string == "extreme_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpExtremePowerSaver;
  } else if (htp_performance_mode_string == "sustained_high_performance") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpSustainedHighPerformance;
  } else {
    LOGS_DEFAULT(WARNING) << "Htp performance mode not valid.";
  }
}

static void ParseQnnContextPriority(std::string context_priority_string, qnn::ContextPriority& context_priority) {
  std::transform(context_priority_string.begin(),
                 context_priority_string.end(),
                 context_priority_string.begin(),
                 [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
  LOGS_DEFAULT(VERBOSE) << "QNN context priority: " << context_priority_string;
  if (context_priority_string == "low") {
    context_priority = qnn::ContextPriority::LOW;
  } else if (context_priority_string == "normal") {
    context_priority = qnn::ContextPriority::NORMAL;
  } else if (context_priority_string == "normal_high") {
    context_priority = qnn::ContextPriority::NORMAL_HIGH;
  } else if (context_priority_string == "high") {
    context_priority = qnn::ContextPriority::HIGH;
  } else {
    context_priority = qnn::ContextPriority::UNDEFINED;
    LOGS_DEFAULT(WARNING) << "QNN context priority: " << context_priority_string << " not valid, set to undefined.";
  }
}

void QNNExecutionProvider::ParseHtpGraphFinalizationOptimizationMode(const std::string& htp_graph_finalization_opt_mode_string) {
  LOGS_DEFAULT(VERBOSE) << "HTP graph finalization optimization mode: "
                        << htp_graph_finalization_opt_mode_string;

  if (htp_graph_finalization_opt_mode_string.empty() || htp_graph_finalization_opt_mode_string == "0") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  } else if (htp_graph_finalization_opt_mode_string == "1") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kMode1;
  } else if (htp_graph_finalization_opt_mode_string == "2") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kMode2;
  } else if (htp_graph_finalization_opt_mode_string == "3") {
    htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kMode3;
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid HTP graph finalization optimization mode: "
                          << htp_graph_finalization_opt_mode_string;
  }
}

static void ParseHtpArchitecture(const std::string& htp_arch_string, QnnHtpDevice_Arch_t& qnn_htp_arch) {
  if (htp_arch_string.empty() || htp_arch_string == "0") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  } else if (htp_arch_string == "68") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V68;
  } else if (htp_arch_string == "69") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V69;
  } else if (htp_arch_string == "73") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V73;
  } else if (htp_arch_string == "75") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V75;
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid HTP architecture: " << htp_arch_string;
  }
}

QNNExecutionProvider::QNNExecutionProvider(const ProviderOptions& provider_options_map,
                                           const SessionOptions* session_options)
    : IExecutionProvider{onnxruntime::kQnnExecutionProvider} {
  if (session_options) {
    disable_cpu_ep_fallback_ = session_options->config_options.GetConfigOrDefault(
                                   kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";

    context_cache_enabled_ = session_options->config_options.GetConfigOrDefault(
                                 kOrtSessionOptionEpContextEnable, "0") == "1";
    LOGS_DEFAULT(VERBOSE) << "Context cache enable: " << context_cache_enabled_;

    std::string embed_mode = session_options->config_options.GetConfigOrDefault(
        kOrtSessionOptionEpContextEmbedMode, "1");
    if ("1" == embed_mode) {
      qnn_context_embed_mode_ = true;
    } else if ("0" == embed_mode) {
      qnn_context_embed_mode_ = false;
    } else {
      LOGS_DEFAULT(VERBOSE) << "Invalid ep.context_embed_mode: " << embed_mode << " only 0 or 1 allowed. Set to 1.";
    }
    LOGS_DEFAULT(VERBOSE) << "User specified context cache embed mode: " << qnn_context_embed_mode_;

    context_cache_path_cfg_ = session_options->config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
    LOGS_DEFAULT(VERBOSE) << "User specified context cache path: " << context_cache_path_cfg_;
  }

  static const std::string BACKEND_PATH = "backend_path";
  auto backend_path_pos = provider_options_map.find(BACKEND_PATH);

  std::string backend_path;
  if (backend_path_pos != provider_options_map.end()) {
    backend_path = backend_path_pos->second;
    LOGS_DEFAULT(VERBOSE) << "Backend path: " << backend_path;
  } else {
    LOGS_DEFAULT(ERROR) << "No backend path provided.";
  }

  static const std::string PROFILING_LEVEL = "profiling_level";
  qnn::ProfilingLevel profiling_level = qnn::ProfilingLevel::OFF;
  const Env& env = Env::Default();
  auto& provider = env.GetTelemetryProvider();
  if (provider.IsEnabled()) {
    auto level = provider.Level();
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)) != 0) {
      if (level != 0) {
        if (level == 5) {
          LOGS_DEFAULT(INFO) << "Overriding profiling to basic based on ETW level: " << static_cast<int>(level);
          ParseProfilingLevel("basic", profiling_level);
        } else if (level < 5) {
          LOGS_DEFAULT(INFO) << "QNN Profiler ETW level not supported below level 5. Level: "
                             << static_cast<int>(level);
        } else {
          LOGS_DEFAULT(INFO) << "Overriding profiling to detailed based on ETW level: " << static_cast<int>(level);
          ParseProfilingLevel("detailed", profiling_level);
        }
      }
    }
  } else {
    auto profiling_level_pos = provider_options_map.find(PROFILING_LEVEL);
    if (profiling_level_pos != provider_options_map.end()) {
      ParseProfilingLevel(profiling_level_pos->second, profiling_level);
    }
  }

  static const std::string RPC_CONTROL_LANTENCY = "rpc_control_latency";
  auto latency_pos = provider_options_map.find(RPC_CONTROL_LANTENCY);
  if (latency_pos != provider_options_map.end()) {
    default_rpc_control_latency_ = static_cast<uint32_t>(std::stoul(latency_pos->second));
    LOGS_DEFAULT(VERBOSE) << "rpc_control_latency: " << default_rpc_control_latency_;
  }

  // default_htp_performance_mode from QNN EP option.
  // set it once only for each thread as default so user don't need to set it for every session run
  static const std::string HTP_PERFORMANCE_MODE = "htp_performance_mode";
  auto htp_performance_mode_pos = provider_options_map.find(HTP_PERFORMANCE_MODE);
  if (htp_performance_mode_pos != provider_options_map.end()) {
    ParseHtpPerformanceMode(htp_performance_mode_pos->second, default_htp_performance_mode_);
  }

  htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  static const std::string HTP_GRAPH_FINALIZATION_OPT_MODE = "htp_graph_finalization_optimization_mode";
  auto htp_graph_finalization_opt_mode_pos = provider_options_map.find(HTP_GRAPH_FINALIZATION_OPT_MODE);
  if (htp_graph_finalization_opt_mode_pos != provider_options_map.end()) {
    ParseHtpGraphFinalizationOptimizationMode(htp_graph_finalization_opt_mode_pos->second);
  }

  // Enable use of QNN Saver if the user provides a path the QNN Saver backend library.
  static const std::string QNN_SAVER_PATH_KEY = "qnn_saver_path";
  std::string qnn_saver_path;
  auto qnn_saver_path_pos = provider_options_map.find(QNN_SAVER_PATH_KEY);
  if (qnn_saver_path_pos != provider_options_map.end()) {
    qnn_saver_path = qnn_saver_path_pos->second;
    LOGS_DEFAULT(VERBOSE) << "User specified QNN Saver path: " << qnn_saver_path;
  }

  static const std::string QNN_CONTEXT_PRIORITY = "qnn_context_priority";
  qnn::ContextPriority context_priority = qnn::ContextPriority::NORMAL;
  auto qnn_context_priority_pos = provider_options_map.find(QNN_CONTEXT_PRIORITY);
  if (qnn_context_priority_pos != provider_options_map.end()) {
    ParseQnnContextPriority(qnn_context_priority_pos->second, context_priority);
  }

  static const std::string QNN_VTCM_MB = "vtcm_mb";
  auto qnn_vtcm_mb_pos = provider_options_map.find(QNN_VTCM_MB);
  if (qnn_vtcm_mb_pos != provider_options_map.end()) {
    vtcm_size_in_mb_ = std::stoi(qnn_vtcm_mb_pos->second);
    LOGS_DEFAULT(VERBOSE) << "vtcm_mb: " << vtcm_size_in_mb_;
    if (vtcm_size_in_mb_ <= 0) {
      LOGS_DEFAULT(WARNING) << "Skip invalid vtcm_mb: " << vtcm_size_in_mb_;
    }
  }

  static const std::string QNN_DEVICE_ID = "device_id";
  auto dev_id_pos = provider_options_map.find(QNN_DEVICE_ID);
  if (dev_id_pos != provider_options_map.end()) {
    int value = std::stoi(dev_id_pos->second);
    if (value < 0) {
      LOGS_DEFAULT(WARNING) << "Invalid device ID '" << value
                            << "', only >= 0 allowed. Set to " << device_id_ << ".";
    } else {
      device_id_ = static_cast<uint32_t>(value);
    }
  }

  static const std::string QNN_HTP_ARCH = "htp_arch";
  QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  auto htp_arch_pos = provider_options_map.find(QNN_HTP_ARCH);
  if (htp_arch_pos != provider_options_map.end()) {
    ParseHtpArchitecture(htp_arch_pos->second, htp_arch);
  }

  static const std::string QNN_SOC_MODEL = "soc_model";
  uint32_t soc_model = QNN_SOC_MODEL_UNKNOWN;
  auto soc_model_pos = provider_options_map.find(QNN_SOC_MODEL);
  if (soc_model_pos != provider_options_map.end()) {
    int value = std::stoi(soc_model_pos->second);
    if (value < 0) {
      LOGS_DEFAULT(WARNING) << "Invalid SoC Model '" << value
                            << "', only >= 0 allowed. Set to " << soc_model << ".";
    } else {
      soc_model = static_cast<uint32_t>(value);
    }
  }

  static const std::string QNN_HTP_FP16_MODE = "enable_htp_fp16_precision";
  auto htp_fp16_mode_pos = provider_options_map.find(QNN_HTP_FP16_MODE);
  if (htp_fp16_mode_pos != provider_options_map.end()) {
    if ("1" == htp_fp16_mode_pos->second) {
      enable_HTP_FP16_precision_ = true;
    } else if ("0" == htp_fp16_mode_pos->second) {
      enable_HTP_FP16_precision_ = false;
    } else {
      LOGS_DEFAULT(VERBOSE) << "Invalid enable_htp_fp16_precision: " << enable_HTP_FP16_precision_ << " only 0 or 1 allowed. Set to 0.";
    }
    LOGS_DEFAULT(VERBOSE) << "User specified enable_htp_fp16_precision: " << enable_HTP_FP16_precision_;
  }

  qnn_backend_manager_ = std::make_unique<qnn::QnnBackendManager>(
      std::move(backend_path),
      profiling_level,
      context_priority,
      std::move(qnn_saver_path),
      device_id_,
      htp_arch,
      soc_model);
}

QNNExecutionProvider::~QNNExecutionProvider() {
  // clean up thread local context caches
  std::lock_guard<OrtMutex> lock(context_state_.mutex);
  for (const auto& cache_weak : context_state_.caches_to_update_on_destruction) {
    const auto cache = cache_weak.lock();
    if (!cache) continue;
    ORT_IGNORE_RETURN_VALUE(cache->erase(this));
  }
}

bool QNNExecutionProvider::IsNodeSupported(qnn::QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                           const logging::Logger& logger) const {
  const std::string& op_type = node_unit.OpType();
  bool supported = false;
  const auto* op_builder = qnn::GetOpBuilder(op_type);
  if (op_builder == nullptr) {
    LOGS(logger, WARNING) << "Operators of type `" << node_unit.OpType() << "` are not supported by QNN EP."
                          << node_unit.OpType() << " node `" << node_unit.Name()
                          << "` will not be assigned to QNN EP.";
  } else {
    auto status = op_builder->IsOpSupported(qnn_model_wrapper,
                                            node_unit, logger);
    if (Status::OK() != status) {
      LOGS(logger, WARNING) << node_unit.OpType() << " node `" << node_unit.Name()
                            << "` is not supported: " << status.ErrorMessage();
    }
    supported = (Status::OK() == status);
  }
  return supported;
}

std::unordered_set<const Node*>
QNNExecutionProvider::GetSupportedNodes(const GraphViewer& graph_viewer,
                                        const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                        const size_t node_unit_size,
                                        bool is_qnn_ctx_model,
                                        const logging::Logger& logger) const {
  std::unordered_set<const Node*> supported_nodes{};
  // Filter in the EPContext node for QNN
  if (is_qnn_ctx_model) {
    for (const auto& node : graph_viewer.Nodes()) {
      NodeAttrHelper node_helper(node);
      std::string cache_source = node_helper.Get(qnn::SOURCE, "");

      std::transform(cache_source.begin(),
                     cache_source.end(),
                     cache_source.begin(),
                     [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

      if (qnn::EPCONTEXT_OP == node.OpType() && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
        LOGS(logger, VERBOSE) << "Node supported: [1] index: [" << node.Index()
                              << "] name: [" << node.Name()
                              << "] Operator type: [EPContext"
                              << "] index: [" << node.Index() << "]";
        supported_nodes.insert(&node);
      }
    }
    return supported_nodes;
  }

  // This holds the result of whether a NodeUnit is supported or not,
  // to prevent nodes in a NodeUnit to be checked for multiple times
  std::unordered_map<const NodeUnit*, bool> node_unit_supported_result;
  node_unit_supported_result.reserve(node_unit_size);

  std::unordered_set<std::string> initializer_input_lookup;
  auto graph_initializers = graph_viewer.GetAllInitializedTensors();
  for (auto graph_ini : graph_initializers) {
    initializer_input_lookup.emplace(graph_ini.first);
  }

  // Util function that initializes a table that maps a graph input or output name to its index.
  auto init_input_output_index_map = [](std::unordered_map<std::string, size_t>& index_map,
                                        const std::vector<const NodeArg*>& node_args) {
    const size_t num_args = node_args.size();
    for (size_t i = 0; i < num_args; i++) {
      index_map.emplace(node_args[i]->Name(), i);
    }
  };

  std::unordered_map<std::string, size_t> model_input_index_map;
  init_input_output_index_map(model_input_index_map, graph_viewer.GetInputs());  // GetInputs excludes initializers.

  std::unordered_map<std::string, size_t> model_output_index_map;
  init_input_output_index_map(model_output_index_map, graph_viewer.GetOutputs());

  auto qnn_model_wrapper = qnn::QnnModelWrapper(graph_viewer, logger,
                                                qnn_backend_manager_->GetQnnInterface(),
                                                qnn_backend_manager_->GetQnnBackendHandle(),
                                                model_input_index_map,
                                                model_output_index_map,
                                                initializer_input_lookup,
                                                qnn_backend_manager_->GetQnnBackendType());

  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    gsl::not_null<const onnxruntime::Node*> node(graph_viewer.GetNode(node_indices[i]));

    // Get the node_unit associated with the node. Note that the node may not be the node_unit's target node.
    const NodeUnit* node_unit = node_unit_map.at(node);

    // Visiting 'nodes' in topological order does not guarantee that 'node_units' are
    // also visited in topological order. Skip this node if it is not the node_unit's target node
    // to ensure 'node_units' are visited in topological order.
    if (node != &node_unit->GetNode()) {
      continue;
    }

    if (node_unit_supported_result.count(node_unit) != 0) {
      continue;  // Already handled this node unit
    }

    // Try to convert certain standalone DQ -> Q sequences into QNN Convert op
    auto convert_result = TryHandleConvertSequence(qnn_model_wrapper,
                                                   *node_unit,
                                                   node_unit_map,
                                                   logger,
                                                   true /*do_op_validation*/);
    if (!convert_result.status.IsOK()) {
      LOGS(logger, WARNING) << "Failed to convert DQ -> Q sequence to QNN Convert. "
                            << "Type: " << node_unit->OpType() << ", Node name: " << node_unit->Name() << ", "
                            << "Message: " << convert_result.status.ErrorMessage();
    }

    bool supported = false;

    if (convert_result.status.IsOK() && convert_result.q_node_unit) {  // Merged DQ -> Q sequence into QNN Convert op
      supported = true;

      // Mark the Q node unit as handled and supported here so that we don't try to process it again.
      node_unit_supported_result.insert({convert_result.q_node_unit, true});
      supported_nodes.insert(&convert_result.q_node_unit->GetNode());
    } else {
      supported = IsNodeSupported(qnn_model_wrapper, *node_unit, logger);
      LOGS(logger, VERBOSE) << "Node supported: [" << supported
                            << "] index: [" << node->Index()
                            << "] name: [" << node->Name()
                            << "] Operator type: [" << node->OpType()
                            << "] as part of the NodeUnit type: [" << node_unit->OpType()
                            << "] index: [" << node_unit->Index()
                            << "] name: [" << node_unit->Name()
                            << "]";
    }

    if (supported) {
      // If the node_unit is supported, add all of its nodes to the supported list.
      for (const auto* node_in_group : node_unit->GetAllNodesInGroup()) {
        supported_nodes.insert(node_in_group);
      }
    }

    node_unit_supported_result.insert({node_unit, supported});
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
  bool is_qnn_ctx_model = qnn::GraphHasEpContextNode(graph_viewer);

  // It will load the QnnSystem lib if is_qnn_ctx_model=true, and
  // delay the Qnn context creation to Compile() using the cached context binary
  auto rt = qnn_backend_manager_->SetupBackend(logger, is_qnn_ctx_model);
  if (Status::OK() != rt) {
    LOGS(logger, ERROR) << "QNN SetupBackend failed " << rt.ErrorMessage();
    return result;
  }

  if ((context_cache_enabled_ || is_qnn_ctx_model) && !IsNpuBackend(qnn_backend_manager_->GetQnnBackendType())) {
    LOGS(logger, ERROR) << "Qnn context cache only works for HTP or DSP backend.";
    return result;
  }

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer);

  const auto supported_nodes = GetSupportedNodes(graph_viewer, node_unit_map, node_unit_holder.size(),
                                                 is_qnn_ctx_model, logger);

  // Helper function that returns a string that lists all unsupported nodes.
  // Ex: { name: mul_123, type: Mul }, {}, ...
  auto get_unsupported_node_names = [&node_unit_holder, &supported_nodes]() -> std::string {
    std::stringstream ss;
    const size_t num_node_units = node_unit_holder.size();

    for (size_t i = 0; i < num_node_units; ++i) {
      const auto& node_unit = node_unit_holder[i];

      if (supported_nodes.find(&node_unit->GetNode()) == supported_nodes.end()) {
        ss << "{ name: " << node_unit->Name() << ", type: " << node_unit->OpType() << " }";
        if (i == num_node_units - 1) {
          ss << ", ";
        }
      }
    }

    return ss.str();
  };

  if (supported_nodes.empty()) {
    LOGS(logger, INFO) << "Number of partitions supported by QNN EP: 0";
    return result;
  }

  const auto gen_metadef_name = [&]() {
    uint64_t model_hash;
    int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
    return MakeString(QNN, "_", model_hash, "_", metadef_id);
  };

  const size_t num_nodes_in_graph = static_cast<size_t>(graph_viewer.NumberOfNodes());
  size_t num_of_supported_nodes = 0;

  // Create partitions from supported nodes.
  std::vector<std::unique_ptr<ComputeCapability>> partitions = utils::CreateSupportedPartitions(
      graph_viewer, supported_nodes, {}, gen_metadef_name, QNN, kQnnExecutionProvider, &node_unit_map, true);

  // Filter out partitions that consist of a single QuantizeLinear or DequantizeLinear node.
  // We also count the number of supported nodes in all valid partitions.
  for (auto& partition : partitions) {
    bool is_valid_partition = true;
    size_t nodes_in_partition = 0;

    if (partition && partition->sub_graph) {
      nodes_in_partition = partition->sub_graph->nodes.size();

      if (nodes_in_partition == 1 && !is_qnn_ctx_model) {
        const Node* node = graph_viewer.GetNode(partition->sub_graph->nodes[0]);

        if (!node) {
          LOGS(logger, ERROR) << "QNN EP: Invalid node in partition of one node.";
          is_valid_partition = false;
        } else if (node->OpType() == "QuantizeLinear" || node->OpType() == "DequantizeLinear") {
          LOGS(logger, WARNING) << "QNN EP does not support a single Quantize/Dequantize node in a partition.";
          is_valid_partition = false;
        }
      }
    } else {
      LOGS(logger, ERROR) << "QNN EP: Invalid partition.";
      is_valid_partition = false;
    }

    if (is_valid_partition) {
      result.push_back(std::move(partition));
      num_of_supported_nodes += nodes_in_partition;
    }
  }  // for

  const size_t num_of_partitions = result.size();
  const auto summary_msg = MakeString("Number of partitions supported by QNN EP: ", num_of_partitions,
                                      ", number of nodes in the graph: ", num_nodes_in_graph,
                                      ", number of nodes supported by QNN: ", num_of_supported_nodes);
  LOGS(logger, INFO) << summary_msg;

  // Print list of unsupported nodes to the ERROR logger if the CPU EP
  // has been disabled for this inference session.
  if (!is_qnn_ctx_model && disable_cpu_ep_fallback_ && num_nodes_in_graph != num_of_supported_nodes) {
    LOGS(logger, ERROR) << "Unsupported nodes in QNN EP: " << get_unsupported_node_names();
  }

  return result;
}

DataLayout QNNExecutionProvider::GetPreferredLayout() const {
  return DataLayout::NHWC;
}

Status QNNExecutionProvider::CreateComputeFunc(std::vector<NodeComputeInfo>& node_compute_funcs,
                                               const logging::Logger& logger) {
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

  return Status::OK();
}

void QNNExecutionProvider::InitQnnGraphConfigs(qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const {
  if (qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::HTP) {
    if (htp_graph_finalization_opt_mode_ != qnn::HtpGraphFinalizationOptimizationMode::kDefault) {
      QnnHtpGraph_CustomConfig_t& htp_graph_opt_config = configs_builder.PushCustomConfig();
      htp_graph_opt_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
      htp_graph_opt_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
      htp_graph_opt_config.optimizationOption.floatValue = static_cast<float>(htp_graph_finalization_opt_mode_);

      QnnGraph_Config_t& graph_opt_config = configs_builder.PushConfig();
      graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config.customConfig = &htp_graph_opt_config;
    }

    if (vtcm_size_in_mb_ > 0) {
      QnnHtpGraph_CustomConfig_t& htp_graph_opt_config_vtcm = configs_builder.PushCustomConfig();
      htp_graph_opt_config_vtcm.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
      htp_graph_opt_config_vtcm.vtcmSizeInMB = static_cast<uint32_t>(vtcm_size_in_mb_);

      QnnGraph_Config_t& graph_opt_config_vtcm = configs_builder.PushConfig();
      graph_opt_config_vtcm.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config_vtcm.customConfig = &htp_graph_opt_config_vtcm;
    }

    if (enable_HTP_FP16_precision_) {
      QnnHtpGraph_CustomConfig_t& htp_graph_precision_config = configs_builder.PushCustomConfig();
      htp_graph_precision_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      htp_graph_precision_config.precision = QNN_PRECISION_FLOAT16;

      QnnGraph_Config_t& graph_precision_config = configs_builder.PushConfig();
      graph_precision_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_precision_config.customConfig = &htp_graph_precision_config;
    }
  }
}

Status QNNExecutionProvider::CompileFromOrtGraph(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs,
                                                 const logging::Logger& logger) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    std::unique_ptr<qnn::QnnModel> qnn_model = std::make_unique<qnn::QnnModel>(logger,
                                                                               qnn_backend_manager_.get());

    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t> graph_configs_builder(QNN_GRAPH_CONFIG_INIT,
                                                                                                QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
    InitQnnGraphConfigs(graph_configs_builder);

    ORT_RETURN_IF_ERROR(qnn_model->ComposeGraph(graph_viewer, fused_node, graph_configs_builder.GetQnnConfigs()));
    ORT_RETURN_IF_ERROR(qnn_model->FinalizeGraphs());
    ORT_RETURN_IF_ERROR(qnn_model->SetupQnnInputOutput());

    LOGS(logger, VERBOSE) << "fused node name: " << fused_node.Name();
    qnn_models_.emplace(fused_node.Name(), std::move(qnn_model));

    ORT_RETURN_IF_ERROR(CreateComputeFunc(node_compute_funcs, logger));
  }
  return Status::OK();
}

Status QNNExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                     std::vector<NodeComputeInfo>& node_compute_funcs) {
  const auto& logger = *GetLogger();

  bool is_qnn_ctx_model = qnn::IsFusedGraphHasCtxNode(fused_nodes_and_graphs);

  onnxruntime::PathString context_cache_path;
  bool is_ctx_file_exist = false;
  if (is_qnn_ctx_model || context_cache_enabled_) {
    const onnxruntime::GraphViewer& graph_viewer_0(fused_nodes_and_graphs[0].filtered_graph);
    is_ctx_file_exist = qnn::ValidateContextCacheFilePath(is_qnn_ctx_model,
                                                          context_cache_path_cfg_,
                                                          graph_viewer_0.ModelPath().ToPathString(),
                                                          context_cache_path);
  }

  ORT_RETURN_IF(is_ctx_file_exist && !is_qnn_ctx_model && context_cache_enabled_,
                "The inference session is created from normal ONNX model. And an EP context model file is provided and existed. ",
                "Please remove the EP context model manually if you want to re-generate it.");

  if (is_qnn_ctx_model) {
    // Table<EPContext node name, QnnModel>, the node name is the graph_meta_id (old) created from user model which used to generate the EP context model
    // for this session (created from an EP context model), the graph_meta_id is new
    std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models;

    int main_context_pos = -1;
    ORT_RETURN_IF_ERROR(qnn::GetMainContextNode(fused_nodes_and_graphs, qnn_backend_manager_.get(),
                                                logger, main_context_pos, qnn_models));

    const onnxruntime::GraphViewer& main_ctx_graph_viewer(fused_nodes_and_graphs[main_context_pos].filtered_graph);
    // Create QNN context from the cached binary, deserialize the QNN graph from the binary
    ORT_RETURN_IF_ERROR(qnn::LoadQnnCtxFromOnnxGraph(main_ctx_graph_viewer,
                                                     context_cache_path,
                                                     qnn_backend_manager_.get(),
                                                     qnn_models,
                                                     logger));

    for (auto fused_node_and_graph : fused_nodes_and_graphs) {
      const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
      const auto& ep_context_node = graph_viewer.Nodes().begin();
      const Node& fused_node = fused_node_and_graph.fused_node;
      const std::string& graph_meta_id = fused_node.Name();
      std::string key = ep_context_node->Name();
      ORT_RETURN_IF(qnn_models.find(key) == qnn_models.end(), key + " key name not exist in table qnn_models.");
      auto qnn_model = std::move(qnn_models[key]);
      ORT_RETURN_IF_ERROR(qnn_model->SetGraphInputOutputInfo(graph_viewer, fused_node));
      ORT_RETURN_IF_ERROR(qnn_model->SetupQnnInputOutput());

      // fused node name is QNNExecutionProvider_QNN_[hash_id]_[id]
      // the name here must be same with context->node_name in compute_info
      qnn_models_.emplace(graph_meta_id, std::move(qnn_model));

      ORT_RETURN_IF_ERROR(CreateComputeFunc(node_compute_funcs, logger));
    }

    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(CompileFromOrtGraph(fused_nodes_and_graphs, node_compute_funcs, logger));
  // Generate QNN context model if it's QDQ model + context_cache_enabled=true + not exist already
  if (!is_qnn_ctx_model && context_cache_enabled_ && !is_ctx_file_exist) {
    // All partitioned graph share single QNN context, included in the same context binary
    uint64_t buffer_size(0);
    auto context_buffer = qnn_backend_manager_->GetContextBinaryBuffer(buffer_size);
    qnn_ep_context_model_ = std::make_unique<Model>("qnn_ep_context_model", false, logger);
    ORT_RETURN_IF_ERROR(qnn::CreateEPContextNodes(qnn_ep_context_model_.get(),
                                                  context_buffer.get(),
                                                  buffer_size,
                                                  qnn_backend_manager_->GetSdkVersion(),
                                                  fused_nodes_and_graphs,
                                                  qnn_models_,
                                                  context_cache_path,
                                                  qnn_context_embed_mode_,
                                                  logger));
  }
  return Status::OK();
}

const InlinedVector<const Node*> QNNExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> ep_context_nodes;
  if (qnn_ep_context_model_) {
    const auto& graph = qnn_ep_context_model_->MainGraph();
    for (const auto& node : graph.Nodes()) {
      ep_context_nodes.push_back(graph.GetNode(node.Index()));
    }
  }

  return ep_context_nodes;
}

QNNExecutionProvider::PerThreadContext::PerThreadContext(qnn::QnnBackendManager* qnn_backend_manager,
                                                         uint32_t device_id,
                                                         uint32_t core_id,
                                                         qnn::HtpPerformanceMode default_htp_performance_mode,
                                                         uint32_t default_rpc_control_latency)
    : qnn_backend_manager_(qnn_backend_manager) {
  Status rt = qnn_backend_manager_->CreateHtpPowerCfgId(device_id, core_id, htp_power_config_id_);
  is_htp_power_config_id_valid_ = rt.IsOK();
  // default_htp_performance_mode and default_rpc_control_latency are from QNN EP option.
  // set it once only for each thread as default so user don't need to set it for every session run
  if (is_htp_power_config_id_valid_) {
    if (qnn::HtpPerformanceMode::kHtpDefault != default_htp_performance_mode) {
      ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->SetHtpPowerConfig(htp_power_config_id_,
                                                                      default_htp_performance_mode));
    }
    if (default_rpc_control_latency > 0) {
      ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->SetRpcControlLatency(htp_power_config_id_,
                                                                         default_rpc_control_latency));
    }
  }
}

QNNExecutionProvider::PerThreadContext::~PerThreadContext() {
  if (is_htp_power_config_id_valid_) {
    ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->DestroyHTPPowerConfigID(htp_power_config_id_));
  }
}

QNNExecutionProvider::PerThreadContext& QNNExecutionProvider::GetPerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  // try to use cached context
  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    ORT_ENFORCE(cached_context);
    return *cached_context;
  }

  // get context and update cache
  std::shared_ptr<PerThreadContext> context;
  {
    std::lock_guard<OrtMutex> lock(context_state_.mutex);

    // get or create a context
    if (context_state_.retired_context_pool.empty()) {
      uint32_t core_id = 0;
      context = std::make_shared<PerThreadContext>(qnn_backend_manager_.get(), device_id_, core_id,
                                                   default_htp_performance_mode_, default_rpc_control_latency_);
    } else {
      context = context_state_.retired_context_pool.back();
      context_state_.retired_context_pool.pop_back();
    }

    // insert into active_contexts, should not already be present
    const auto active_contexts_insert_result = context_state_.active_contexts.insert(context);
    ORT_ENFORCE(active_contexts_insert_result.second);

    // insert into caches_to_update_on_destruction, may already be present
    ORT_IGNORE_RETURN_VALUE(context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache));
  }

  per_thread_context_cache->insert(std::make_pair(this, context));

  return *context;
}

void QNNExecutionProvider::ReleasePerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  ORT_ENFORCE(cached_context_it != per_thread_context_cache->end());
  auto cached_context = cached_context_it->second.lock();
  ORT_ENFORCE(cached_context);

  {
    std::lock_guard<OrtMutex> lock(context_state_.mutex);
    context_state_.active_contexts.erase(cached_context);
    context_state_.retired_context_pool.push_back(cached_context);
  }

  per_thread_context_cache->erase(cached_context_it);
}

Status QNNExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
  auto backend_type = qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return Status::OK();
  }

  std::string htp_perf_mode = "";
  qnn::HtpPerformanceMode htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  if (run_options.config_options.TryGetConfigEntry(kOrtRunOptionsConfigQnnPerfMode, htp_perf_mode)) {
    // set power mode
    ParseHtpPerformanceMode(htp_perf_mode, htp_performance_mode);
  }

  std::string rpc_latency = "";
  uint32_t rpc_control_latency = 0;
  if (run_options.config_options.TryGetConfigEntry(kOrtRunOptionsConfigQnnRpcControlLatency, rpc_latency)) {
    rpc_control_latency = static_cast<uint32_t>(std::stoul(rpc_latency));
    LOGS_DEFAULT(VERBOSE) << "rpc_control_latency: " << rpc_control_latency;
  }

  if (GetPerThreadContext().IsHtpPowerConfigIdValid()) {
    if (qnn::HtpPerformanceMode::kHtpDefault != htp_performance_mode) {
      ORT_RETURN_IF_ERROR(qnn_backend_manager_->SetHtpPowerConfig(GetPerThreadContext().GetHtpPowerConfigId(),
                                                                  htp_performance_mode));
    }

    if (rpc_control_latency > 0) {
      ORT_RETURN_IF_ERROR(qnn_backend_manager_->SetRpcControlLatency(GetPerThreadContext().GetHtpPowerConfigId(),
                                                                     rpc_control_latency));
    }
  }

  return Status::OK();
}

Status QNNExecutionProvider::OnRunEnd(bool /*sync_stream*/, const onnxruntime::RunOptions& run_options) {
  auto backend_type = qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return Status::OK();
  }

  std::string htp_perf_mode = "";
  qnn::HtpPerformanceMode htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  if (run_options.config_options.TryGetConfigEntry(kOrtRunOptionsConfigQnnPerfModePostRun, htp_perf_mode)) {
    // set power mode
    ParseHtpPerformanceMode(htp_perf_mode, htp_performance_mode);
  }

  if (qnn::HtpPerformanceMode::kHtpDefault != htp_performance_mode) {
    if (!GetPerThreadContext().IsHtpPowerConfigIdValid()) {
      return Status::OK();
    }
    ORT_RETURN_IF_ERROR(qnn_backend_manager_->SetHtpPowerConfig(GetPerThreadContext().GetHtpPowerConfigId(),
                                                                htp_performance_mode));
  }

  return Status::OK();
}
}  // namespace onnxruntime
