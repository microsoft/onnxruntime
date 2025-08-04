// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn-abi/qnn_ep.h"

#include <unordered_map>
#include <vector>
#include <memory>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>

#include "HTP/QnnHtpGraph.h"

#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/qnn_ep_factory.h"
#include "core/providers/qnn-abi/shared_context.h"
#include "core/providers/qnn-abi/builder/qnn_backend_manager.h"
#include "core/providers/qnn-abi/builder/qnn_configs_helper.h"
#include "core/providers/qnn-abi/builder/qnn_model.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/qnn_ep_utils.h"

// Forward declarations for NodeUnit-related classes
namespace onnxruntime {

constexpr const char* QNN = "QNN";

static std::string MakeSharedLibraryPath(std::string_view name) {
#if defined(_WIN32)
  return MakeString(name, ".dll");
#else
  return MakeString("lib", name, ".so");
#endif
}

const std::string kDefaultCpuBackendPath = MakeSharedLibraryPath("QnnCpu");
const std::string kDefaultGpuBackendPath = MakeSharedLibraryPath("QnnGpu");
const std::string kDefaultHtpBackendPath = MakeSharedLibraryPath("QnnHtp");
const std::string kDefaultSaverBackendPath = MakeSharedLibraryPath("QnnSaver");
const std::string kDefaultIrBackendPath = MakeSharedLibraryPath("QnnIr");

static bool ParseBackendTypeName(std::string_view backend_type_name,
                                 std::string& backend_path,
                                 const logging::Logger& logger) {
  constexpr std::string_view kCpuBackendTypeName{"cpu"};
  constexpr std::string_view kGpuBackendTypeName{"gpu"};
  constexpr std::string_view kHtpBackendTypeName{"htp"};
  constexpr std::string_view kSaverBackendTypeName{"saver"};
  constexpr std::string_view kIrBackendTypeName{"ir"};

  constexpr std::array kAllowedBackendTypeNames{
      kCpuBackendTypeName,
      kGpuBackendTypeName,
      kHtpBackendTypeName,
      kSaverBackendTypeName,
      kIrBackendTypeName,
  };

  std::optional<std::string> associated_backend_path{};
  if (backend_type_name == kCpuBackendTypeName) {
    associated_backend_path = kDefaultCpuBackendPath;
  } else if (backend_type_name == kGpuBackendTypeName) {
    associated_backend_path = kDefaultGpuBackendPath;
  } else if (backend_type_name == kHtpBackendTypeName) {
    associated_backend_path = kDefaultHtpBackendPath;
  } else if (backend_type_name == kSaverBackendTypeName) {
    associated_backend_path = kDefaultSaverBackendPath;
  } else if (backend_type_name == kIrBackendTypeName) {
    associated_backend_path = kDefaultIrBackendPath;
  }

  if (associated_backend_path.has_value()) {
    backend_path = std::move(*associated_backend_path);
    return true;
  }

  std::ostringstream warning{};
  warning << "Invalid backend type name: " << backend_type_name << ". Allowed backend type names: ";
  for (size_t i = 0; i < kAllowedBackendTypeNames.size(); ++i) {
    warning << kAllowedBackendTypeNames[i];
    if (i + 1 < kAllowedBackendTypeNames.size()) {
      warning << ", ";
    }
  }
  LOGS(logger, WARNING) << warning.str();
  return false;
}

static void ParseProfilingLevel(std::string profiling_level_string,
                                qnn::ProfilingLevel& profiling_level,
                                const logging::Logger& logger) {
  std::transform(profiling_level_string.begin(), profiling_level_string.end(),
                 profiling_level_string.begin(), [](unsigned char c) { return std::tolower(c); });
  LOGS(logger, INFO) << "profiling_level: " << profiling_level_string;
  if (profiling_level_string == "off") {
    profiling_level = qnn::ProfilingLevel::OFF;
  } else if (profiling_level_string == "basic") {
    profiling_level = qnn::ProfilingLevel::BASIC;
  } else if (profiling_level_string == "detailed") {
    profiling_level = qnn::ProfilingLevel::DETAILED;
  } else {
    LOGS(logger, WARNING) << "Profiling level not valid.";
  }
}

static void ParseHtpPerformanceMode(std::string htp_performance_mode_string,
                                    qnn::HtpPerformanceMode& htp_performance_mode,
                                    const logging::Logger& logger) {
  std::transform(htp_performance_mode_string.begin(), htp_performance_mode_string.end(),
                 htp_performance_mode_string.begin(), [](unsigned char c) { return std::tolower(c); });
  LOGS(logger, VERBOSE) << "Htp performance mode: " << htp_performance_mode_string;
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
    LOGS(logger, WARNING) << "Htp performance mode not valid.";
  }
}

static void ParseQnnContextPriority(std::string context_priority_string,
                                    qnn::ContextPriority& context_priority,
                                    const logging::Logger& logger) {
  std::transform(context_priority_string.begin(), context_priority_string.end(),
                 context_priority_string.begin(), [](unsigned char c) { return std::tolower(c); });
  LOGS(logger, VERBOSE) << "QNN context priority: " << context_priority_string;
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
    LOGS(logger, WARNING) << "QNN context priority: " << context_priority_string << " not valid, set to undefined.";
  }
}

void QnnEp::ParseHtpGraphFinalizationOptimizationMode(const std::string& htp_graph_finalization_opt_mode_string,
                                                      const logging::Logger& logger) {
  LOGS(logger, VERBOSE) << "HTP graph finalization optimization mode: "
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
    LOGS(logger, WARNING) << "Invalid HTP graph finalization optimization mode: "
                          << htp_graph_finalization_opt_mode_string;
  }
}

static void ParseHtpArchitecture(const std::string& htp_arch_string,
                                 QnnHtpDevice_Arch_t& qnn_htp_arch,
                                 const logging::Logger& logger) {
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
    LOGS(logger, WARNING) << "Invalid HTP architecture: " << htp_arch_string;
  }
}

static void ParseOpPackages(const std::string& op_packages_string,
                            std::vector<onnxruntime::qnn::OpPackage>& op_packages,
                            const logging::Logger& logger) {
  for (const auto& op_package : utils::SplitString(op_packages_string, ",", true)) {
    auto splitStrings = utils::SplitString(op_package, ":", true);
    if (splitStrings.size() < 3 || splitStrings.size() > 4) {
      LOGS(logger, WARNING) << "Invalid op_package passed, "
                            << "expected <OpType>:<PackagePath>:<InterfaceSymbolName>[:<Target>], got "
                            << op_package;
      LOGS(logger, WARNING) << "Skip registration.";
      continue;
    }

    std::string op_type = std::string(splitStrings[0]);
    std::string op_package_path = std::string(splitStrings[1]);
    std::string op_package_interface = std::string(splitStrings[2]);
    std::string op_package_target;

    if (op_type.empty()) {
      LOGS(logger, WARNING) << "Op type is empty. Skip registration";
      continue;
    }

    if (op_package_path.empty()) {
      LOGS(logger, WARNING) << "Op package path is empty. Skip registration";
      continue;
    }

    if (op_package_interface.empty()) {
      LOGS(logger, WARNING) << "Op package interface is empty. Skip registration";
      continue;
    }

    LOGS(logger, VERBOSE) << "Loading op package from path: " << op_package_path << " for op " << op_type;
    LOGS(logger, VERBOSE) << "Op package interface: " << op_package_interface;
    if (splitStrings.size() > 3 && splitStrings[3].size()) {
      op_package_target = std::string(splitStrings[3]);
      LOGS(logger, VERBOSE) << "Op package target: " << op_package_target;
    }
    op_packages.push_back({op_type, op_package_path, op_package_interface, op_package_target});
  }
}

static bool ParseBoolOption(const OrtApi& ort_api,
                            const OrtSessionOptions& session_options,
                            const char* key,
                            bool default_value,
                            const logging::Logger& logger) {
  bool result = default_value;
  std::string value_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options, key, default_value ? "1" : "0", value_str);

  if ("1" == value_str) {
    result = true;
  } else if ("0" == value_str) {
    result = false;
  } else {
    LOGS(logger, VERBOSE) << "Invalid value for " << key << " (" << value_str << "). Only 0 or 1 allowed.";
  }
  LOGS(logger, VERBOSE) << "Using " << key << ": " << result;

  return result;
}

static qnn::ProfilingLevel GetProfilingLevelFromETWLevel(unsigned char level, const logging::Logger& logger) {
  if (level == 5) {
    LOGS(logger, INFO) << "Overriding profiling to basic based on ETW level: " << static_cast<int>(level);
    return qnn::ProfilingLevel::BASIC;
  } else if (level < 5) {
    LOGS(logger, INFO) << "QNN Profiler ETW level not supported below level 5. Level: "
                       << static_cast<int>(level);
    return qnn::ProfilingLevel::OFF;
  } else {
    LOGS(logger, INFO) << "Overriding profiling to detailed based on ETW level: " << static_cast<int>(level);
    return qnn::ProfilingLevel::DETAILED;
  }
}

static std::unique_ptr<qnn::QnnSerializerConfig> ParseSerializerBackendOptions(const OrtApi& ort_api,
                                                                               const OrtSessionOptions& session_options,
                                                                               const logging::Logger& logger) {
  // Enable use of QNN Saver if the user provides a path the QNN Saver backend library.
  static const std::string QNN_SAVER_PATH_KEY = "qnn_saver_path";
  std::string qnn_saver_path;
  GetSessionConfigEntryOrDefault(ort_api, session_options, QNN_SAVER_PATH_KEY.c_str(), "", qnn_saver_path);
  if (!qnn_saver_path.empty()) {
    LOGS(logger, VERBOSE) << "User specified QNN Saver path: " << qnn_saver_path;
    return qnn::QnnSerializerConfig::CreateSaver(qnn_saver_path);
  }

  static const std::string DUMP_QNN_IR_DLC = "dump_qnn_ir_dlc";
  auto dump_qnn_ir_dlc = ParseBoolOption(ort_api, session_options, DUMP_QNN_IR_DLC.c_str(), false, logger);

  static const std::string DUMP_QNN_IR_DLC_DIR = "dump_qnn_ir_dlc_dir";
  std::string qnn_ir_dlc_dir;
  GetSessionConfigEntryOrDefault(ort_api, session_options, DUMP_QNN_IR_DLC_DIR.c_str(), "", qnn_ir_dlc_dir);
  if (!qnn_ir_dlc_dir.empty()) {
    if (dump_qnn_ir_dlc) {
      LOGS(logger, INFO) << "IR DLC directory: " << qnn_ir_dlc_dir;
    } else {
      LOGS(logger, WARNING) << "Provided a directory for dumping QNN graphs to DLC, "
                            << "but did not set dump_qnn_ir_dlc to 1.";
    }
  }

  static const std::string QNN_IR_BACKEND_PATH = "qnn_ir_backend_path";
  std::string qnn_ir_backend_path = kDefaultIrBackendPath;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options,
                                 QNN_IR_BACKEND_PATH.c_str(),
                                 kDefaultIrBackendPath,
                                 qnn_ir_backend_path);
  if (qnn_ir_backend_path != kDefaultIrBackendPath) {
    if (dump_qnn_ir_dlc) {
      LOGS(logger, INFO) << "IR backend path: " << qnn_ir_backend_path;
    } else {
      LOGS(logger, WARNING) << "Provided a path to the Ir backend for dumping QNN graphs to DLC, "
                            << "but did not set dump_qnn_ir_dlc to 1.";
    }
  }

  if (dump_qnn_ir_dlc) {
    return qnn::QnnSerializerConfig::CreateIr(std::move(qnn_ir_backend_path), std::move(qnn_ir_dlc_dir));
  }

  return nullptr;
}

QnnEp::QnnEp(QnnEpFactory& factory,
             const std::string& name,
             const OrtSessionOptions& session_options,
             const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      logger_{logger},
      logger_in_{*(logger_.ToInternal())},
      session_options_{session_options} {
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;

  // Initialize from session options
  {
    // Get disable_cpu_ep_fallback setting from session options
    std::string disable_cpu_ep_fallback_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionsDisableCPUEPFallback,
                                   "0",
                                   disable_cpu_ep_fallback_str);
    disable_cpu_ep_fallback_ = disable_cpu_ep_fallback_str == "1";

    // Get context cache settings
    std::string context_cache_enabled_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextEnable,
                                   "0",
                                   context_cache_enabled_str);
    context_cache_enabled_ = context_cache_enabled_str == "1";
    LOGS(logger_in_, VERBOSE) << "Context cache enable: " << context_cache_enabled_;

    std::string embed_mode;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextEmbedMode,
                                   "0",
                                   embed_mode);
    if ("1" == embed_mode) {
      qnn_context_embed_mode_ = true;
    } else if ("0" == embed_mode) {
      qnn_context_embed_mode_ = false;
    } else {
      LOGS(logger_in_, VERBOSE) << "Invalid ep.context_embed_mode: " << embed_mode << " only 0 or 1 allowed. Set to 1.";
    }
    LOGS(logger_in_, VERBOSE) << "User specified context cache embed mode: " << qnn_context_embed_mode_;

    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextFilePath,
                                   "",
                                   context_cache_path_cfg_);
    LOGS(logger_in_, VERBOSE) << "User specified context cache path: " << context_cache_path_cfg_;

    // For the case that workaround QNN context PD memory limit, user need split the model into pieces and
    // generate the QNN context model separately.
    // It could happen that the generated EPContext node in separate graph has same node name.
    // User can set this context_node_name_prefix for each split pieces to avoid that happens.
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextNodeNamePrefix,
                                   "",
                                   context_node_name_prefix_);
    LOGS(logger_in_, VERBOSE) << "User specified QNN context node name prefix: " << context_node_name_prefix_;

    std::string share_ep_contexts_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionShareEpContexts,
                                   "0",
                                   share_ep_contexts_str);
    share_ep_contexts_ = share_ep_contexts_str == "1";
    LOGS(logger_in_, VERBOSE) << "User specified option - share EP contexts across sessions: " << share_ep_contexts_;

    std::string stop_share_ep_contexts_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionStopShareEpContexts,
                                   "0",
                                   stop_share_ep_contexts_str);
    stop_share_ep_contexts_ = stop_share_ep_contexts_str == "1";
    LOGS(logger_in_, VERBOSE) << "User specified option - stop share EP contexts across sessions: "
                              << stop_share_ep_contexts_;
  }

  std::string backend_path = kDefaultHtpBackendPath;
  {
    std::optional<std::string> backend_path_from_options{};

    // Get backend type and path from session options
    std::string backend_type;
    std::string backend_path_option;

    // TODO: Fix key with prefix.
    GetSessionConfigEntryOrDefault(ort_api, session_options_, "ep.qnnabitestprovider.backend_type", "", backend_type);
    GetSessionConfigEntryOrDefault(ort_api, session_options_, "ep.qnnabitestprovider.backend_path", "", backend_path_option);
    std::cout << "DEBUG: BackendType " << backend_type << std::endl;

    // Check if both options are provided
    if (!backend_type.empty() && !backend_path_option.empty()) {
      LOGS(logger_in_, ERROR) << "Only one of 'backend_type' and 'backend_path' should be set.";
    }
    if (!backend_type.empty()) {
      if (std::string parsed_backend_path; ParseBackendTypeName(backend_type, parsed_backend_path, logger_in_)) {
        backend_path_from_options = parsed_backend_path;
      } else {
        LOGS(logger_in_, ERROR) << "Failed to parse '" << "backend_type" << "' value.";
      }
    } else if (!backend_path_option.empty()) {
      backend_path_from_options = backend_path_option;
    }

    // Use the determined backend path or default
    if (backend_path_from_options.has_value()) {
      backend_path = std::move(*backend_path_from_options);
    } else {
      LOGS(logger_in_, VERBOSE) << "Using default backend path: " << backend_path;
    }

    LOGS(logger_in_, VERBOSE) << "Using backend path: " << backend_path;
  }

  std::unique_ptr<qnn::QnnSerializerConfig> qnn_serializer_config = ParseSerializerBackendOptions(ort_api,
                                                                                                  session_options_,
                                                                                                  logger_in_);

  std::string profiling_file_path;
  static const std::string PROFILING_LEVEL = "profiling_level";
  qnn::ProfilingLevel profiling_level = qnn::ProfilingLevel::OFF;
  // separate out the profiling level for ETW in case it gets disabled later when we extract the events
  // set to invalid to indicate that ETW is no enabled when we setup QNN
  qnn::ProfilingLevel profiling_level_etw = qnn::ProfilingLevel::INVALID;

#ifdef _WIN32
  auto& provider = qnn::QnnTelemetry::Instance();
  if (provider.IsEnabled()) {
    auto level = provider.Level();
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)) != 0) {
      if (level != 0) {
        profiling_level_etw = GetProfilingLevelFromETWLevel(level, logger_in_);
      }
    }
  }
#endif  // defined(_WIN32)

  // Get profiling settings from session options
  std::string profiling_level_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "profiling_level", "off", profiling_level_str);
  ParseProfilingLevel(profiling_level_str, profiling_level, logger_in_);

  GetSessionConfigEntryOrDefault(ort_api, session_options_, "profiling_file_path", "", profiling_file_path);
  LOGS(logger_in_, VERBOSE) << "Profiling file path: " << profiling_file_path;

  // Get RPC control latency from session options
  std::string rpc_control_latency_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "rpc_control_latency", "0", rpc_control_latency_str);
  if (!rpc_control_latency_str.empty() && rpc_control_latency_str != "0") {
    default_rpc_control_latency_ = static_cast<uint32_t>(std::stoul(rpc_control_latency_str));
    LOGS(logger_in_, VERBOSE) << "rpc_control_latency: " << default_rpc_control_latency_;
  }

  // default_htp_performance_mode from QNN EP option.
  // set it once only for each thread as default so user don't need to set it for every session run
  std::string htp_performance_mode_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "htp_performance_mode", "", htp_performance_mode_str);
  if (!htp_performance_mode_str.empty()) {
    ParseHtpPerformanceMode(htp_performance_mode_str, default_htp_performance_mode_, logger_in_);
  }

  // HTP graph finalization optimization mode
  htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  std::string htp_graph_finalization_opt_mode_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 "htp_graph_finalization_optimization_mode",
                                 "",
                                 htp_graph_finalization_opt_mode_str);
  if (!htp_graph_finalization_opt_mode_str.empty()) {
    ParseHtpGraphFinalizationOptimizationMode(htp_graph_finalization_opt_mode_str, logger_in_);
  }

  // QNN context priority
  qnn::ContextPriority context_priority = qnn::ContextPriority::NORMAL;
  std::string context_priority_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "qnn_context_priority", "", context_priority_str);
  if (!context_priority_str.empty()) {
    ParseQnnContextPriority(context_priority_str, context_priority, logger_in_);
  }

  // VTCM MB
  std::string vtcm_mb_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "vtcm_mb", "0", vtcm_mb_str);
  if (!vtcm_mb_str.empty() && vtcm_mb_str != "0") {
    vtcm_size_in_mb_ = std::stoi(vtcm_mb_str);
    LOGS(logger_in_, VERBOSE) << "vtcm_mb: " << vtcm_size_in_mb_;
    if (vtcm_size_in_mb_ <= 0) {
      LOGS(logger_in_, WARNING) << "Skip invalid vtcm_mb: " << vtcm_size_in_mb_;
    }
  }

  // VTCM backup buffer sharing
  std::string enable_vtcm_backup_buffer_sharing_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 "enable_vtcm_backup_buffer_sharing",
                                 "0",
                                 enable_vtcm_backup_buffer_sharing_str);
  if (enable_vtcm_backup_buffer_sharing_str == "1") {
    enable_vtcm_backup_buffer_sharing_ = true;
  } else if (enable_vtcm_backup_buffer_sharing_str != "0") {
    LOGS(logger_in_, WARNING) << "Invalid value entered for enable_vtcm_backup_buffer_sharing"
                              << ": " << enable_vtcm_backup_buffer_sharing_str
                              << ", only 1 or 0 are allowed. Setting to 0.";
  }

  LOGS(logger_in_, VERBOSE) << "User specified enable_vtcm_backup_buffer_sharing: " << enable_vtcm_backup_buffer_sharing_;

#if QNN_API_VERSION_MAJOR < 2 || ((QNN_API_VERSION_MAJOR) == 2 && (QNN_API_VERSION_MINOR < 26))
  if (enable_vtcm_backup_buffer_sharing_) {
    LOGS(logger_in_, WARNING) << "User specified enable_vtcm_backup_buffer_sharing but QNN API version is older than 2.26.";
  }
#endif

  // Device ID
  std::string device_id_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "device_id", "0", device_id_str);
  if (!device_id_str.empty()) {
    int value = std::stoi(device_id_str);
    if (value < 0) {
      LOGS(logger_in_, WARNING) << "Invalid device ID '" << value
                                << "', only >= 0 allowed. Set to " << device_id_ << ".";
    } else {
      device_id_ = static_cast<uint32_t>(value);
    }
  }

  // HTP architecture
  std::string htp_arch_str;
  QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "htp_arch", "", htp_arch_str);
  if (!htp_arch_str.empty()) {
    ParseHtpArchitecture(htp_arch_str, htp_arch, logger_in_);
  }

  // SoC model
  std::string soc_model_str;
  uint32_t soc_model = QNN_SOC_MODEL_UNKNOWN;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "soc_model", "0", soc_model_str);
  if (!soc_model_str.empty()) {
    int value = std::stoi(soc_model_str);
    if (value < 0) {
      LOGS(logger_in_, WARNING) << "Invalid SoC Model '" << value
                                << "', only >= 0 allowed. Set to " << soc_model << ".";
    } else {
      soc_model = static_cast<uint32_t>(value);
    }
  }

  // Op packages
  std::string op_packages_str;
  std::vector<onnxruntime::qnn::OpPackage> op_packages;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, "op_packages", "", op_packages_str);
  if (!op_packages_str.empty()) {
    ParseOpPackages(op_packages_str, op_packages, logger_in_);
  }

  // HTP FP16 precision mode
  std::string enable_htp_fp16_precision_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 "enable_htp_fp16_precision", "0",
                                 enable_htp_fp16_precision_str);
  if (enable_htp_fp16_precision_str == "1") {
    enable_HTP_FP16_precision_ = true;
  } else if (enable_htp_fp16_precision_str == "0") {
    enable_HTP_FP16_precision_ = false;
  } else {
    LOGS(logger_in_, VERBOSE) << "Invalid enable_htp_fp16_precision: "
                              << enable_HTP_FP16_precision_
                              << " only 0 or 1 allowed. Set to 0.";
  }
  LOGS(logger_in_, VERBOSE) << "User specified enable_htp_fp16_precision: " << enable_HTP_FP16_precision_;

  // Check for conflicts
  if (qnn_context_embed_mode_ && share_ep_contexts_) {
    LOGS(logger_in_, ERROR) << "[EP context generation:] Weight sharing enabled conflict with EP context embed mode. "
                            << "Inference will not work as expected!";
  }

  if (qnn_context_embed_mode_ && enable_vtcm_backup_buffer_sharing_) {
    LOGS(logger_in_, ERROR) << "[EP context generation:] VTCM backup buffer sharing enabled conflict with EP context embed mode. "
                            << "Inference will not work as expected!";
  }

  // HTP spill fill buffer
  std::string enable_htp_spill_fill_buffer_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 "enable_htp_spill_fill_buffer",
                                 "0",
                                 enable_htp_spill_fill_buffer_str);
  enable_spill_fill_buffer_ = enable_htp_spill_fill_buffer_str == "1";

  model_settings_.offload_graph_io_quantization = ParseBoolOption(ort_api,
                                                                  session_options_,
                                                                  "offload_graph_io_quantization",
                                                                  true,
                                                                  logger_in_);

  if (disable_cpu_ep_fallback_ && model_settings_.offload_graph_io_quantization) {
    LOGS(logger_in_, INFO) << "Fallback to CPU EP is disabled, but user tried to configure QNN EP to offload graph I/O "
                           << "quantization/dequantization to another EP. These are conflicting options. Fallback to CPU "
                           << "EP will remain disabled and graph I/O quantization/dequantization will not be offloaded "
                           << "to another EP.";
    model_settings_.offload_graph_io_quantization = false;
  }

  static const std::string QNN_HTP_SHARED_MEMORY_ALLOCATOR_ENABLED = "enable_htp_shared_memory_allocator";
  if (ParseBoolOption(ort_api, session_options_, QNN_HTP_SHARED_MEMORY_ALLOCATOR_ENABLED.c_str(), false, logger_in_)) {
    // Initialize rpcmem_library_.
    // This is necessary for HtpSharedMemoryAllocator to function and also indicates that the allocator is available.
    rpcmem_library_ = std::make_shared<qnn::RpcMemLibrary>();
    model_settings_.htp_shared_memory = true;
  }

  dump_json_qnn_graph_ = ParseBoolOption(ort_api, session_options_, "dump_json_qnn_graph", false, logger_in_);

  static const std::string QNN_GRAPH_DUMP_DIR = "json_qnn_graph_dir";
  std::string json_graph_dir_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, QNN_GRAPH_DUMP_DIR.c_str(), "", json_graph_dir_str);

  if (!json_graph_dir_str.empty()) {
    json_qnn_graph_dir_ = json_graph_dir_str;
    if (dump_json_qnn_graph_) {
      LOGS(logger_in_, INFO) << "JSON graphs directory: " << json_qnn_graph_dir_;
    } else {
      LOGS(logger_in_, WARNING) << "Provided a directory for dumping QNN JSON graphs, "
                                << "but did not enable dumping of QNN JSON graphs.";
    }
  }

  // For context binary generation with weight sharing enabled, use the QnnBackendManager from the shared context if it exits
  // So that all graphs from later sessions will be compiled into the same QNN context
  if (
      ((context_cache_enabled_ && share_ep_contexts_) || enable_vtcm_backup_buffer_sharing_) &&
      SharedContext::GetInstance().GetSharedQnnBackendManager()) {
    qnn_backend_manager_ = SharedContext::GetInstance().GetSharedQnnBackendManager();
    // Clear the QnnBackendManager from singleton to stop the resource share
    if (stop_share_ep_contexts_) {
      SharedContext::GetInstance().ResetSharedQnnBackendManager();
    }
  } else {
    qnn_backend_manager_ = qnn::QnnBackendManager::Create(
        qnn::QnnBackendManagerConfig{backend_path,
                                     profiling_level_etw,
                                     profiling_level,
                                     profiling_file_path,
                                     context_priority,
                                     std::move(qnn_serializer_config),
                                     device_id_,
                                     htp_arch,
                                     soc_model,
                                     op_packages},
        ApiPtrs{ort_api, ep_api, model_editor_api}, logger_in_);
    if (enable_vtcm_backup_buffer_sharing_) {
      SharedContext::GetInstance().SetSharedQnnBackendManager(qnn_backend_manager_);
    }
  }

#if defined(_WIN32)
  if (onnxruntime::logging::EtwRegistrationManager::SupportsETW()) {
    auto& etwRegistrationManager = logging::EtwRegistrationManager::Instance();
    // Register callback for ETW capture state (rundown)
    callback_ETWSink_provider_ = onnxruntime::logging::EtwRegistrationManager::EtwInternalCallback(
        [&etwRegistrationManager, this](
            LPCGUID SourceId,
            ULONG IsEnabled,
            UCHAR Level,
            ULONGLONG MatchAnyKeyword,
            ULONGLONG MatchAllKeyword,
            PEVENT_FILTER_DESCRIPTOR FilterData,
            PVOID CallbackContext) {
          ORT_UNUSED_PARAMETER(SourceId);
          ORT_UNUSED_PARAMETER(MatchAnyKeyword);
          ORT_UNUSED_PARAMETER(MatchAllKeyword);
          ORT_UNUSED_PARAMETER(FilterData);
          ORT_UNUSED_PARAMETER(CallbackContext);

          if (IsEnabled == EVENT_CONTROL_CODE_ENABLE_PROVIDER) {
            if ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Logs)) != 0) {
              auto ortETWSeverity = etwRegistrationManager.MapLevelToSeverity();
              (void)qnn_backend_manager_->ResetQnnLogLevel(ortETWSeverity);
            }
            if ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)) != 0) {
              if (Level != 0) {
                // Commenting out Dynamic QNN Profiling for now
                // There seems to be a crash in 3rd party QC QnnHtp.dll with this.
                // Repro Scenario - start ETW tracing prior to session creation.
                //    Then disable/enable ETW Tracing with the code below uncommented a few times
                // auto profiling_level_etw = GetProfilingLevelFromETWLevel(Level);
                // (void)qnn_backend_manager_->SetProfilingLevelETW(profiling_level_etw);
                //
                // NOTE(1/2/2025): It is possible that the above was not working in part because it is using the
                // *logging ETW* subsystem to modify profiling, which should use an entirely different
                // ETW provider (see QnnTelemetry). Should add callbacks for profiling to the QnnTelemetry ETW provider.
              }
            }
          }

          if (IsEnabled == EVENT_CONTROL_CODE_DISABLE_PROVIDER) {
            // (void)qnn_backend_manager_->SetProfilingLevelETW(qnn::ProfilingLevel::INVALID);
            (void)qnn_backend_manager_->ResetQnnLogLevel(std::nullopt);
          }
        });
    etwRegistrationManager.RegisterInternalCallback(callback_ETWSink_provider_);
  }
#endif
}

QnnEp::~QnnEp() {
  // Release any per-thread contexts that might be active
  ReleasePerThreadContext();

  // Explicitly clear the QNN models map to ensure proper cleanup
  if (!qnn_models_.empty()) {
    qnn_models_.clear();
  }

  // Clean up any thread context resources
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);
    context_state_.active_contexts.clear();
    context_state_.retired_context_pool.clear();

    // Remove this instance from all thread-local caches
    for (auto& weak_cache : context_state_.caches_to_update_on_destruction) {
      if (auto cache = weak_cache.lock()) {
        cache->erase(this);
      }
    }
    context_state_.caches_to_update_on_destruction.clear();
  }

#if defined(_WIN32)
  // Clean up ETW callback if registered
  if (callback_ETWSink_provider_) {
    if (onnxruntime::logging::EtwRegistrationManager::SupportsETW()) {
      auto& etwRegistrationManager = logging::EtwRegistrationManager::Instance();
      etwRegistrationManager.UnregisterInternalCallback(callback_ETWSink_provider_);
      callback_ETWSink_provider_ = nullptr;
    }
  }
#endif
}

const char* ORT_API_CALL QnnEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* qnn_ep = static_cast<const QnnEp*>(this_ptr);
  return qnn_ep->name_.c_str();
}

// Logs information about the supported/unsupported nodes.
static void LogNodeSupport(const logging::Logger& logger,
                           logging::Severity log_severity,
                           logging::DataType log_data_type,
                           const qnn::IQnnNodeGroup& qnn_node_group,
                           Status support_status) {
  if (!logger.OutputIsEnabled(log_severity, log_data_type)) {
    return;
  }

  size_t num_nodes = 0;
  std::ostringstream oss;
  for (const OrtNodeUnit* node_unit : qnn_node_group.GetNodeUnits()) {
    for (const OrtNode* node : node_unit->GetAllNodesInGroup()) {
      size_t node_id = node->GetId();
      const std::string& op_type = node->GetOpType();
      const std::string& name = node->GetName();

      oss << "\tOperator type: " << op_type
          << " Node name: " << name
          << " Node index: " << node_id << std::endl;
      num_nodes += 1;
    }
  }
  if (!support_status.IsOK()) {
    oss << "\tREASON : " << support_status.ErrorMessage() << std::endl;
  }

  LOGS(logger, INFO) << (support_status.IsOK() ? "Validation PASSED " : "Validation FAILED ")
                     << "for " << num_nodes
                     << " nodes in " << qnn_node_group.Type() << " (" << qnn_node_group.GetTargetNodeUnit()->OpType() << ") :"
                     << std::endl
                     << oss.str();
}

OrtStatus* QnnEp::GetSupportedNodes(OrtEp* this_ptr,
                                    const OrtGraph* graph,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const size_t node_unit_size,
                                    const logging::Logger& logger,
                                    std::vector<const OrtNode*>& supported_nodes) const {
  const QnnEp* ep = static_cast<const QnnEp*>(this_ptr);

  size_t num_graph_inputs = 0;
  size_t num_graph_outputs = 0;
  ep->ort_api.Graph_GetNumInputs(graph, &num_graph_inputs);
  ep->ort_api.Graph_GetNumOutputs(graph, &num_graph_outputs);

  std::vector<const OrtValueInfo*> graph_inputs(num_graph_inputs);
  std::vector<const OrtValueInfo*> graph_outputs(num_graph_outputs);
  ep->ort_api.Graph_GetInputs(graph, graph_inputs.data(), graph_inputs.size());
  ep->ort_api.Graph_GetOutputs(graph, graph_outputs.data(), graph_outputs.size());

  // Util function that initializes a table that maps a graph input or output name to its index.
  auto init_input_output_index_map = [&](std::unordered_map<std::string, size_t>& index_map,
                                         std::vector<const OrtValueInfo*> inouts) {
    size_t num_elements = inouts.size();
    for (size_t idx = 0; idx < num_elements; ++idx) {
      const OrtValueInfo* inout = inouts[idx];
      const char* name = nullptr;
      ep->ort_api.GetValueInfoName(inout, &name);

      index_map.emplace(name, idx);
    }
  };

  std::unordered_map<std::string, size_t> model_input_index_map;
  // TODO: Handle initializers as inputs.
  init_input_output_index_map(model_input_index_map, graph_inputs);

  std::unordered_map<std::string, size_t> model_output_index_map;
  init_input_output_index_map(model_output_index_map, graph_outputs);

  auto qnn_model_wrapper = qnn::QnnModelWrapper(*graph,
                                                ApiPtrs{ep->ort_api, ep->ep_api, ep->model_editor_api},
                                                logger,
                                                qnn_backend_manager_->GetQnnInterface(),
                                                qnn_backend_manager_->GetQnnBackendHandle(),
                                                model_input_index_map,
                                                model_output_index_map,
                                                qnn_backend_manager_->GetQnnBackendType(),
                                                model_settings_);

  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> qnn_node_groups;
  qnn_node_groups.reserve(node_unit_size);

  Status status = qnn::GetQnnNodeGroups(qnn_node_groups, qnn_model_wrapper, node_unit_map, node_unit_size, logger);
  if (!status.IsOK()) {
    return this->ort_api.CreateStatus(ORT_EP_FAIL, status.ErrorMessage().c_str());
  }

  for (const std::unique_ptr<qnn::IQnnNodeGroup>& qnn_node_group : qnn_node_groups) {
    const bool supported = qnn_node_group->IsSupported(qnn_model_wrapper, logger).IsOK();

    constexpr auto log_severity = logging::Severity::kINFO;
    constexpr auto log_data_type = logging::DataType::SYSTEM;
    if (logger.OutputIsEnabled(log_severity, log_data_type)) {
      LogNodeSupport(logger, log_severity, log_data_type, *qnn_node_group, status);
    }

    if (supported) {
      for (const OrtNodeUnit* node_unit : qnn_node_group->GetNodeUnits()) {
        for (const OrtNode* node : node_unit->GetAllNodesInGroup()) {
          supported_nodes.push_back(node);
        }
      }
    }
  }
  return nullptr;
}

void QnnEp::InitQnnHtpGraphConfigs(
    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const {
  if (qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::HTP) {
    if (htp_graph_finalization_opt_mode_ != qnn::HtpGraphFinalizationOptimizationMode::kDefault) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config = configs_builder.PushCustomConfig();
      htp_graph_opt_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
      htp_graph_opt_config->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
      htp_graph_opt_config->optimizationOption.floatValue = static_cast<float>(htp_graph_finalization_opt_mode_);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config = configs_builder.PushConfig();
      graph_opt_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config->customConfig = htp_graph_opt_config;
    }

    if (vtcm_size_in_mb_ > 0) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config_vtcm = configs_builder.PushCustomConfig();
      htp_graph_opt_config_vtcm->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
      htp_graph_opt_config_vtcm->vtcmSizeInMB = static_cast<uint32_t>(vtcm_size_in_mb_);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config_vtcm = configs_builder.PushConfig();
      graph_opt_config_vtcm->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config_vtcm->customConfig = htp_graph_opt_config_vtcm;
    }

    if (enable_HTP_FP16_precision_) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_precision_config = configs_builder.PushCustomConfig();
      htp_graph_precision_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      htp_graph_precision_config->precision = QNN_PRECISION_FLOAT16;

      gsl::not_null<QnnGraph_Config_t*> graph_precision_config = configs_builder.PushConfig();
      graph_precision_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_precision_config->customConfig = htp_graph_precision_config;
    }
  }
}

static bool EpSharedContextsHasAllGraphs(const OrtGraph* graph, const OrtApi& ort_api, const logging::Logger& logger) {
  size_t num_nodes = 0;
  if (ort_api.Graph_GetNumNodes(graph, &num_nodes) != nullptr) {
    return false;
  }

  std::vector<const OrtNode*> graph_nodes(num_nodes);
  if (ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size()) != nullptr) {
    return false;
  }

  bool all_graphs_found = true;

  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = graph_nodes[i];
    const char* op_type = nullptr;

    if (ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
      if (std::string(op_type) == "EPContext") {
        // Check the 'source' attribute to verify it's from QNN
        const OrtOpAttr* source_attr = nullptr;
        if (ort_api.Node_GetAttributeByName(node, "source", &source_attr) == nullptr && source_attr != nullptr) {
          char source_buffer[256] = {0};
          size_t source_len = 0;
          if (ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len) == nullptr) {
            std::string cache_source(source_buffer, source_len);

            // Convert to lowercase for comparison
            std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
                           [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

            if (cache_source == "qnnexecutionprovider" || cache_source == "qnn") {
              // Get the graph name (node name)
              const char* node_name = nullptr;
              if (ort_api.Node_GetName(node, &node_name) == nullptr && node_name != nullptr) {
                std::string graph_name(node_name);
                bool has_shared_qnn_model = SharedContext::GetInstance().HasQnnModel(graph_name);
                if (!has_shared_qnn_model) {
                  // Log the missing graph (equivalent to LOGS(logger, VERBOSE))
                  std::string log_message = "Graph: " + graph_name + " from EpContext node not found from shared EP contexts.";
                  LOGS(logger, VERBOSE) << log_message;
                  all_graphs_found = false;
                  break;
                }
              }
            }
          }
        }
      }
    }
  }
  return all_graphs_found;
}

static void GetMainEPCtxNodes(const OrtGraph* graph,
                              const OrtApi& ort_api,
                              std::unordered_set<const OrtNode*>& ep_context_nodes,
                              const logging::Logger& logger) {
  size_t num_nodes = 0;
  if (ort_api.Graph_GetNumNodes(graph, &num_nodes) != nullptr) {
    return;
  }

  std::vector<const OrtNode*> graph_nodes(num_nodes);
  if (ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size()) != nullptr) {
    return;
  }

  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = graph_nodes[i];
    const char* op_type = nullptr;

    if (ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
      if (std::string(op_type) == "EPContext") {
        // Check main_context attribute
        const OrtOpAttr* main_context_attr = nullptr;
        if (ort_api.Node_GetAttributeByName(node, "main_context", &main_context_attr) == nullptr && main_context_attr != nullptr) {
          int64_t is_main_context = 0;
          size_t out_size = 0;
          if (ort_api.ReadOpAttr(main_context_attr, ORT_OP_ATTR_INT, &is_main_context, sizeof(is_main_context), &out_size) == nullptr) {
            // Check source attribute
            const OrtOpAttr* source_attr = nullptr;
            if (ort_api.Node_GetAttributeByName(node, "source", &source_attr) == nullptr && source_attr != nullptr) {
              char source_buffer[256] = {0};
              size_t source_len = 0;
              if (ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len) == nullptr) {
                std::string cache_source(source_buffer, source_len);

                // Convert to lowercase for comparison
                std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
                               [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

                if (is_main_context && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
                  // Log the found EPContext node
                  const char* node_name = nullptr;
                  size_t node_id = 0;
                  ort_api.Node_GetName(node, &node_name);
                  ort_api.Node_GetId(node, &node_id);

                  std::string log_message = "EPContext Node found: [1] index: [" + std::to_string(node_id) +
                                            "] name: [" + (node_name ? node_name : "unknown") + "]";
                  LOGS(logger, VERBOSE) << log_message;
                  ep_context_nodes.insert(node);
                }
              }
            }
          }
        }
      }
    }
  }
}

void QnnEp::PartitionCtxModel(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) {
  // Get all nodes from the graph
  size_t num_nodes = 0;
  if (ort_api.Graph_GetNumNodes(graph, &num_nodes) != nullptr) {
    return;
  }

  std::vector<const OrtNode*> graph_nodes(num_nodes);
  if (ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size()) != nullptr) {
    return;
  }

  std::vector<const OrtNode*> supported_nodes;
  std::vector<std::vector<const OrtNode*>> supported_groups;

  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = graph_nodes[i];
    const char* op_type = nullptr;

    if (ort_api.Node_GetOperatorType(node, &op_type) && op_type != nullptr) {
      if (std::string(op_type) == "EPContext") {
        const OrtOpAttr* source_attr = nullptr;
        if (ort_api.Node_GetAttributeByName(node, "source", &source_attr) && source_attr != nullptr) {
          char source_buffer[256] = {0};
          size_t source_len = 0;
          if (ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len)) {
            std::string cache_source(source_buffer, source_len);

            std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
                           [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

            if (cache_source == "qnnexecutionprovider" || cache_source == "qnn") {
              const char* node_name = nullptr;
              size_t node_id = 0;
              ort_api.Node_GetName(node, &node_name);
              ort_api.Node_GetId(node, &node_id);

              std::string log_message = "Node supported: [1] index: [" + std::to_string(node_id) +
                                        "] name: [" + (node_name ? node_name : "unknown") +
                                        "] Operator type: [EPContext] index: [" + std::to_string(node_id) + "]";
              LOGS(logger_in_, VERBOSE) << log_message;

              supported_nodes.push_back(node);

              std::vector<const OrtNode*> supported_group{node};
              supported_groups.emplace_back(std::move(supported_group));
            }
          }
        }
      }
    }
  }

  for (const auto& supported_partition : supported_groups) {
    if (!supported_partition.empty()) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;

      ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                               supported_partition.data(),
                                               supported_partition.size(),
                                               &node_fusion_options);
    }
  }

  const size_t num_of_partitions = supported_groups.size();

  std::string summary_msg = "Number of cf supported by QNN EP: " + std::to_string(num_of_partitions) +
                            ", number of nodes in the graph: " + std::to_string(num_nodes) +
                            ", number of nodes supported by QNN: " + std::to_string(num_of_partitions);
  LOGS(logger_in_, INFO) << summary_msg;
}

static void GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
                                        const PathString& model_path_string,
                                        PathString& context_model_path) {
  // always try the path set by user first, it's the only way to set it if load model from memory
  if (!user_context_cache_path.empty()) {
    context_model_path = ToPathString(user_context_cache_path);
  } else if (!model_path_string.empty()) {  // model loaded from file
    context_model_path = model_path_string;
  }
}

OrtStatus* ORT_API_CALL QnnEp::GetCapabilityImpl(OrtEp* this_ptr,
                                                 const OrtGraph* graph,
                                                 OrtEpGraphSupportInfo* graph_support_info) noexcept {
  std::cout << "DEBUG: QnnEp::GetCapabilityImpl" << std::endl;
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  const OrtNode* parent_node = nullptr;
  RETURN_IF_ERROR(ep->ort_api.Graph_GetParentNode(graph, &parent_node));
  if (parent_node != nullptr) {
    return nullptr;
  }

  size_t num_nodes_in_graph = 0;
  RETURN_IF_ERROR(ep->ort_api.Graph_GetNumNodes(graph, &num_nodes_in_graph));
  if (num_nodes_in_graph == 0) {
    return nullptr;
  }

  bool is_qnn_ctx_model = qnn::GraphHasEpContextNode(graph, ep->ort_api);

  // auto gen_metadef_name = [ep, graph]() -> std::string {
  //     return ep->MakeMetadefName(graph);
  // };

  if (is_qnn_ctx_model && ep->config_.share_ep_contexts && SharedContext::GetInstance().HasSharedQnnModels()) {
    if (EpSharedContextsHasAllGraphs(graph, ep->ort_api, ep->logger_in_)) {
      ep->PartitionCtxModel(graph, graph_support_info);
      return nullptr;
    }
  }

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> context_bin_map;
  if (ep->enable_vtcm_backup_buffer_sharing_) {
    std::unordered_set<const OrtNode*> ep_ctx_nodes;
    GetMainEPCtxNodes(graph, ep->ort_api, ep_ctx_nodes, ep->logger_in_);

    PathString context_model_path;
    GetContextOnnxModelFilePath(ep->context_cache_path_cfg_, ToPathString(""), context_model_path);

    std::filesystem::path parent_path = std::filesystem::path(context_model_path).parent_path();

    for (auto& ep_ctx_node : ep_ctx_nodes) {
      // Get the ep_cache_context attribute from the node
      const OrtOpAttr* ep_cache_context_attr = nullptr;
      if (ep->ort_api.Node_GetAttributeByName(ep_ctx_node, "ep_cache_context", &ep_cache_context_attr) == nullptr && ep_cache_context_attr != nullptr) {
        char context_buffer[512] = {0};
        size_t context_len = 0;
        if (ep->ort_api.ReadOpAttr(ep_cache_context_attr, ORT_OP_ATTR_STRING, context_buffer, sizeof(context_buffer) - 1, &context_len) == nullptr) {
          std::string context_bin_filepath(parent_path.string());
          context_bin_filepath.append("/").append(std::string(context_buffer, context_len));

          if (context_bin_map.find(context_bin_filepath) == context_bin_map.end()) {
            context_bin_map.emplace(context_bin_filepath, std::make_unique<std::vector<std::string>>());
            // Push context bin filepath for lookup between sessions
            context_bin_map.at(context_bin_filepath)->push_back(context_bin_filepath);
          }

          // Add the node name to the context bin map
          const char* node_name = nullptr;
          if (ep->ort_api.Node_GetName(ep_ctx_node, &node_name) == nullptr && node_name != nullptr) {
            context_bin_map.at(context_bin_filepath)->push_back(std::string(node_name));
          }
        }
      }
    }
  }

  Status rt = ep->qnn_backend_manager_->SetupBackend(is_qnn_ctx_model,
                                                     ep->context_cache_enabled_ && false,  // enable_spill_fill_buffer_ (not implemented)
                                                     ep->share_ep_contexts_,
                                                     ep->enable_vtcm_backup_buffer_sharing_,
                                                     context_bin_map);

  context_bin_map.clear();

  if (Status::OK() != rt) {
    LOGS(ep->logger_in_, ERROR) << "QNN SetupBackend failed " << rt.ErrorMessage();
    return nullptr;
  }

  if (qnn::IsNpuBackend(ep->qnn_backend_manager_->GetQnnBackendType())) {
    // Set the power config id and the default power mode from provider option for main thread,
    // otherwise it will mess up the power mode if user just create session without run it.
    ep->GetPerThreadContext();
  }

  // Report error if QNN CPU backend is loaded while CPU fallback is disabled
  if (ep->config_.disable_cpu_ep_fallback && ep->qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::CPU) {
    LOGS(ep->logger_in_, ERROR) << "Qnn CPU backend is loaded while CPU fallback is disabled.";
    return nullptr;
  }

  if ((ep->context_cache_enabled_ || is_qnn_ctx_model) && !qnn::IsQpuBackend(ep->qnn_backend_manager_->GetQnnBackendType())) {
    LOGS(ep->logger_in_, ERROR) << "Qnn context cache only works for HTP/DSP/GPU backend.";
    return nullptr;
  }

  if (is_qnn_ctx_model) {
    ep->PartitionCtxModel(graph, graph_support_info);
    return nullptr;
  }

  // Get node units for the ABI layer
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;

  std::tie(node_unit_holder, node_unit_map) = GetAllOrtNodeUnits(ep->ort_api, graph, ep->logger_in_);
  std::cout << "DEBUG: #nodes: " << node_unit_holder.size() << std::endl;

  // Analyze nodes for QNN support
  std::vector<const OrtNode*> supported_nodes;
  ep->GetSupportedNodes(this_ptr, graph, node_unit_map, node_unit_holder.size(), ep->logger_in_, supported_nodes);

  // Helper function that returns a string that lists all unsupported nodes.
  // Ex: { name: mul_123, type: Mul }, {}, ...
  auto get_unsupported_node_names = [&node_unit_holder, &supported_nodes]() -> std::string {
    std::stringstream ss;
    const size_t num_node_units = node_unit_holder.size();

    for (size_t i = 0; i < num_node_units; ++i) {
      const auto& node_unit = node_unit_holder[i];

      auto it = std::find(supported_nodes.begin(), supported_nodes.end(), &node_unit->GetNode());
      if (it == supported_nodes.end()) {
        ss << "{ name: " << node_unit->Name() << ", type: " << node_unit->OpType() << " }";
        if (i == num_node_units - 1) {
          ss << ", ";
        }
      }
    }

    return ss.str();
  };

  std::cout << "DEBUG: #supported nodes " << supported_nodes.size() << std::endl;

  // If no supported nodes, return empty
  if (supported_nodes.empty()) {
    return nullptr;
  }

  size_t num_of_supported_nodes = supported_nodes.size();

  // Print list of unsupported nodes to the ERROR logger if the CPU EP
  // has been disabled for this inference session.
  if (!is_qnn_ctx_model && ep->disable_cpu_ep_fallback_ && num_nodes_in_graph != num_of_supported_nodes) {
    LOGS(ep->logger_in_, ERROR) << "Unsupported nodes in QNN EP: " << get_unsupported_node_names();
  }

  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;
  node_fusion_options.drop_constant_initializers = true;
  RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                               supported_nodes.data(),
                                                               supported_nodes.size(),
                                                               &node_fusion_options));

  return nullptr;
}

OrtStatus* QnnEp::CompileContextModel(const OrtGraph** graphs,
                                      const OrtNode** fused_nodes,
                                      size_t count,
                                      OrtNodeComputeInfo** node_compute_infos) {
  // Collect graph and fused nodes names.
  std::vector<std::pair<std::string, std::string>> names;
  names.reserve(count);

  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    const char* graph_name = nullptr;
    ort_api.Node_GetName(fused_nodes[graph_idx], &graph_name);

    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(graphs[graph_idx], &num_nodes);

    std::vector<const OrtNode*> graph_nodes(num_nodes);
    ort_api.Graph_GetNodes(graphs[graph_idx], graph_nodes.data(), graph_nodes.size());

    const OrtNode* ep_context_node = graph_nodes[0];
    const char* ep_context_node_name = nullptr;
    ort_api.Node_GetName(ep_context_node, &ep_context_node_name);

    names.push_back(std::pair<std::string, std::string>(graph_name, ep_context_node_name));
  }

  // Get QnnModel from EP shared contexts
  if (share_ep_contexts_ && SharedContext::GetInstance().HasSharedQnnModels()) {
    bool has_all_graphs = true;
    for (const auto& name_pair : names) {
      if (!SharedContext::GetInstance().HasQnnModel(name_pair.second)) {
        has_all_graphs = false;
        LOGS(logger_in_, VERBOSE) << "Graph: "
                                  << name_pair.second
                                  << " from EpContext node not found from shared EP contexts.";
        break;
      }
    }

    if (has_all_graphs) {
      for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
        auto qnn_model_shared = SharedContext::GetInstance().GetSharedQnnModel(names[graph_idx].second);
        RETURN_IF(qnn_model_shared == nullptr,
                  ort_api,
                  ("Graph: " + names[graph_idx].second + " not found from shared EP contexts.").c_str());

        RETURN_IF_NOT_OK(qnn_model_shared->SetGraphInputOutputInfo(*graphs[graph_idx],
                                                                   *fused_nodes[graph_idx],
                                                                   logger_in_),
                         ort_api);
        RETURN_IF_NOT_OK(qnn_model_shared->SetupQnnInputOutput(logger_in_), ort_api);
        qnn_models_.emplace(names[graph_idx].first, std::move(qnn_model_shared));

        auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*this);
        node_compute_infos[graph_idx] = node_compute_info.release();

        return nullptr;
      }
    }
  }

  // Table<EPContext node name, QnnModel>, the node name is the graph_meta_id (old) created from user model which used
  // to generate the EP context model for this session (created from an EP context model), the graph_meta_id is new
  qnn::QnnModelLookupTable qnn_models;

  std::vector<int> main_context_pos_list;
  RETURN_IF_NOT_OK(qnn::GetMainContextNode(graphs, count, ort_api, main_context_pos_list), ort_api);
  uint32_t total_context_size = SafeInt<uint32_t>(main_context_pos_list.size());

  int64_t max_spill_fill_size = 0;

  // Adjust the main_context_pos_list, move the one with max spill fill buffer to the beginning
  // HTP spill fill buffer only works for multiple QNN contexts generated after QNN v2.28
  if (total_context_size > 1) {
    RETURN_IF_NOT_OK(qnn::TryGetMaxSpillFillSize(graphs,
                                                 ort_api,
                                                 total_context_size,
                                                 max_spill_fill_size,
                                                 main_context_pos_list),
                     ort_api);
  }

  // Figure out the EP context model path from session option
  PathString context_model_path;
  GetContextOnnxModelFilePath(context_cache_path_cfg_, ToPathString(""), context_model_path);

  for (auto main_context_pos : main_context_pos_list) {
    // Create QNN context from the cached binary, deserialize the QNN graph from the binary
    RETURN_IF_NOT_OK(qnn::LoadQnnCtxFromOnnxGraph(graphs[main_context_pos],
                                                  ort_api,
                                                  context_model_path,
                                                  qnn_backend_manager_.get(),
                                                  qnn_models,
                                                  logger_in_,
                                                  max_spill_fill_size),
                     ort_api);
  }

  std::string graph_name;
  std::string ep_context_node_name;
  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    std::tie(graph_name, ep_context_node_name) = names[graph_idx];

    RETURN_IF(qnn_models.find(ep_context_node_name) == qnn_models.end(),
              ort_api,
              (ep_context_node_name + " context node name not exists in table qnn_models.").c_str());
    auto qnn_model = std::move(qnn_models[ep_context_node_name]);
    RETURN_IF_NOT_OK(qnn_model->SetGraphInputOutputInfo(*graphs[graph_idx], *fused_nodes[graph_idx], logger_in_), ort_api);
    RETURN_IF_NOT_OK(qnn_model->SetupQnnInputOutput(logger_in_), ort_api);

    // fused node name is QNNExecutionProvider_QNN_[hash_id]_[id]
    // the name here must be same with context->node_name in compute_info
    qnn_models_.emplace(graph_name, std::move(qnn_model));
    qnn_models.erase(ep_context_node_name);

    auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*this);
    node_compute_infos[graph_idx] = node_compute_info.release();
  }

  if (share_ep_contexts_ && qnn_models.size() > 0) {
    std::vector<std::unique_ptr<qnn::QnnModel>> shared_qnn_models;
    for (auto& [key, value] : qnn_models) {
      shared_qnn_models.push_back(std::move(qnn_models[key]));
    }
    std::string duplicate_graph_names;
    bool has_duplicate_graph = SharedContext::GetInstance().SetSharedQnnModel(std::move(shared_qnn_models),
                                                                              duplicate_graph_names);
    RETURN_IF(has_duplicate_graph,
              ort_api,
              ("Duplicate graph names detect across sessions: " + duplicate_graph_names).c_str());
  }

  return nullptr;
}

OrtStatus* QnnEp::CreateEPContextNodes(const OrtNode** fused_nodes, size_t count, OrtNode** ep_context_nodes) {
  // All partitioned graph share single QNN context, included in the same context binary
  uint64_t buffer_size(0);
  auto context_buffer = qnn_backend_manager_->GetContextBinaryBuffer(buffer_size);
  // Get max spill fill buffer size
  uint64_t max_spill_fill_buffer_size = 0;
  if (enable_spill_fill_buffer_) {
    RETURN_IF_NOT_OK(qnn_backend_manager_->GetMaxSpillFillBufferSize(context_buffer.get(),
                                                                     buffer_size,
                                                                     max_spill_fill_buffer_size),
                     ort_api);
  }

  // Figure out the EP context model path from session option
  PathString context_model_path;
  GetContextOnnxModelFilePath(context_cache_path_cfg_, ToPathString(""), context_model_path);

  RETURN_IF_NOT_OK(qnn::CreateEPContextNodes(fused_nodes,
                                             count,
                                             ep_context_nodes,
                                             ort_api,
                                             model_editor_api,
                                             context_buffer.get(),
                                             buffer_size,
                                             qnn_backend_manager_->GetSdkVersion(),
                                             qnn_models_,
                                             context_model_path,
                                             qnn_context_embed_mode_,
                                             max_spill_fill_buffer_size,
                                             *(logger_.ToInternal()),
                                             share_ep_contexts_,
                                             stop_share_ep_contexts_),
                   ort_api);

  if (share_ep_contexts_ &&
      !stop_share_ep_contexts_ &&
      nullptr == SharedContext::GetInstance().GetSharedQnnBackendManager()) {
    RETURN_IF(SharedContext::GetInstance().SetSharedQnnBackendManager(qnn_backend_manager_),
              ort_api,
              "Failed to set shared QnnBackendManager.");
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::CompileImpl(_In_ OrtEp* this_ptr,
                                           _In_ const OrtGraph** graphs,
                                           _In_ const OrtNode** fused_nodes,
                                           _In_ size_t count,
                                           _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                           _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  std::cout << "DEBUG: QnnEp::CompileImpl" << std::endl;
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  if (qnn::IsOrtGraphHasCtxNode(graphs, count, ep->ort_api)) {
    return ep->CompileContextModel(graphs, fused_nodes, count, node_compute_infos);
  }

  ep_context_nodes;

  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    const OrtGraph* graph = graphs[graph_idx];
    const OrtNode* fused_node = fused_nodes[graph_idx];

    const char* name = nullptr;
    ep->ort_api.Node_GetName(fused_node, &name);
    const std::string fused_node_name{name};

    std::unique_ptr<qnn::QnnModel> qnn_model = std::make_unique<qnn::QnnModel>(
        ep->qnn_backend_manager_.get(), ApiPtrs{ep->ort_api, ep->ep_api, ep->model_editor_api});

    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t> htp_graph_configs_builder(
        QNN_GRAPH_CONFIG_INIT, QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
    ep->InitQnnHtpGraphConfigs(htp_graph_configs_builder);

    std::vector<const QnnGraph_Config_t*> all_graph_configs;
    const QnnGraph_Config_t** htp_configs = htp_graph_configs_builder.GetQnnConfigs();
    if (htp_configs) {
      // Reserve enough for configs + nullptr
      all_graph_configs.reserve(htp_graph_configs_builder.GetSize() + 1);
      for (const QnnGraph_Config_t** config = htp_configs; *config; ++config) {
        all_graph_configs.push_back(*config);
      }
    }

    qnn::QnnSerializerConfig* qnn_serializer_config = ep->qnn_backend_manager_->GetQnnSerializerConfig();
    if (qnn_serializer_config) {
      // We don't bother reserving here to keep the API simpler. Also note that if we're here,
      // we're likely debugging and not waiting for inference.
      qnn_serializer_config->SetGraphName(fused_node_name);
      const QnnGraph_Config_t** serializer_configs = qnn_serializer_config->Configure();
      if (serializer_configs) {
        for (const QnnGraph_Config_t** config = serializer_configs; *config; ++config) {
          all_graph_configs.push_back(*config);
        }
      }
    }

    const QnnGraph_Config_t** all_graph_configs_ptr = nullptr;
    if (!all_graph_configs.empty()) {
      all_graph_configs.push_back(nullptr);
      all_graph_configs_ptr = all_graph_configs.data();
    }

    std::string json_graph_filepath;

    if (ep->dump_json_qnn_graph_) {
      namespace fs = std::filesystem;
      fs::path path = fs::path(ep->json_qnn_graph_dir_) / fs::path(fused_node_name + ".json");
      json_graph_filepath = path.string();
    }

    RETURN_IF_NOT_OK(qnn_model->ComposeGraph(*graph,
                                             *fused_node,
                                             ep->model_settings_,
                                             ep->logger_in_,
                                             all_graph_configs_ptr,
                                             json_graph_filepath),
                     ep->ort_api);
    RETURN_IF_NOT_OK(qnn_model->FinalizeGraphs(ep->logger_in_), ep->ort_api);
    RETURN_IF_NOT_OK(qnn_model->SetupQnnInputOutput(ep->logger_in_), ep->ort_api);

    ep->qnn_models_.emplace(fused_node_name, std::move(qnn_model));

    auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*ep);
    node_compute_infos[graph_idx] = node_compute_info.release();
  }

  if (ep->context_cache_enabled_) {
    RETURN_IF_ERROR(ep->CreateEPContextNodes(fused_nodes, count, ep_context_nodes));
  }

  std::cout << "DEBUG: QnnEp::CompileImpl complete" << std::endl;
  return nullptr;
}

void ORT_API_CALL QnnEp::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                     OrtNodeComputeInfo** node_compute_infos,
                                                     size_t num_node_compute_infos) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);
  for (size_t idx = 0; idx < num_node_compute_infos; ++idx) {
    delete node_compute_infos[idx];
  }
}

OrtStatus* ORT_API_CALL QnnEp::GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                          _Out_ OrtEpDataLayout* preferred_data_layout) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);
  *preferred_data_layout = OrtEpDataLayout::OrtEpDataLayout_NHWC;
  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                                _In_z_ const char* domain,
                                                                _In_z_ const char* op_type,
                                                                _In_ OrtEpDataLayout target_data_layout,
                                                                _Outptr_ int* should_convert) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);
  ORT_UNUSED_PARAMETER(target_data_layout);

  *should_convert = -1;

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "Upsample") {
    // Upsample is translated to QNN's Resize, which requires the NHWC layout for processing.
    *should_convert = 1;
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::OnRunStartImpl(_In_ OrtEp* this_ptr, _In_ const OrtRunOptions* run_options) noexcept {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  auto backend_type = ep->qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return nullptr;
  }

  const char* htp_perf_mode = nullptr;
  htp_perf_mode = ep->ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnPerfMode);
  qnn::HtpPerformanceMode htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  if (htp_perf_mode != nullptr) {
    ParseHtpPerformanceMode(htp_perf_mode, htp_performance_mode, ep->logger_in_);
  }

  const char* rpc_latency = nullptr;
  rpc_latency = ep->ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnRpcControlLatency);
  uint32_t rpc_control_latency = 0;
  if (rpc_latency != nullptr) {
    rpc_control_latency = static_cast<uint32_t>(std::stoul(rpc_latency));
    LOGS(ep->logger_in_, VERBOSE) << "rpc_control_latency: " << rpc_control_latency;
  }

  if (ep->GetPerThreadContext().IsHtpPowerConfigIdValid()) {
    if (qnn::HtpPerformanceMode::kHtpDefault != htp_performance_mode) {
      RETURN_IF_NOT_OK(ep->qnn_backend_manager_->SetHtpPowerConfig(ep->GetPerThreadContext().GetHtpPowerConfigId(),
                                                                   htp_performance_mode),
                       ep->ort_api);
    }

    if (rpc_control_latency > 0) {
      RETURN_IF_NOT_OK(ep->qnn_backend_manager_->SetRpcControlLatency(ep->GetPerThreadContext().GetHtpPowerConfigId(),
                                                                      rpc_control_latency),
                       ep->ort_api);
    }
  }

  const char* lora_config = nullptr;
  lora_config = ep->ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnLoraConfig);
  if (lora_config != nullptr) {
    LOGS(ep->logger_in_, VERBOSE) << "lora_config: " << lora_config;
    RETURN_IF_NOT_OK(ep->qnn_backend_manager_->ParseLoraConfig(lora_config), ep->ort_api);
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::OnRunEndImpl(_In_ OrtEp* this_ptr,
                                            _In_ const OrtRunOptions* run_options,
                                            _In_ bool sync_stream) noexcept {
  ORT_UNUSED_PARAMETER(sync_stream);

  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  auto backend_type = ep->qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return nullptr;
  }

  const char* htp_perf_mode = nullptr;
  htp_perf_mode = ep->ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnPerfMode);
  qnn::HtpPerformanceMode htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  if (htp_perf_mode != nullptr) {
    ParseHtpPerformanceMode(htp_perf_mode, htp_performance_mode, ep->logger_in_);
  }

  if (ep->GetPerThreadContext().IsHtpPowerConfigIdValid()) {
    if (qnn::HtpPerformanceMode::kHtpDefault != htp_performance_mode) {
      RETURN_IF_NOT_OK(ep->qnn_backend_manager_->SetHtpPowerConfig(ep->GetPerThreadContext().GetHtpPowerConfigId(),
                                                                   htp_performance_mode),
                       ep->ort_api);
    }
  }

  return nullptr;
}

QnnEp::PerThreadContext::PerThreadContext(qnn::QnnBackendManager* qnn_backend_manager,
                                          uint32_t device_id,
                                          uint32_t core_id,
                                          qnn::HtpPerformanceMode default_htp_performance_mode,
                                          uint32_t default_rpc_control_latency)
    : qnn_backend_manager_(qnn_backend_manager) {
  Status rt = qnn_backend_manager_->CreateHtpPowerCfgId(device_id, core_id, htp_power_config_id_);
  is_htp_power_config_id_valid_ = rt.IsOK();

  // Set default performance mode and latency for each thread as default
  // so user doesn't need to set it for every session run
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

QnnEp::PerThreadContext::~PerThreadContext() {
  if (is_htp_power_config_id_valid_) {
    ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->DestroyHTPPowerConfigID(htp_power_config_id_));
  }
}

QnnEp::PerThreadContext& QnnEp::GetPerThreadContext() {
  const auto& per_thread_context_cache = PerThreadContextCache();

  // Try to use cached context
  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    if (cached_context) {
      return *cached_context;
    }
  }

  // Get context and update cache
  std::shared_ptr<PerThreadContext> context;
  {
    std::lock_guard<std::mutex> lock(context_state_.mutex);

    // Get or create a context
    if (context_state_.retired_context_pool.empty()) {
      uint32_t core_id = 0;
      context = std::make_shared<PerThreadContext>(qnn_backend_manager_.get(), device_id_, core_id,
                                                   default_htp_performance_mode_, default_rpc_control_latency_);
    } else {
      context = context_state_.retired_context_pool.back();
      context_state_.retired_context_pool.pop_back();
    }

    // Insert into active_contexts
    context_state_.active_contexts.insert(context);

    // Insert into caches_to_update_on_destruction
    context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache);
  }

  per_thread_context_cache->insert(std::make_pair(this, context));

  return *context;
}

void QnnEp::ReleasePerThreadContext() {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    auto cached_context = cached_context_it->second.lock();
    if (cached_context) {
      {
        std::lock_guard<std::mutex> lock(context_state_.mutex);
        context_state_.active_contexts.erase(cached_context);
        context_state_.retired_context_pool.push_back(cached_context);
      }

      per_thread_context_cache->erase(cached_context_it);
    }
  }
}

QnnEp::QnnNodeComputeInfo::QnnNodeComputeInfo(QnnEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* QnnEp::QnnNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                      OrtNodeComputeContext* compute_context,
                                                      void** compute_state) {
  auto* node_compute_info = static_cast<QnnNodeComputeInfo*>(this_ptr);
  QnnEp& ep = node_compute_info->ep;

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto qnn_model_it = ep.qnn_models_.find(fused_node_name);
  if (qnn_model_it == ep.qnn_models_.end()) {
    std::string message = "Unable to get QnnModel with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  *compute_state = qnn_model_it->second.get();
  return nullptr;
}

OrtStatus* QnnEp::QnnNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                                  void* compute_state,
                                                  OrtKernelContext* kernel_context) {
  auto* node_compute_info = static_cast<QnnNodeComputeInfo*>(this_ptr);
  QnnEp& ep = node_compute_info->ep;

  qnn::QnnModel* model = reinterpret_cast<qnn::QnnModel*>(compute_state);
  Status status = model->ExecuteGraph(kernel_context, ep.logger_in_);
  if (!status.IsOK()) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, status.ErrorMessage().c_str());
  }

  return nullptr;
}

void QnnEp::QnnNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  // The 'state' is a qnn::QnnModel managed by unique_ptr.
  ORT_UNUSED_PARAMETER(this_ptr);
  ORT_UNUSED_PARAMETER(compute_state);
}

}  // namespace onnxruntime
