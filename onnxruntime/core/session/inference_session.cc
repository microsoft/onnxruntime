// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"

#include <memory>
#include <sstream>
#include <list>
#include <string>
#include <thread>
#include <queue>

#include "core/common/denormal.h"
#include "core/common/logging/logging.h"
#include "core/common/parse_string.h"
#include "core/common/path_string.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/ort_format_version.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_frame.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/kernel_type_str_resolver_utils.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/transform_layout_functions.h"
#include "core/framework/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/layout_transformation/layout_transformation.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/qdq_transformer/ensure_unique_dq_for_node_unit.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/selectors_actions/selector_action_transformer_apply_contexts.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"
#include "core/platform/Barrier.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#ifdef USE_DML  // TODO: This is necessary for the workaround in TransformGraph
#include "core/providers/dml/DmlExecutionProvider/src/DmlGraphFusionTransformer.h"
#include "core/providers/dml/DmlExecutionProvider/src/GraphTransformer.h"
#include "core/providers/dml/dml_session_options_config_keys.h"
#endif
#include "core/session/environment.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/util/protobuf_parsing_utils.h"
#include "core/util/thread_utils.h"

// custom ops are not available in a minimal build unless ORT_MINIMAL_BUILD_CUSTOM_OPS is set
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#include "core/framework/customregistry.h"
#include "core/session/custom_ops.h"
#endif
#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#include "core/framework/stream_execution_context.h"
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace {
template <typename T>
const T* GetDateFormatString();

template <>
inline const char* GetDateFormatString<char>() {
  return "%Y-%m-%d_%H-%M-%S";
}
#ifdef _WIN32
template <>
inline const wchar_t* GetDateFormatString<wchar_t>() {
  return L"%Y-%m-%d_%H-%M-%S";
}
#endif
// TODO: use LoggingManager::GetTimestamp and date::operator<<
// (see ostream_sink.cc for an example)
// to simplify this and match the log file timestamp format.
template <typename T>
inline std::basic_string<T> GetCurrentTimeString() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm;  // NOLINT

#ifdef _WIN32
  ORT_ENFORCE(localtime_s(&local_tm, &in_time_t) == 0);
#else
  localtime_r(&in_time_t, &local_tm);
#endif

  T time_str[32];
  OrtStrftime<T>(time_str, sizeof(time_str), GetDateFormatString<T>(), &local_tm);
  return std::basic_string<T>(time_str);
}

#if !defined(ORT_MINIMAL_BUILD)

static bool HasControlflowNodes(const Graph& graph) {
  for (const auto& node : graph.Nodes()) {
    if (node.ContainsSubgraph()) {
      return true;
    }
  }

  return false;
}

static bool HasMemcpyNodes(const Graph& graph) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MemcpyFromHost" || node.OpType() == "MemcpyToHost") {
      return true;
    }
  }

  return false;
}

static bool AreAllComputeNodesAssignedToCudaEp(const Graph& graph) {
  bool nodes_on_cpu_and_cuda_eps_only = true;

  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();

    // Empty node provider means CPU EP
    if (!node_provider.empty() &&
        node_provider != kCudaExecutionProvider &&
        node_provider != kCpuExecutionProvider) {
      nodes_on_cpu_and_cuda_eps_only = false;
      break;
    }
  }

  // If we see nodes assigned to EPs other than CPU or CUDA
  // (or) if there are Memcpy nodes, then all compute nodes have
  // not been parititoned to the CUDA EP.
  // We allow CPU EPs to show up in the EP list as long as thre is no Memcpy
  // involved as shape subgraphs will be forced onto CPU and these will not have
  // Memcpy nodes involved.
  return nodes_on_cpu_and_cuda_eps_only && !HasMemcpyNodes(graph);
}

static bool AreAllNodesInMainGraphAssignedToOneEp(const Graph& graph, ProviderType provider) {
  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();

    if (node_provider.empty() || node_provider != provider) {
      return false;
    }
  }

  return true;
}

static bool HasShapeSubgraphNodes(const Graph& graph) {
  bool has_shape_nodes = false;
  bool has_cpu_ep_nodes = false;

  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Shape") {
      has_shape_nodes = true;
      break;
    }
  }

  for (const auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();

    if (node_provider.empty() || node_provider == kCpuExecutionProvider) {
      has_cpu_ep_nodes = true;
      break;
    }
  }

  return has_shape_nodes && has_cpu_ep_nodes;
}

Status GetMinimalBuildOptimizationHandling(
    std::string_view config_value, bool saving_ort_format,
    InferenceSession::MinimalBuildOptimizationHandling& minimal_build_optimization_handling) {
  if (config_value == "save") {
    if (saving_ort_format) {
      minimal_build_optimization_handling =
          InferenceSession::MinimalBuildOptimizationHandling::SaveMinimalBuildRuntimeOptimizations;
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           kOrtSessionOptionsConfigMinimalBuildOptimizations,
                           " value of 'save' is only valid when saving an ORT format model.");
  }

  if (config_value == "apply") {
    minimal_build_optimization_handling =
        InferenceSession::MinimalBuildOptimizationHandling::OnlyApplyMinimalBuildOptimizations;
    return Status::OK();
  }

  if (config_value.empty()) {
    minimal_build_optimization_handling =
        InferenceSession::MinimalBuildOptimizationHandling::ApplyFullBuildOptimizations;
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Invalid value for ", kOrtSessionOptionsConfigMinimalBuildOptimizations, ": ", config_value);
};

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace

std::atomic<uint32_t> InferenceSession::global_session_id_{1};

static Status FinalizeSessionOptions(const SessionOptions& user_provided_session_options,
                                     const ONNX_NAMESPACE::ModelProto& model_proto,
                                     bool is_model_proto_parsed,
                                     /*out*/ SessionOptions& finalized_session_options) {
#if !defined(ORT_MINIMAL_BUILD)
  const logging::Logger& default_logger = logging::LoggingManager::DefaultLogger();

  // By now the environment should have initialized. (It is enforced prior to this.)
  const Env& env_instance = Env::Default();

  bool session_options_from_model = false;

  // Get the value held by the environment variable - kOrtLoadConfigFromModelEnvVar
  const std::string load_config_from_model_env_var_value =
      env_instance.GetEnvironmentVar(inference_session_utils::kOrtLoadConfigFromModelEnvVar);

  // Ascertain if the model is to be read for the ORT config from the afore parsed env var
  if (!load_config_from_model_env_var_value.empty()) {
    // Check if the env var contains an unsupported value
    if (load_config_from_model_env_var_value.length() > 1 ||
        (load_config_from_model_env_var_value[0] != '0' && load_config_from_model_env_var_value[0] != '1')) {
      std::ostringstream oss;
      oss << "The only supported values for the environment variable "
          << inference_session_utils::kOrtLoadConfigFromModelEnvVar << " are '0' and '1'. "
          << "The environment variable contained the value: " << load_config_from_model_env_var_value;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, oss.str());
    }

    if (load_config_from_model_env_var_value[0] == '1') {
      LOGS(default_logger, INFO) << "Reading the provided model for the ORT config";
      session_options_from_model = true;
    }
  }

  // The model is to be read for an ORT config json that may hold some/all session options
  if (session_options_from_model) {
    SessionOptions constructed_session_options;

    // In theory we should not hit this condition unless this internal class' APIs are being called incorrectly.
    // This is a good sanity check to enforce that the model has been parsed prior to looking into it for ort config.
    ORT_ENFORCE(is_model_proto_parsed, "ModelProto needs to be parsed to check for ORT config within it");

    // Use default logger as the session_logger_ hasn't been initialized yet.
    inference_session_utils::JsonConfigParser config_parser(default_logger);

    auto status = config_parser.ParseOrtConfigJsonInModelProto(model_proto);
    if (!status.IsOK()) {
      return status;
    }

    status = config_parser.ParseSessionOptionsFromModelProto(constructed_session_options);
    if (!status.IsOK()) {
      return status;
    }

    // use the constructed session options
    finalized_session_options = constructed_session_options;
  } else {
    // use user provided session options instance
    finalized_session_options = user_provided_session_options;
  }
#else
  ORT_UNUSED_PARAMETER(model_proto);
  ORT_UNUSED_PARAMETER(is_model_proto_parsed);
  finalized_session_options = user_provided_session_options;
#endif  // !defined(ORT_MINIMAL_BUILD)

  return Status::OK();
}

void InferenceSession::ConstructorCommon(const SessionOptions& session_options,
                                         const Environment& session_env) {
  auto status = FinalizeSessionOptions(session_options, model_proto_, is_model_proto_parsed_, session_options_);
  // a monotonically increasing session id for use in telemetry
  session_id_ = global_session_id_.fetch_add(1);
  ORT_ENFORCE(status.IsOK(), "Could not finalize session options while constructing the inference session. Error Message: ",
              status.ErrorMessage());

  // The call to InitLogger depends on the final state of session_options_. Hence it should be invoked
  // after the invocation of FinalizeSessionOptions.
  InitLogger(logging_manager_);  // this sets session_logger_ so that it can be used for logging after this point.

#if !defined(ORT_MINIMAL_BUILD)
  // Update the number of steps for the graph transformer manager using the "finalized" session options
  ORT_ENFORCE(graph_transformer_mgr_.SetSteps(session_options_.max_num_graph_transformation_steps).IsOK());
#endif

  bool set_denormal_as_zero =
      session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigSetDenormalAsZero, "0") == "1";

  // The only first session option for flush-to-zero and denormal-as-zero is effective to main thread and OpenMP threads.
  {
    static std::once_flag once;

    std::call_once(once, [&] {
      SetDenormalAsZero(set_denormal_as_zero);

      LOGS(*session_logger_, INFO) << "Flush-to-zero and denormal-as-zero are " << ((set_denormal_as_zero) ? "on" : "off");
    });
  }

  use_per_session_threads_ = session_options.use_per_session_threads;
  force_spinning_stop_between_runs_ = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigForceSpinningStop, "0") == "1";

  if (use_per_session_threads_) {
    LOGS(*session_logger_, INFO) << "Creating and using per session threadpools since use_per_session_threads_ is true";
    {
      if (!external_intra_op_thread_pool_) {
        bool allow_intra_op_spinning =
            session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowIntraOpSpinning, "1") == "1";
        OrtThreadPoolParams to = session_options_.intra_op_param;
        std::basic_stringstream<ORTCHAR_T> ss;
        if (to.name) {
          ss << to.name << ORT_TSTR("-");
        }
        ss << ORT_TSTR("session-") << session_id_ << ORT_TSTR("-intra-op");
        thread_pool_name_ = ss.str();
        to.name = thread_pool_name_.c_str();
        to.set_denormal_as_zero = set_denormal_as_zero;
        // If the thread pool can use all the processors, then
        // we set affinity of each thread to each processor.
        to.allow_spinning = allow_intra_op_spinning;
        to.dynamic_block_base_ = std::stoi(session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDynamicBlockBase, "0"));
        LOGS(*session_logger_, INFO) << "Dynamic block base set to " << to.dynamic_block_base_;

        // Set custom threading functions
        to.custom_create_thread_fn = session_options_.custom_create_thread_fn;
        to.custom_thread_creation_options = session_options.custom_thread_creation_options;
        to.custom_join_thread_fn = session_options_.custom_join_thread_fn;
        if (session_options_.config_options.TryGetConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, to.affinity_str)) {
          ORT_ENFORCE(!to.affinity_str.empty(), "Affinity string must not be empty");
        }
        to.auto_set_affinity = to.thread_pool_size == 0 &&
                               session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL &&
                               to.affinity_str.empty();

        if (to.custom_create_thread_fn) {
          ORT_ENFORCE(to.custom_join_thread_fn, "custom join thread function not set for intra op thread pool");
        }

        thread_pool_ =
            concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
      }
    }
    if (session_options_.execution_mode == ExecutionMode::ORT_PARALLEL) {
      if (!external_inter_op_thread_pool_) {
        bool allow_inter_op_spinning =
            session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigAllowInterOpSpinning, "1") == "1";
        OrtThreadPoolParams to = session_options_.inter_op_param;
        to.auto_set_affinity = to.thread_pool_size == 0 && session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL;
        std::basic_stringstream<ORTCHAR_T> ss;
        if (to.name) {
          ss << to.name << ORT_TSTR("-");
        }
        ss << ORT_TSTR("session-") << session_id_ << ORT_TSTR("-inter-op");
        inter_thread_pool_name_ = ss.str();
        to.name = inter_thread_pool_name_.c_str();
        to.set_denormal_as_zero = set_denormal_as_zero;
        to.allow_spinning = allow_inter_op_spinning;
        to.dynamic_block_base_ = std::stoi(session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDynamicBlockBase, "0"));

        // Set custom threading functions
        to.custom_create_thread_fn = session_options_.custom_create_thread_fn;
        to.custom_thread_creation_options = session_options.custom_thread_creation_options;
        to.custom_join_thread_fn = session_options_.custom_join_thread_fn;

        if (to.custom_create_thread_fn) {
          ORT_ENFORCE(to.custom_join_thread_fn, "custom join thread function not set for inter op thread pool");
        }
        inter_op_thread_pool_ =
            concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTER_OP);
        if (inter_op_thread_pool_ == nullptr) {
          LOGS(*session_logger_, INFO) << "Failed to create the inter-op thread pool for the parallel executor, setting ExecutionMode to SEQUENTIAL";
          session_options_.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
        }
      }
    }
  } else {
    LOGS(*session_logger_, INFO) << "Using global/env threadpools since use_per_session_threads_ is false";
    intra_op_thread_pool_from_env_ = session_env.GetIntraOpThreadPool();
    inter_op_thread_pool_from_env_ = session_env.GetInterOpThreadPool();
    ORT_ENFORCE(session_env.EnvCreatedWithGlobalThreadPools(),
                "When the session is not configured to use per session"
                " threadpools, the env must be created with the the CreateEnvWithGlobalThreadPools API.");
  }

  session_profiler_.Initialize(session_logger_);
  if (session_options_.enable_profiling) {
    StartProfiling(session_options_.profile_file_prefix);
  }

  telemetry_ = {};
}

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env)
    :
#if !defined(ORT_MINIMAL_BUILD)
      graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
#endif
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  // Initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   const Environment& session_env,
                                   onnxruntime::concurrency::ThreadPool* external_intra_op_thread_pool,
                                   onnxruntime::concurrency::ThreadPool* external_inter_op_thread_pool)
    :
#if !defined(ORT_MINIMAL_BUILD)
      graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
#endif
      logging_manager_(session_env.GetLoggingManager()),
      external_intra_op_thread_pool_(external_intra_op_thread_pool),
      external_inter_op_thread_pool_(external_inter_op_thread_pool),
      environment_(session_env) {
  // Initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#if !defined(ORT_MINIMAL_BUILD)
InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   const PathString& model_uri)
    : model_location_(model_uri),
      graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  auto status = Model::Load(model_location_, model_proto_);
  ORT_ENFORCE(status.IsOK(), "Given model could not be parsed while creating inference session. Error message: ",
              status.ErrorMessage());
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#ifdef _WIN32
InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   const Environment& session_env,
                                   const std::string& model_uri)
    : InferenceSession(session_options, session_env, ToPathString(model_uri)) {
}
#endif

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   std::istream& model_istream)
    : graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  Status st = Model::Load(model_istream, &model_proto_);
  ORT_ENFORCE(st.IsOK(), "Could not parse model successfully while constructing the inference session");
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

InferenceSession::InferenceSession(const SessionOptions& session_options, const Environment& session_env,
                                   const void* model_data, int model_data_len)
    : graph_transformer_mgr_(session_options.max_num_graph_transformation_steps),
      logging_manager_(session_env.GetLoggingManager()),
      environment_(session_env) {
  const bool result = model_proto_.ParseFromArray(model_data, model_data_len);
  ORT_ENFORCE(result, "Could not parse model successfully while constructing the inference session");
  is_model_proto_parsed_ = true;
  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, session_env);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

InferenceSession::~InferenceSession() {
  if (session_options_.enable_profiling) {
    ORT_TRY {
      EndProfiling();
    }
    ORT_CATCH(const std::exception& e) {
      // TODO: Currently we have no way to transport this error to the API user
      // Maybe this should be refactored, so that profiling must be explicitly
      // started and stopped via C-API functions.
      // And not like now a session option and therefore profiling must be started
      // and stopped implicitly.
      ORT_HANDLE_EXCEPTION([&]() {
        LOGS(*session_logger_, ERROR) << "Error during EndProfiling(): " << e.what();
      });
    }
    ORT_CATCH(...) {
      LOGS(*session_logger_, ERROR) << "Unknown error during EndProfiling()";
    }
  }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  if (session_activity_started_)
    TraceLoggingWriteStop(session_activity, "OrtInferenceSessionActivity");
#endif
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  GetMemoryProfiler().GenerateMemoryProfile();
#endif
}

common::Status InferenceSession::RegisterExecutionProvider(const std::shared_ptr<IExecutionProvider>& p_exec_provider) {
  if (p_exec_provider == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for exec provider");
  }

  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

  if (is_inited_) {
    // adding an EP is pointless as the graph as already been partitioned so no nodes will be assigned to
    // the new EP
    LOGS(*session_logger_, ERROR) << "Execution providers must be registered before the session is initialized. ";
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Execution providers must be registered before the session is initialized.");
  }

  const std::string& provider_type = p_exec_provider->Type();

  // Some session option values (default or user provided) may not work with some EPs.
  // Rather than put the onus on the user to know these, make the appropriate change while logging the change.
  if (provider_type == onnxruntime::kDmlExecutionProvider) {
    // DML's memory is not byte addressable and hence mem pattern doesn't work.
    if (session_options_.enable_mem_pattern) {
      LOGS(*session_logger_, INFO)
          << "Having memory pattern enabled is not supported while using the DML Execution Provider. "
          << "So disabling it for this session since it uses the DML Execution Provider.";
      session_options_.enable_mem_pattern = false;
    }

    // Default this option to true when the DML EP is registered.
    // This should be removed if QDQ is supported for DML through QDQSelectorActionTransformer and the DML EP does not
    // rely on the constant folding pass for DequantizeLinear.
    optional<std::string> disable_quant_qdq = session_options_.config_options.GetConfigEntry(kOrtSessionOptionsDisableQuantQDQ);

    if (disable_quant_qdq == std::nullopt) {
      LOGS(*session_logger_, INFO)
          << "QDQ quantization is not supported while using the DML Execution Provider. "
          << "So disabling it for this session since it uses the DML Execution Provider.";

      auto st = session_options_.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1");
      if (!st.IsOK()) {
        return st;
      }
    } else if (*disable_quant_qdq != "1") {
      LOGS(*session_logger_, WARNING)
          << "QDQ quantization is not supported while using the DML Execution Provider. "
          << "It is enabled within session options which may result in lower performance.";
    }

    // Parallel execution mode does not support DML EP
    if (session_options_.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
      LOGS(*session_logger_, INFO)
          << "Parallel execution mode does not support the DML Execution Provider. "
          << "So making the execution mode sequential for this session since it uses the DML Execution Provider.";

      session_options_.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    }
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Create Custom Op if EP requests it
  std::vector<OrtCustomOpDomain*> custom_op_domains;
  p_exec_provider->GetCustomOpDomainList(custom_op_domains);

  if (!custom_op_domains.empty()) {
    if (AddCustomOpDomains(custom_op_domains) != Status::OK()) {
      LOGS(*session_logger_, WARNING) << "Can't register custom op domains with ORT for " << provider_type;
    }
  }
#endif

  // if any EPs do not support concurrent calls to Run we add locking around graph execution
  if (p_exec_provider->ConcurrentRunSupported() == false) {
    is_concurrent_run_supported_ = false;
  }

  VLOGS(*session_logger_, 1) << "Adding execution provider of type: " << provider_type;
  auto p_data_xfr = p_exec_provider->GetDataTransfer();
  if (p_data_xfr) {
    auto st = data_transfer_mgr_.RegisterDataTransfer(std::move(p_data_xfr));
    if (!st.IsOK()) {
      return st;
    }
  }

  p_exec_provider->SetLogger(session_logger_);
  session_profiler_.AddEpProfilers(p_exec_provider->GetProfiler());
  return execution_providers_.Add(provider_type, p_exec_provider);
}

// Custom Op support
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
common::Status InferenceSession::AddCustomOpDomains(gsl::span<OrtCustomOpDomain* const> op_domains) {
  std::shared_ptr<CustomRegistry> custom_registry;
  ORT_RETURN_IF_ERROR_SESSIONID_(CreateCustomRegistry(op_domains, custom_registry));
  ORT_RETURN_IF_ERROR_SESSIONID_(RegisterCustomRegistry(custom_registry));
  return Status::OK();
}

common::Status InferenceSession::RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
  if (custom_registry == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for custom registry");
  }

  custom_registries_.push_back(custom_registry);

  // Insert session-level customized kernel registry.
  kernel_registry_manager_.RegisterKernelRegistry(custom_registry->GetKernelRegistry());

#if !defined(ORT_MINIMAL_BUILD)
  custom_schema_registries_.push_back(custom_registry->GetOpschemaRegistry());
#endif
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

#if !defined(ORT_MINIMAL_BUILD)
common::Status InferenceSession::RegisterGraphTransformer(
    std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer, TransformerLevel level) {
  if (p_graph_transformer == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for graph transformer");
  }

  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

  if (is_inited_) {
    // adding a transformer now is pointless as the graph as already been transformed
    LOGS(*session_logger_, ERROR) << "Graph transformers must be registered before the session is initialized.";
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Graph transformers must be registered before the session is initialized.");
  }

  return graph_transformer_mgr_.Register(std::move(p_graph_transformer), level);
}

common::Status InferenceSession::SaveToOrtFormat(const PathString& filepath) const {
  ORT_RETURN_IF_NOT(FLATBUFFERS_LITTLEENDIAN, "ort format only supports little-endian machines");

  // Get the byte size of the ModelProto and round it to the next MB and use it as flatbuffers' init_size
  // TODO: Investigate whether we should set a max size, and clarify the cost of having a buffer smaller than
  // what the total flatbuffers serialized size will be.
  constexpr size_t m_bytes = 1024 * 1024;
  size_t fbs_buffer_size = std::max(m_bytes, model_->ToProto().ByteSizeLong());
  fbs_buffer_size = ((fbs_buffer_size + m_bytes - 1) / m_bytes) * m_bytes;
  flatbuffers::FlatBufferBuilder builder(fbs_buffer_size);

  auto ort_model_version = builder.CreateString(std::to_string(kOrtModelVersion));
  flatbuffers::Offset<fbs::Model> fbs_model;
  ORT_RETURN_IF_ERROR(
      model_->SaveToOrtFormat(builder, fbs_model));

  flatbuffers::Offset<fbs::KernelTypeStrResolver> fbs_kernel_type_str_resolver;
  KernelTypeStrResolver kernel_type_str_resolver{};
  ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterGraphNodeOpSchemas(model_->MainGraph()));
  ORT_RETURN_IF_ERROR(standalone::RegisterCustomOpNodeSchemas(kernel_type_str_resolver, model_->MainGraph()));

  for (const auto op_schema : saved_runtime_optimization_produced_node_op_schemas_) {
    ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterOpSchema(*op_schema));
  }

  ORT_RETURN_IF_ERROR(
      kernel_type_str_resolver.SaveToOrtFormat(builder, fbs_kernel_type_str_resolver));

  fbs::InferenceSessionBuilder sb(builder);
  sb.add_ort_version(ort_model_version);
  sb.add_model(fbs_model);
  sb.add_kernel_type_str_resolver(fbs_kernel_type_str_resolver);
  auto session = sb.Finish();
  builder.Finish(session, fbs::InferenceSessionIdentifier());

  {
    std::ofstream file(filepath, std::ios::binary);
    uint8_t* buf = builder.GetBufferPointer();
    int size = builder.GetSize();
    file.write(reinterpret_cast<const char*>(buf), size);
    ORT_RETURN_IF_NOT(file, "Failed to save ORT format model to file: ", ToUTF8String(filepath));
  }

  return Status::OK();
}

common::Status InferenceSession::LoadWithLoader(std::function<common::Status(std::shared_ptr<Model>&)> loader,
                                                const std::string& event_name) {
  Status status = Status::OK();
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.Start();
  }
  ORT_TRY {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (is_model_loaded_) {  // already loaded
      LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
      return common::Status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session already contains a loaded model.");
    }

    std::shared_ptr<onnxruntime::Model> p_tmp_model;
    status = loader(p_tmp_model);
    ORT_RETURN_IF_ERROR_SESSIONID_(status);

    model_ = p_tmp_model;

    status = DoPostLoadProcessing(*model_);
    ORT_RETURN_IF_ERROR_SESSIONID_(status);

    // all steps complete, mark the model as loaded.
    is_model_loaded_ = true;

    telemetry_.event_name_ = event_name;
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(common::ONNXRUNTIME, common::FAIL, "Exception during loading: " + std::string(ex.what()));
    });
  }
  ORT_CATCH(...) {
    LOGS(*session_logger_, ERROR) << "Unknown exception";
    status = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION,
                    "Encountered unknown exception in LoadWithLoader()");
  }

  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, event_name, tp);
  }

  return status;
}

common::Status InferenceSession::LoadOnnxModel(const PathString& model_uri) {
  model_location_ = model_uri;
  auto loader = [this](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_location_, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif
    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    return onnxruntime::Model::Load(model_location_, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                    *session_logger_,
                                    ModelOptions(true, strict_shape_type_inference));
  };

  common::Status st = LoadWithLoader(loader, "model_loading_uri");
  if (!st.IsOK()) {
    std::ostringstream oss;
    oss << "Load model from " << ToUTF8String(model_uri) << " failed:" << st.ErrorMessage();
    return common::Status(st.Category(), st.Code(), oss.str());
  }
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status InferenceSession::FilterEnabledOptimizers(InlinedHashSet<std::string>&& optimizers_to_disable) {
  optimizers_to_disable_ = std::move(optimizers_to_disable);
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

common::Status InferenceSession::Load(const PathString& model_uri) {
  std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigLoadModelFormat, "");
  bool has_explicit_type = !model_type.empty();

  if ((has_explicit_type && model_type == "ORT") ||
      (!has_explicit_type && fbs::utils::IsOrtFormatModel(model_uri))) {
    return LoadOrtModel(model_uri);
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  return LoadOnnxModel(model_uri);
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ONNX format model is not supported in this build.");
#endif
}

#ifdef _WIN32
common::Status InferenceSession::Load(const std::string& model_uri) {
  return Load(ToPathString(model_uri));
}
#endif

common::Status InferenceSession::Load(const void* model_data, int model_data_len) {
  std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigLoadModelFormat, "");
  bool has_explicit_type = !model_type.empty();

  if ((has_explicit_type && model_type == "ORT") ||
      (!has_explicit_type &&
       fbs::utils::IsOrtFormatModelBytes(model_data, model_data_len))) {
    return LoadOrtModel(model_data, model_data_len);
  }

#if !defined(ORT_MINIMAL_BUILD)
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, model_data, model_data_len](std::shared_ptr<onnxruntime::Model>& model) {
    ModelProto model_proto;

    const bool result = model_proto.ParseFromArray(model_data, model_data_len);
    if (!result) {
      return Status(common::ONNXRUNTIME, common::INVALID_PROTOBUF,
                    "Failed to load model because protobuf parsing failed.");
    }
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif

    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    return onnxruntime::Model::Load(std::move(model_proto), PathString(), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_,
                                    ModelOptions(true, strict_shape_type_inference));
  };

  return LoadWithLoader(loader, "model_loading_array");
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ONNX format model is not supported in this build.");
#endif
}

#if !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::LoadOnnxModel(ModelProto model_proto) {
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, &model_proto](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif
    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    // This call will move model_proto to the constructed model instance
    return onnxruntime::Model::Load(std::move(model_proto), PathString(), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_,
                                    ModelOptions(true, strict_shape_type_inference));
  };

  return LoadWithLoader(loader, "model_loading_proto");
}

common::Status InferenceSession::LoadOnnxModel(std::unique_ptr<ModelProto> p_model_proto) {
  return LoadOnnxModel(std::move(*p_model_proto));
}

common::Status InferenceSession::Load(std::istream& model_istream, bool allow_released_opsets_only) {
  if (is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has already been parsed. "
                           "Invoke Load().");
  }

  auto loader = [this, &model_istream, &allow_released_opsets_only](std::shared_ptr<onnxruntime::Model>& model) {
    ModelProto model_proto;
    Status st = Model::Load(model_istream, &model_proto);
    if (!st.IsOK()) {
      return st;
    }
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif
    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    ModelOptions model_opts(allow_released_opsets_only,
                            strict_shape_type_inference);
    return onnxruntime::Model::Load(std::move(model_proto), PathString(), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                    *session_logger_, model_opts);
  };

  return LoadWithLoader(loader, "model_loading_istream");
}

common::Status InferenceSession::Load() {
  if (!is_model_proto_parsed_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "ModelProto corresponding to the model to be loaded has not been parsed yet. "
                           "This API should be called in conjunction with a ctor that takes a model abstraction.");
  }

  auto loader = [this](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(this->model_proto_, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    InlinedVector<OrtCustomOpDomain*> domain_ptrs;
    domain_ptrs.reserve(interop_domains_.size());
    std::copy(std::begin(interop_domains_), std::end(interop_domains_), std::back_inserter(domain_ptrs));
    ORT_RETURN_IF_ERROR(AddCustomOpDomains(domain_ptrs));
#endif
    const bool strict_shape_type_inference = session_options_.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "0") == "1";
    const bool allow_released_opsets_only = session_options_.config_options.GetConfigOrDefault(
                                                kOrtSessionOptionsConfigStrictAllowReleasedOpsetsOnly, "1") == "1";

    // Pass on ownership of the parsed ModelProto to the Model instance (its job here is done by this stage)
    return Model::Load(std::move(this->model_proto_), model_location_, model,
                       HasLocalSchema() ? &custom_schema_registries_ : nullptr, *session_logger_,
                       ModelOptions(allow_released_opsets_only, strict_shape_type_inference));
  };

  return LoadWithLoader(loader, "model_loading_from_saved_proto");
}

common::Status InferenceSession::TransformGraph(onnxruntime::Graph& graph, bool saving_model_in_ort_format) {
  // The transformer order:
  // 1. ensure potential QDQ node units have unique DQ nodes (required transformer).
  //    - This is a required transformer as the ORT code has a hard requirement there are no overlapping QDQ node units.
  //    - We run it here in case optimizers are disabled.
  // 2. run level 1 optimizations. these only use ONNX operators.
  // 3. partition nodes based on EP capabilities. EPs may fuse nodes during this process.
  // 4. run level 2+ optimizations. level 2 and 3 optimizations use contrib ops.
  // 5. insert cast nodes (required transformer).
  // 6. insert copy nodes (required transformer).

  auto apply_transformer_once = [](const GraphTransformer& transformer, const logging::Logger& logger,
                                   Graph& graph) {
    bool modified = false;
    return transformer.Apply(graph, modified, logger);
  };

  // ensure potential QDQ node units have unique DQ nodes
  if (const bool disable_quant_qdq =
          session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsDisableQuantQDQ, "0") == "1";
      !disable_quant_qdq) {
    EnsureUniqueDQForNodeUnit ensure_unique_dq_for_node_unit{};
    ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(ensure_unique_dq_for_node_unit, *session_logger_, graph));
  }

  // apply execution provider independent level 1 graph optimizations.
  ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.ApplyTransformers(graph, TransformerLevel::Level1, *session_logger_));

  // if saving model to ORT format we only assign nodes a custom EP can handle and don't compile them.
  // we do this to preserve the original nodes in the model but prevent optimizers from changing them.
  // at runtime, the ORT format model will re-do the partitioning/compilation of these nodes, which may change
  // to cover fewer nodes due to device capabilities.
  auto mode = saving_model_in_ort_format ? GraphPartitioner::Mode::kAssignOnly
                                         : GraphPartitioner::Mode::kNormal;

  layout_transformation::TransformLayoutFunction transform_layout_fn = nullptr;

  // only provide NCWH to NHWC layout transformer if supported
  if (layout_transformation::IsSupportedOpset(graph)) {
    // we want to run L1 transformers after the layout transform primarily to constant fold any initializers
    // that get converted to an alternative layout.
    // create a lambda to combine the two operations in the layout transformation function
    transform_layout_fn = [this](Graph& graph_to_transform, bool& modified,
                                 const IExecutionProvider& execution_provider,
                                 const layout_transformation::DebugGraphFn& debug_graph_fn) -> Status {
      AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
      ORT_RETURN_IF_ERROR_SESSIONID_(
          layout_transformation::TransformLayoutForEP(graph_to_transform, modified, execution_provider,
                                                      std::move(cpu_allocator), debug_graph_fn));

      if (modified) {
        ORT_RETURN_IF_ERROR_SESSIONID_(
            graph_transformer_mgr_.ApplyTransformers(graph_to_transform, TransformerLevel::Level1, *session_logger_));

        // debug the graph after the L1 transformers have run against any layout transformation changes.
        // this is prior to GraphPartitioner::GetCapabilityForEP calling IExecutionProvider::GetCapability the second
        // time to validate the EP that requested the layout transformation can take all nodes using the new layout.
        // if that fails, this allows debugging the graph used in that GetCapability call.
        if (debug_graph_fn) {
          debug_graph_fn(graph_to_transform);
        }
      }

      return Status::OK();
    };
  }

  // debug infrastructure for layout transformation. it's extremely difficult to trace the transpose optimizer changes
  // manually, so dumping out the model so it can be viewed in Netron makes it far easier
  layout_transformation::DebugGraphFn debug_graph_fn;
  if (transform_layout_fn) {
    bool enable_debug = session_options_.config_options.GetConfigOrDefault(kDebugLayoutTransformation, "0") == "1";

    if (enable_debug) {
      // init counter to 1 to match to documentation and have a more natural output filename of '..._step_1.onnx'
      // for the result of the first step in layout transformation
      debug_graph_fn = [counter = 1, this](const Graph& graph) mutable {
        if (graph.GraphProtoSyncNeeded()) {
          ORT_THROW_IF_ERROR(
              Model::Save(*model_, "post_layout_transform_step_" + std::to_string(counter) + ".onnx"));
        }

        // counter is used to denote the step, so increment regardless of whether we wrote out the model in this step.
        ++counter;
      };
    }
  }

  // Do partitioning based on execution providers' capabilities.
  GraphPartitioner partitioner(kernel_registry_manager_, execution_providers_);
  ORT_RETURN_IF_ERROR_SESSIONID_(partitioner.Partition(graph, session_state_->GetMutableFuncMgr(), transform_layout_fn,
                                                       mode, debug_graph_fn));

  // apply Level2 and higher transformers.
  // we do not run Level 1 again as those transformers assume partitioning will run later to do node assignment.
  for (int i = static_cast<int>(TransformerLevel::Level2); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR_SESSIONID_(
        graph_transformer_mgr_.ApplyTransformers(graph, static_cast<TransformerLevel>(i), *session_logger_));
  }

  // Insert cast node/s.
  {
    const InlinedVector<gsl::not_null<const KernelRegistry*>> kernel_regs =
        kernel_registry_manager_.GetKernelRegistriesByProviderType(kCpuExecutionProvider);
    const KernelRegistry* cpu_regs = nullptr;
    if (!kernel_regs.empty()) {
      cpu_regs = kernel_regs[0];
    }
    InsertCastTransformer insert_cast_transformer{"CastFloat16Transformer", cpu_regs};
    ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(insert_cast_transformer, *session_logger_, graph));
  }

  // Insert copy node/s.
  {
    std::vector<std::string> provider_types;
    for (auto& provider_ptr : execution_providers_) {
      provider_types.push_back(provider_ptr->Type());
    }

    MemcpyTransformer copy_transformer{provider_types, kernel_registry_manager_};
    ORT_RETURN_IF_ERROR_SESSIONID_(apply_transformer_once(copy_transformer, *session_logger_, graph));
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

static Status LoadOrtModelBytes(const PathString& model_uri,
                                gsl::span<const uint8_t>& bytes,
                                std::vector<uint8_t>& bytes_data_holder) {
  size_t num_bytes = 0;
  ORT_RETURN_IF_ERROR(Env::Default().GetFileLength(model_uri.c_str(), num_bytes));

  bytes_data_holder.resize(num_bytes);

  std::ifstream bytes_stream(model_uri, std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(bytes_data_holder.data()), num_bytes);

  if (!bytes_stream) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Load model from ", ToUTF8String(model_uri), " failed. Only ",
                           bytes_stream.gcount(), "/", num_bytes, " bytes were able to be read.");
  }

  bytes = gsl::span<const uint8_t>(bytes_data_holder.data(), num_bytes);

  return Status::OK();
}

Status InferenceSession::LoadOrtModel(const PathString& model_uri) {
  return LoadOrtModelWithLoader(
      [&]() {
        model_location_ = model_uri;
        ORT_RETURN_IF_ERROR(
            LoadOrtModelBytes(model_location_, ort_format_model_bytes_, ort_format_model_bytes_data_holder_));
        return Status::OK();
      });
}

Status InferenceSession::LoadOrtModel(const void* model_data, int model_data_len) {
  return LoadOrtModelWithLoader([&]() {
    const auto& config_options = GetSessionOptions().config_options;
    const auto use_ort_model_bytes_directly =
        config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseORTModelBytesDirectly, "0") == "1";

    if (!use_ort_model_bytes_directly) {
      // copy bytes as we need them to be available when InferenceSession::Initialize is called later.
      ort_format_model_bytes_data_holder_.resize(model_data_len);
      std::copy_n(reinterpret_cast<const uint8_t*>(model_data), model_data_len,
                  ort_format_model_bytes_data_holder_.data());
      ort_format_model_bytes_ = gsl::span<const uint8_t>(ort_format_model_bytes_data_holder_.data(), model_data_len);
    } else {
      // Use the model_data directly to reduce memory consumption
      // This will require the model_data to be alive until the InferenceSession is initialized
      ort_format_model_bytes_ = gsl::span<const uint8_t>(reinterpret_cast<const uint8_t*>(model_data), model_data_len);
    }
    return Status::OK();
  });
}

Status InferenceSession::LoadOrtModelWithLoader(std::function<Status()> load_ort_format_model_bytes) {
  static_assert(FLATBUFFERS_LITTLEENDIAN, "ORT format only supports little-endian machines");

  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

  if (is_model_loaded_) {  // already loaded
    Status status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session already contains a loaded model.");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    return status;
  }

  if (is_inited_) {
    Status status(common::ONNXRUNTIME, common::MODEL_LOADED, "This session has already been initialized.");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    return status;
  }

  ORT_RETURN_IF_ERROR(load_ort_format_model_bytes());

  // Verify the ort_format_model_bytes_ is a valid InferenceSessionBuffer before we access the data
  flatbuffers::Verifier verifier(ort_format_model_bytes_.data(), ort_format_model_bytes_.size());
  ORT_RETURN_IF_NOT(fbs::VerifyInferenceSessionBuffer(verifier), "ORT model verification failed.");

  const auto* fbs_session = fbs::GetInferenceSession(ort_format_model_bytes_.data());
  ORT_RETURN_IF(nullptr == fbs_session, "InferenceSession is null. Invalid ORT format model.");

  // Check version mismatch, for now we will only proceed when runtime version matches the model's ort version
  const auto* fbs_ort_model_version = fbs_session->ort_version();
  ORT_RETURN_IF(fbs_ort_model_version == nullptr, "Serialized version info is null. Invalid ORT format model.");

  const auto model_version = std::stoi(fbs_ort_model_version->str());
  const bool is_supported = IsOrtModelVersionSupported(model_version);

  OrtFormatLoadOptions load_options{};

#if defined(ORT_MINIMAL_BUILD)
  // Note about the ORT format version 5 breaking change.
  // TODO This change was introduced in 1.13. Remove this note a few releases later, e.g., 1.15.
  constexpr auto* kOrtFormatVersion5BreakingChangeNote =
      "This build doesn't support ORT format models older than version 5. "
      "See: https://github.com/microsoft/onnxruntime/blob/rel-1.14.0/docs/ORT_Format_Update_in_1.13.md";

  ORT_RETURN_IF(!is_supported,
                "The ORT format model version [", fbs_ort_model_version->string_view(),
                "] is not supported in this build ", ORT_VERSION, ". ",
                kOrtFormatVersion5BreakingChangeNote);
#else   // ^^ defined(ORT_MINIMAL_BUILD) ^^ / vv !defined(ORT_MINIMAL_BUILD) vv
  const auto has_saved_runtime_optimizations = [](const fbs::InferenceSession& fbs_session) -> bool {
    if (const auto* fbs_model = fbs_session.model()) {
      if (const auto* fbs_graph = fbs_model->graph()) {
        if (const auto* fbs_runtime_opts = fbs_graph->runtime_optimizations()) {
          if (const auto* fbs_runtime_opt_records = fbs_runtime_opts->records()) {
            return fbs_runtime_opt_records->size() > 0;
          }
        }
      }
    }
    return false;
  };

  // models prior to v5 can be handled by inserting the kernel constraints in a full build
  const bool is_supported_with_update = model_version < 5;

  if (is_supported_with_update && has_saved_runtime_optimizations(*fbs_session)) {
    LOGS(*session_logger_, WARNING)
        << "The old ORT format model (version " << fbs_ort_model_version->string_view()
        << ") has saved runtime optimizations. They will be ignored.";
    load_options.ignore_saved_runtime_optimizations = true;
  }

  ORT_RETURN_IF_NOT(is_supported || is_supported_with_update,
                    "The ORT format model version [", fbs_ort_model_version->string_view(),
                    "] is not supported in this build ", ORT_VERSION, ".");
#endif  // !defined(ORT_MINIMAL_BUILD)

  const auto* fbs_model = fbs_session->model();
  ORT_RETURN_IF(nullptr == fbs_model, "Missing Model. Invalid ORT format model.");

  // if we're using the bytes directly because kOrtSessionOptionsConfigUseORTModelBytesDirectly was set and the user
  // provided an existing buffer of bytes when creating the InferenceSession, ort_format_model_bytes_data_holder_
  // will be empty.
  // if that is the case we also allow creating initializers that directly use those bytes.
  const auto& config_options = session_options_.config_options;
  using_ort_model_bytes_for_initializers_ =
      load_options.can_use_flatbuffer_for_initializers =
          ort_format_model_bytes_data_holder_.empty() &&
          config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseORTModelBytesForInitializers, "0") == "1";

  // need to go from unique_ptr to shared_ptr when moving into model_
  std::unique_ptr<Model> tmp_model;
#if !defined(ORT_MINIMAL_BUILD)
  ORT_RETURN_IF_ERROR(Model::LoadFromOrtFormat(*fbs_model,
                                               HasLocalSchema() ? &custom_schema_registries_ : nullptr,
                                               load_options, *session_logger_, tmp_model));
#else
  ORT_RETURN_IF_ERROR(Model::LoadFromOrtFormat(*fbs_model, load_options, *session_logger_, tmp_model));
#endif

  ORT_RETURN_IF_ERROR(SaveModelMetadata(*tmp_model));
  model_ = std::move(tmp_model);

  KernelTypeStrResolver kernel_type_str_resolver{};
  if (const auto* fbs_kernel_type_str_resolver = fbs_session->kernel_type_str_resolver();
      fbs_kernel_type_str_resolver != nullptr) {
    ORT_RETURN_IF_ERROR(kernel_type_str_resolver.LoadFromOrtFormat(*fbs_kernel_type_str_resolver));
  } else {
#if !defined(ORT_MINIMAL_BUILD)
    // insert the kernel type constraints if we're updating an old model that had kernel hashes.
    if (is_supported_with_update) {
      ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterGraphNodeOpSchemas(model_->MainGraph()));
    }
#endif
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  ORT_RETURN_IF_ERROR(
      kernel_type_str_resolver_utils::AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(
          kernel_type_str_resolver));
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  kernel_registry_manager_.SetKernelTypeStrResolver(std::move(kernel_type_str_resolver));

  is_model_loaded_ = true;

  return Status::OK();
}

bool InferenceSession::IsInitialized() const {
  std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
  return is_inited_;
}

static bool ModelHasFP16InputsHelper(const onnx::TypeProto& type_proto) {
  switch (type_proto.value_case()) {
    case ::onnx::TypeProto::ValueCase::kTensorType: {
      if (type_proto.has_tensor_type()) {
        auto& tensor_type = type_proto.tensor_type();
        if (tensor_type.elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
          return true;
        }
      }
      break;
    }
    case ::onnx::TypeProto::ValueCase::kSequenceType: {
      if (type_proto.has_sequence_type()) {
        auto& sequence_type = type_proto.sequence_type();
        return ModelHasFP16InputsHelper(sequence_type.elem_type());
      }
      break;
    }
    case ::onnx::TypeProto::ValueCase::kMapType: {
      if (type_proto.has_map_type()) {
        auto& map_type = type_proto.map_type();
        return ModelHasFP16InputsHelper(map_type.value_type());
      }
      break;
    }
    default:
      break;
  }
  return false;
}

static bool ModelHasFP16Inputs(const Graph& graph) {
  for (auto& input : graph.GetInputs()) {
    if (input->Exists() && ModelHasFP16InputsHelper(*(input->TypeAsProto()))) {
      return true;
    }
  }
  return false;
}

common::Status InferenceSession::AddPrePackedWeightsContainer(PrepackedWeightsContainer* prepacked_weights_container) {
  if (prepacked_weights_container == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The provided PrePackedWeightsContainer instance to be added to the session is null");
  }

  if (prepacked_weights_container_ != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The session already has a PrePackedWeightsContainer instance");
  }

  prepacked_weights_container_ = prepacked_weights_container;

  return Status::OK();
}

namespace {
Status PartitionOrtFormatModel(onnxruntime::Graph& graph,
                               const ExecutionProviders& providers,
                               KernelRegistryManager& kernel_registry_manager,
                               SessionState& session_state) {
  layout_transformation::TransformLayoutFunction transform_layout_fn = nullptr;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // only provide NCWH to NHWC layout transformer if supported
  if (layout_transformation::IsSupportedOpset(graph)) {
    transform_layout_fn =
        [](Graph& graph_to_transform, bool& modified,
           const IExecutionProvider& execution_provider,
           const layout_transformation::DebugGraphFn& debug_graph_fn) -> Status {
      AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
      return layout_transformation::TransformLayoutForEP(graph_to_transform, modified, execution_provider,
                                                         std::move(cpu_allocator), debug_graph_fn);
    };
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  GraphPartitioner partitioner(kernel_registry_manager, providers);
  ORT_RETURN_IF_ERROR(partitioner.Partition(graph,
                                            session_state.GetMutableFuncMgr(),
                                            transform_layout_fn,
                                            GraphPartitioner::Mode::kOrtFormatLoad));

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
Status ApplyOrtFormatModelRuntimeOptimizations(
    onnxruntime::Graph& graph, const logging::Logger& logger, const SessionOptions& session_options,
    const InlinedHashSet<std::string>& optimizers_to_disable, const IExecutionProvider& cpu_ep) {
  bool modified = false;

  for (int level = static_cast<int>(TransformerLevel::Level2);
       level <= static_cast<int>(session_options.graph_optimization_level);
       ++level) {
    const auto transformers = optimizer_utils::GenerateTransformersForMinimalBuild(
        static_cast<TransformerLevel>(level), session_options, SatRuntimeOptimizationLoadContext{}, cpu_ep,
        optimizers_to_disable);

    for (const auto& transformer : transformers) {
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));
    }
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}  // namespace

static void ResolveMemoryPatternFlags(SessionState& session_state) {
  session_state.ResolveMemoryPatternFlag();

  for (const auto& entry : session_state.GetSubgraphSessionStateMap()) {
    for (const auto& name_to_subgraph_session_state : entry.second) {
      ResolveMemoryPatternFlags(*name_to_subgraph_session_state.second);
    }
  }
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// VC++ reports: "Releasing unheld lock 'l' in function 'onnxruntime::InferenceSession::Initialize'". But I don't see anything wrong.
#pragma warning(disable : 26117)
#endif
common::Status InferenceSession::Initialize() {
  Status status = Status::OK();
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.Start();
  }

  ORT_TRY {
    LOGS(*session_logger_, INFO) << "Initializing session.";
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogSessionCreationStart();

    bool have_cpu_ep = false;

    {
      std::lock_guard<onnxruntime::OrtMutex> initial_guard(session_mutex_);

      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded.");
      }

      if (is_inited_) {  // already initialized
        LOGS(*session_logger_, INFO) << "Session has already been initialized.";
        return common::Status::OK();
      }

      have_cpu_ep = execution_providers_.Get(onnxruntime::kCpuExecutionProvider) != nullptr;
    }

    // Verify that there are no external initializers in the graph if external data is disabled.
    onnxruntime::Graph& graph = model_->MainGraph();
#ifdef DISABLE_EXTERNAL_INITIALIZERS
    const InitializedTensorSet& initializers = graph.GetAllInitializedTensors();
    for (const auto& it : initializers) {
      if (utils::HasExternalData(*it.second)) {
        return common::Status(common::ONNXRUNTIME, common::FAIL,
                              "Initializer tensors with external data is not allowed.");
      }
    }
#endif

    // Register default CPUExecutionProvider if user didn't provide it through the Register() calls.
    // RegisterExecutionProvider locks the session_mutex_ so we can't be holding it when we call that
    if (!have_cpu_ep) {
      LOGS(*session_logger_, INFO) << "Adding default CPU execution provider.";
      CPUExecutionProviderInfo epi{session_options_.enable_cpu_mem_arena};
      auto p_cpu_exec_provider = std::make_unique<CPUExecutionProvider>(epi);
      ORT_RETURN_IF_ERROR_SESSIONID_(RegisterExecutionProvider(std::move(p_cpu_exec_provider)));
      execution_providers_.SetCpuProviderWasImplicitlyAdded(true);
    }

    // re-acquire mutex
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);

#if !defined(DISABLE_EXTERNAL_INITIALIZERS) && !defined(ORT_MINIMAL_BUILD)
    if (!session_options_.external_initializers.empty()) {
      ORT_RETURN_IF_ERROR_SESSIONID_(graph.InjectExternalInitializedTensors(session_options_.external_initializers));
      InlinedHashMap<std::string, OrtValue>{}.swap(session_options_.external_initializers);
    }
#endif

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    TraceLoggingWriteStart(session_activity, "OrtInferenceSessionActivity");
    session_activity_started_ = true;
#endif

    // now that we have all the execution providers, create the session state
    session_state_ = std::make_unique<SessionState>(
        model_->MainGraph(),
        execution_providers_,
        GetIntraOpThreadPoolToUse(),
        GetInterOpThreadPoolToUse(),
        data_transfer_mgr_,
        *session_logger_,
        session_profiler_,
        session_options_,
        prepacked_weights_container_);

    bool use_env_allocators =
        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseEnvAllocators, "0") == "1";
    if (use_env_allocators) {
      LOGS(*session_logger_, INFO) << "This session will use the allocator registered with the environment.";
      session_state_->UpdateAllocatorsWithEnvAllocators(environment_.GetRegisteredSharedAllocators());
    }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
    // Don't want to pollute SessionState constructor since memory profile is enabled optionally.
    session_state_->SetMemoryProfiler(&memory_profiler_);
#endif

    // Collect the kernel registries from execution provider instances;
    // There are 2 kinds of kernel registries with priority from high to low as below,
    // 1. Custom execution provider type specific kernel registries.
    // 2. common execution provider type specific kernel registries.
    // Kernel registries are shared across sessions.
    // The 1st ones should have already been registered via session-level API into KernelRegistryManager.
    //
    // Register 2nd registries into KernelRegistryManager.
    ORT_RETURN_IF_ERROR_SESSIONID_(kernel_registry_manager_.RegisterKernels(execution_providers_));

    const bool loading_ort_format = !ort_format_model_bytes_.empty();
    const bool saving_model = !session_options_.optimized_model_filepath.empty();
    const bool saving_ort_format = [&]() {
      if (saving_model) {
        const std::string model_type = session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigSaveModelFormat, "");
        const bool has_explicit_type = !model_type.empty();
        return ((has_explicit_type && model_type == "ORT") ||
                (!has_explicit_type &&
                 fbs::utils::IsOrtFormatModel(session_options_.optimized_model_filepath)));
      }
      return false;
    }();

    if (!loading_ort_format) {
#if !defined(ORT_MINIMAL_BUILD)
      const auto minimal_build_opt_config_value = session_options_.config_options.GetConfigOrDefault(
          kOrtSessionOptionsConfigMinimalBuildOptimizations, "");
      MinimalBuildOptimizationHandling minimal_build_optimization_handling{};
      ORT_RETURN_IF_ERROR_SESSIONID_(GetMinimalBuildOptimizationHandling(minimal_build_opt_config_value,
                                                                         saving_ort_format,
                                                                         minimal_build_optimization_handling));

      auto record_runtime_optimization_produced_op_schema = [this](const ONNX_NAMESPACE::OpSchema& op_schema) {
        saved_runtime_optimization_produced_node_op_schemas_.insert(&op_schema);
        return Status::OK();
      };

      // add predefined transformers
      ORT_RETURN_IF_ERROR_SESSIONID_(AddPredefinedTransformers(graph_transformer_mgr_,
                                                               session_options_.graph_optimization_level,
                                                               minimal_build_optimization_handling,
                                                               record_runtime_optimization_produced_op_schema));

#ifdef USE_DML
      if (execution_providers_.Get(kDmlExecutionProvider)) {
        // DML graph fusion is an important runtime optimization that cannot be done ahead of time; it must be disabled
        // when running in "offline mode" and saving an optimized model to disk. To support users that want to optimize
        // models offline, and then disable graph optimizations when running "online", this transformer ignores the ORT
        // graph optimization level and is generally always applied.
        bool dml_graph_fusion_enabled = session_options_.optimized_model_filepath.empty() &&
                                        session_options_.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDisableDmlGraphFusion, "0") == "0";

        if (dml_graph_fusion_enabled) {
          std::unique_ptr<onnxruntime::GraphTransformer> dmlGraphFusionTransformer = std::make_unique<Dml::DmlGraphFusionTransformer>("DmlGraphFusionTransformer",
                                                                                                                                      execution_providers_.Get(kDmlExecutionProvider));
          if (dmlGraphFusionTransformer == nullptr) {
            return Status(common::ONNXRUNTIME, common::FAIL, "DmlGraphFusionTransformer is nullptr");
          }
          ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.Register(std::move(dmlGraphFusionTransformer), onnxruntime::TransformerLevel::Level3));
        }

        // This transformer applies DML-specific fusions that go beyond what ORT offers by default
        bool dml_operator_fusion_enabled = session_options_.graph_optimization_level >= TransformerLevel::Level2;
        if (dml_operator_fusion_enabled) {
          std::unique_ptr<onnxruntime::GraphTransformer> dmlOperatorFusionTransformer = std::make_unique<Dml::GraphTransformer>("DmlOperatorFusionTransformer");
          if (dmlOperatorFusionTransformer == nullptr) {
            return Status(common::ONNXRUNTIME, common::FAIL, "DmlOperatorFusionTransformer is nullptr");
          }
          ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr_.Register(std::move(dmlOperatorFusionTransformer), onnxruntime::TransformerLevel::Level2));
        }
      }
#endif

      // apply any transformations to the main graph and any subgraphs
      ORT_RETURN_IF_ERROR_SESSIONID_(TransformGraph(graph, saving_ort_format));

      // now that all the transforms are done, call Resolve on the main graph. this will recurse into the subgraphs.
      ORT_RETURN_IF_ERROR_SESSIONID_(graph.Resolve());

      // Currently CUDA graph is only considered by CUDA EP and TRT EP.
      //
      // Check for CUDA EP:
      // If the CUDA EP is part of the providers list for this session AND
      // The CUDA EP is configured to do a graph capture AND
      // All the "compute" graph nodes have been assigned to the CUDA EP,
      // Then the CUDA EP is cached for triggering a ReplayGraph() in Run().
      //
      // Check for TRT EP:
      // If the TRT EP is part of the providers list for this session AND
      // The TRT EP is configured to do a graph capture AND
      // All the graph nodes have been assigned to the TRT EP,
      // Then the TRT EP is cached for triggering a ReplayGraph() in Run().
      std::vector<const char*> cuda_graph_support_ep_list = {onnxruntime::kTensorrtExecutionProvider, onnxruntime::kCudaExecutionProvider};

      for (auto& it : cuda_graph_support_ep_list) {
        auto* target_ep = execution_providers_.Get(it);

        if (target_ep && target_ep->IsGraphCaptureEnabled()) {
          // CUDA Graphs can't work with control flow nodes
          if (HasControlflowNodes(graph)) {
            LOGS(*session_logger_, ERROR) << "This session cannot use the CUDA Graph feature as requested by the user "
                                          << "as the model has control flow nodes which can't be supported by CUDA Graphs.";

            ORT_RETURN_IF_ERROR_SESSIONID_(
                ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                "This session cannot use the CUDA Graph feature as requested by the user "
                                "as the model has control flow nodes which can't be supported by CUDA Graphs."));
          }

          if (strcmp(target_ep->Type().c_str(), onnxruntime::kCudaExecutionProvider) == 0) {
            // Ensure that all nodes have been partitioned to CUDA or CPU EP && there are no memcpy nodes
            // The reasoning behind this logic is that certain shape nodes will be forced onto CPU
            // and as long as there are no memcpy nodes this is confirmation that no compute nodes have been placed on the CPU EP
            // which is all we care about.
            if (!AreAllComputeNodesAssignedToCudaEp(graph)) {
              LOGS(*session_logger_, ERROR) << "This session cannot use the CUDA Graph feature as requested by the user "
                                            << " as all compute graph nodes have not been partitioned to the CUDA EP.";

              ORT_RETURN_IF_ERROR_SESSIONID_(
                  ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                  "This session cannot use the CUDA Graph feature as requested by the user "
                                  " as all compute graph nodes have not been partitioned to the CUDA EP."));
            }

            // Log a warning for the user to know that there are shape subgraphs that will execute on CPU
            if (HasShapeSubgraphNodes(graph)) {
              LOGS(*session_logger_, WARNING) << "This model has shape massaging nodes that will execute on CPU. "
                                              << "Use the CUDA Graph feature with caution. "
                                              << "As long as the intermediate shapes produced in the model "
                                              << "using the representative input used to capture the CUDA graph, "
                                              << "will match the shapes produced in the model for other inputs "
                                              << "of the same shape as the representative input (common case), "
                                              << "it is safe to use the CUDA Graph feature.";
            }
          } else {
            // Following code path is for TRT EP currently.
            if (!AreAllNodesInMainGraphAssignedToOneEp(graph, target_ep->Type())) {
              LOGS(*session_logger_, ERROR) << "This session cannot use the CUDA Graph feature as requested by the user "
                                            << "as all the graph nodes have not been assigned to "
                                            << target_ep->Type();

              // Return error status as we don't want the session initialization to complete successfully
              // if the user has requested usage of CUDA Graph feature and we cannot honor that.
              ORT_RETURN_IF_ERROR_SESSIONID_(
                  ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                  "This session cannot use the CUDA Graph feature as requested by the user "
                                  "as all the graph nodes have not been assigned to " +
                                      target_ep->Type()));
            }
          }

          LOGS(*session_logger_, INFO) << "This session will use the CUDA Graph feature as requested by the user.";
          cached_execution_provider_for_graph_replay_.SetExecutionProvider(target_ep);
          break;  // Make sure only one ep can run CUDA graph.
        }
      }

      const bool disable_cpu_ep_fallback = session_options_.config_options.GetConfigOrDefault(
                                               kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";

      // Handle the option to disable the fallback of graph nodes to the CPU EP.
      // If the user disabled fallback, but also explicitly added the CPU EP to the session, return an error status.
      // If the user disabled fallback and any graph node is assigned to the CPU EP, return an error status.
      if (disable_cpu_ep_fallback) {
        // Returns true if any graph nodes have been assigned to the CPU EP.
        auto are_nodes_assigned_to_cpu_ep = [](const Graph& graph) -> bool {
          for (const auto& node : graph.Nodes()) {
            const auto& node_provider = node.GetExecutionProviderType();

            if (node_provider.empty() || node_provider == onnxruntime::kCpuExecutionProvider) {
              return true;
            }
          }

          return false;
        };

        if (!execution_providers_.GetCpuProviderWasImplicitlyAdded()) {
          const char* err_msg =
              "Conflicting session configuration: explicitly added the CPU EP to the "
              "session, but also disabled fallback to the CPU EP via session configuration options.";

          LOGS(*session_logger_, ERROR) << err_msg;
          ORT_RETURN_IF_ERROR_SESSIONID_(ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, err_msg));
        } else if (are_nodes_assigned_to_cpu_ep(graph)) {
          const char* err_msg =
              "This session contains graph nodes that are assigned to the default CPU EP, "
              "but fallback to CPU EP has been explicitly disabled by the user.";
          LOGS(*session_logger_, ERROR) << err_msg;
          ORT_RETURN_IF_ERROR_SESSIONID_(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, err_msg));
        }
      }

      // Update temporary copies of metadata, input- and output definitions to the same state as the resolved graph
      ORT_RETURN_IF_ERROR_SESSIONID_(SaveModelMetadata(*model_));
#else   // !defined(ORT_MINIMAL_BUILD)
      ORT_RETURN_IF_ERROR_SESSIONID_(
          ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "Loading anything other than ORT format models is not enabled in this build."));
#endif  // !defined(ORT_MINIMAL_BUILD)
    } else {
      ORT_RETURN_IF_ERROR_SESSIONID_(PartitionOrtFormatModel(graph, execution_providers_, kernel_registry_manager_,
                                                             *session_state_));

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
      const auto& cpu_ep = *execution_providers_.Get(onnxruntime::kCpuExecutionProvider);
      ORT_RETURN_IF_ERROR_SESSIONID_(
          ApplyOrtFormatModelRuntimeOptimizations(graph, *session_logger_, session_options_, optimizers_to_disable_, cpu_ep));
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
    }

    ORT_RETURN_IF_ERROR_SESSIONID_(
        session_state_->FinalizeSessionState(model_location_, kernel_registry_manager_,
                                             // need to keep the initializers if saving the optimized model
                                             !saving_model,
                                             saving_ort_format));

#if !defined(ORT_MINIMAL_BUILD)
    if (saving_model) {
      if (session_state_->GetFuncMgr().NumFuncs() > 0) {
        ORT_RETURN_IF_ERROR_SESSIONID_(
            ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                            "Unable to serialize model as it contains compiled nodes. "
                            "Please disable any execution providers which generate compiled nodes."));
      }

      // add a warning if the NchwcTransformer was enabled, as it contains the hardware specific logic
      if (session_options_.graph_optimization_level >= TransformerLevel::Level3 &&
          optimizers_to_disable_.find("NchwcTransformer") == optimizers_to_disable_.cend()) {
        LOGS(*session_logger_, WARNING)
            << "Serializing optimized model with Graph Optimization level greater than ORT_ENABLE_EXTENDED and the "
               "NchwcTransformer enabled. The generated model may contain hardware specific optimizations, and "
               "should only be used in the same environment the model was optimized in.";
      }

      if (saving_ort_format) {
        ORT_RETURN_IF_ERROR_SESSIONID_(SaveToOrtFormat(session_options_.optimized_model_filepath));
      } else {
        const std::string optimized_model_external_initializers_file_name =
            session_options_.config_options.GetConfigOrDefault(
                kOrtSessionOptionsOptimizedModelExternalInitializersFileName, "");
        if (optimized_model_external_initializers_file_name.empty()) {
          ORT_RETURN_IF_ERROR_SESSIONID_(Model::Save(*model_, session_options_.optimized_model_filepath));
        } else {
          const size_t optimized_model_external_initializers_min_size_in_bytes =
              ParseStringWithClassicLocale<size_t>(session_options_.config_options.GetConfigOrDefault(
                  kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes, "1024"));
          ORT_RETURN_IF_ERROR_SESSIONID_(Model::SaveWithExternalInitializers(*model_,
                                                                             session_options_.optimized_model_filepath,
                                                                             optimized_model_external_initializers_file_name,
                                                                             optimized_model_external_initializers_min_size_in_bytes));
        }
      }
    }

    std::vector<TuningResults> tuning_results;
    bool found_tuning_results = false;
    ORT_RETURN_IF_ERROR_SESSIONID_(inference_session_utils::ParseTuningResultsFromModelMetadata(
        model_metadata_, tuning_results, found_tuning_results));
    if (found_tuning_results) {
      ORT_RETURN_IF_ERROR_SESSIONID_(SetTuningResults(tuning_results, /*error_on_invalid*/ false, /*auto_enable*/ true));
    }
#endif  // !defined(ORT_MINIMAL_BUILD)

    // Resolve memory pattern flags of the main graph and subgraph session states
    ResolveMemoryPatternFlags(*session_state_);

    is_inited_ = true;

    if (!using_ort_model_bytes_for_initializers_) {
      ort_format_model_bytes_ = gsl::span<const uint8_t>();
      std::vector<uint8_t>().swap(ort_format_model_bytes_data_holder_);
    }

    // once the model is saved, we may remove unnecessary attributes for inference
    session_state_->PruneRemovableAttributes();

    // and log telemetry
    bool model_has_fp16_inputs = ModelHasFP16Inputs(graph);
    env.GetTelemetryProvider().LogSessionCreation(
        session_id_, model_->IrVersion(), model_->ProducerName(), model_->ProducerVersion(), model_->Domain(),
        model_->MainGraph().DomainToVersionMap(), model_->MainGraph().Name(), model_->MetaData(),
        telemetry_.event_name_, execution_providers_.GetIds(), model_has_fp16_inputs);

    LOGS(*session_logger_, INFO) << "Session successfully initialized.";
  }
  ORT_CATCH(const NotImplementedException& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Exception during initialization: ", ex.what());
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    });
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Exception during initialization: ", ex.what());
      LOGS(*session_logger_, ERROR) << status.ErrorMessage();
    });
  }
  ORT_CATCH(...) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Encountered unknown exception in Initialize()");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
  }

  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "session_initialization", tp);
  }

  if (status.IsOK()) {
    for (auto& xp : execution_providers_) {
      auto end_status = xp->OnSessionInitializationEnd();
      if (status.IsOK()) {
        status = end_status;
      }
    }
  }

  return status;
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

int InferenceSession::GetCurrentNumRuns() const {
  return current_num_runs_.load();
}

const std::vector<std::string>& InferenceSession::GetRegisteredProviderTypes() const {
  return execution_providers_.GetIds();
}

const ProviderOptionsMap& InferenceSession::GetAllProviderOptions() const {
  return execution_providers_.GetAllProviderOptions();
}

const SessionOptions& InferenceSession::GetSessionOptions() const {
  return session_options_;
}

const DataTransferManager& InferenceSession::GetDataTransferManager() const {
  return data_transfer_mgr_;
}

common::Status InferenceSession::CheckShapes(const std::string& input_name, const TensorShape& input_shape,
                                             const TensorShape& expected_shape) const {
  auto input_shape_sz = input_shape.NumDimensions();
  auto expected_shape_sz = expected_shape.NumDimensions();
  if (input_shape_sz != expected_shape_sz) {
    std::ostringstream ostr;
    ostr << "Invalid rank for input: " << input_name << " Got: " << input_shape_sz << " Expected: " << expected_shape_sz
         << " Please fix either the inputs or the model.";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, ostr.str());
  }

  std::vector<size_t> invalid_dim_indices;
  for (size_t i = 0; i < input_shape_sz; ++i) {
    if (expected_shape[i] < 0) {
      continue;  // this represents a symbolic shape dimension
    }
    if (input_shape[i] != expected_shape[i]) {
      invalid_dim_indices.push_back(i);
    }
  }

  if (!invalid_dim_indices.empty()) {
    std::ostringstream ostr;
    ostr << "Got invalid dimensions for input: " << input_name << " for the following indices\n";
    for (size_t i = 0, end = invalid_dim_indices.size(); i < end; ++i) {
      size_t idx = invalid_dim_indices[i];
      ostr << " index: " << idx << " Got: " << input_shape[idx] << " Expected: " << expected_shape[idx] << "\n";
    }
    ostr << " Please fix either the inputs or the model.";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, ostr.str());
  }
  return Status::OK();
}

static common::Status CheckTypes(MLDataType actual, MLDataType expected, const std::string& base_type) {
  if (actual == expected) {
    return Status::OK();
  }
  std::ostringstream ostr;
  ostr << "Unexpected input data type. Actual: (";
  ostr << base_type;
  ostr << "(";
  ostr << DataTypeImpl::ToString(actual);
  ostr << ")) , expected: (";
  ostr << base_type;
  ostr << "(";
  ostr << DataTypeImpl::ToString(expected);
  ostr << "))";

  return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
}

common::Status InferenceSession::ValidateInputs(gsl::span<const std::string> feed_names,
                                                gsl::span<const OrtValue> feeds) const {
  if (feed_names.size() != feeds.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Size mismatch: feed_names has ", feed_names.size(),
                           "elements, but feeds has ", feeds.size(), " elements.");
  }

  for (size_t i = 0; i < feeds.size(); ++i) {
    const auto& feed_name = feed_names[i];

    auto iter = input_def_map_.find(feed_name);
    if (input_def_map_.end() == iter) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid Feed Input Name:", feed_name);
    }

    auto expected_type = iter->second.ml_data_type;
    auto& input_ml_value = feeds[i];
    if (input_ml_value.IsTensor()) {
      if (!expected_type->IsTensorType()
#if !defined(DISABLE_OPTIONAL_TYPE)
          && !utils::IsOptionalTensor(expected_type)
#endif
      ) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name: ", feed_name,
                               " is not expected to be of type tensor.");
      }

      // check for type
#if !defined(DISABLE_OPTIONAL_TYPE)
      auto expected_element_type = expected_type->IsTensorType()
                                       ? expected_type
                                             ->AsTensorType()
                                             ->GetElementType()
                                       : utils::GetElementTypeFromOptionalTensor(expected_type);
#else
      auto expected_element_type = expected_type->AsTensorType()->GetElementType();
#endif

      auto input_element_type = input_ml_value.Get<Tensor>().DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_element_type, expected_element_type, "tensor"));

      // check for shape
      const auto& expected_shape = iter->second.tensor_shape;
      if (expected_shape.NumDimensions() > 0) {
        const auto& input_shape = input_ml_value.Get<Tensor>().Shape();
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(feed_name, input_shape, expected_shape));
      }
    } else if (input_ml_value.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
      if (!expected_type->IsSparseTensorType()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name: ", feed_name,
                               " is not expected to be of type sparse tensor.");
      }
      auto expected_element_type = expected_type->AsSparseTensorType()->GetElementType();
      const SparseTensor& sparse_tensor = input_ml_value.Get<SparseTensor>();
      auto input_element_type = sparse_tensor.DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_element_type, expected_element_type, "sparse_tensor"));
      // Check shape
      const auto& expected_shape = iter->second.tensor_shape;
      if (expected_shape.NumDimensions() > 0) {
        const auto& input_shape = sparse_tensor.DenseShape();
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(feed_name, input_shape, expected_shape));
      }
#else
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name ", feed_name,
                             " is a sparse tensor, which is not supported in this build.");
#endif

    } else if (input_ml_value.IsTensorSequence()) {
      if (!expected_type->IsTensorSequenceType()
#if !defined(DISABLE_OPTIONAL_TYPE)
          && !utils::IsOptionalSeqTensor(expected_type)
#endif
      ) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name: ", feed_name,
                               " is not expected to be of type tensor sequence.");
      }

#if !defined(DISABLE_OPTIONAL_TYPE)
      auto expected_element_type = expected_type->IsTensorSequenceType()
                                       ? expected_type
                                             ->AsSequenceTensorType()
                                             ->GetElementType()
                                       : utils::GetElementTypeFromOptionalSeqTensor(expected_type);
#else
      auto expected_element_type = expected_type->AsSequenceTensorType()->GetElementType();
#endif

      auto input_element_type = input_ml_value.Get<TensorSeq>().DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_element_type, expected_element_type, "seq"));
    } else {
      auto input_type = input_ml_value.Type();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_type, expected_type, ""));
    }
  }

  return Status::OK();
}

common::Status InferenceSession::ValidateOutputs(gsl::span<const std::string> output_names,
                                                 const std::vector<OrtValue>* p_fetches) const {
  if (p_fetches == nullptr) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Output vector pointer is NULL");
  }

  if (output_names.empty()) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "At least one output should be requested.");
  }

  if (!p_fetches->empty() && (output_names.size() != p_fetches->size())) {
    std::ostringstream ostr;
    ostr << "Output vector incorrectly sized: output_names.size(): " << output_names.size()
         << "p_fetches->size(): " << p_fetches->size();
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
  }

  for (const auto& name : output_names) {
    if (model_output_names_.find(name) == model_output_names_.end()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid Output Name:" + name);
    }
  }

  // TODO add more validation here like checking shape of the allocated buffers

  return common::Status::OK();
}

#ifdef ENABLE_TRAINING
Status InferenceSession::PartialRun(onnxruntime::RunOptions& run_options,
                                    const std::vector<OrtValue>& feeds,
                                    std::vector<OrtValue>& fetches,
                                    PartialGraphExecutionState& state,
                                    FeedsFetchesManager& feeds_fetches_manager,
                                    const OrtValueCachePtr& cache,
                                    int32_t partial_graph_index) {
  Status retval = Status::OK();
  std::vector<IExecutionProvider*> exec_providers_to_stop;
  exec_providers_to_stop.reserve(execution_providers_.NumProviders());

  ORT_TRY {
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }

    if (!run_options.run_tag.empty()) {
      LOGS(*session_logger_, INFO) << "Running with tag: " << run_options.run_tag;
    }

    // scope of owned_run_logger is just the call to Execute.
    // If Execute ever becomes async we need a different approach
    std::unique_ptr<logging::Logger> owned_run_logger;
    auto run_logger = CreateLoggerForRun(run_options, owned_run_logger);

    // info all execution providers InferenceSession:Run started
    // TODO: only call OnRunStart for all providers in-use
    for (auto& xp : execution_providers_) {
      // call OnRunStart and add to exec_providers_to_stop if successful
      auto start_func = [&xp, &exec_providers_to_stop]() {
        auto status = xp->OnRunStart();
        if (status.IsOK())
          exec_providers_to_stop.push_back(xp.get());

        return status;
      };

      ORT_CHECK_AND_SET_RETVAL(start_func());
    }

    ORT_ENFORCE(run_options.only_execute_path_to_fetches == false, "only_execute_path_to_fetches is not supported.");

    ORT_ENFORCE(session_options_.execution_mode == ExecutionMode::ORT_SEQUENTIAL, "Only sequential mode is supported.");

    // execute the graph
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
    if (state.GetProgramCounterStart() == 0) {
      session_state_->IncrementGraphExecutionCounter();
    }
#endif
    ORT_CHECK_AND_SET_RETVAL(utils::ExecutePartialGraph(*session_state_, feeds_fetches_manager, feeds, fetches,
                                                        run_logger, state, cache, run_options.terminate,
                                                        partial_graph_index,
                                                        /*parent stream*/ nullptr));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      retval = Status(common::ONNXRUNTIME, common::FAIL, e.what());
    });
  }
  ORT_CATCH(...) {
    retval = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
  }

  // info all execution providers InferenceSession:Run ended
  for (auto* xp : exec_providers_to_stop) {
    auto status = xp->OnRunEnd(/*sync_stream*/ false);
    ORT_CHECK_AND_SET_RETVAL(status);
  }

  return retval;
}
#endif

namespace {
// Concurrent runs counting and thread-pool spin control
struct ThreadPoolSpinningSwitch {
  concurrency::ThreadPool* intra_tp_{nullptr};
  concurrency::ThreadPool* inter_tp_{nullptr};
  std::atomic<int>& concurrent_num_runs_;
  // __Ctor Refcounting and spinning control
  ThreadPoolSpinningSwitch(concurrency::ThreadPool* intra_tp,
                           concurrency::ThreadPool* inter_tp,
                           std::atomic<int>& ref) noexcept
      : intra_tp_(intra_tp), inter_tp_(inter_tp), concurrent_num_runs_(ref) {
    if (concurrent_num_runs_.fetch_add(1, std::memory_order_relaxed) == 0) {
      if (intra_tp_) intra_tp_->EnableSpinning();
      if (inter_tp_) inter_tp_->EnableSpinning();
    }
  }
  ~ThreadPoolSpinningSwitch() {
    if (1 == concurrent_num_runs_.fetch_sub(1, std::memory_order_acq_rel)) {
      if (intra_tp_) intra_tp_->DisableSpinning();
      if (inter_tp_) inter_tp_->DisableSpinning();
    }
  }
};
}  // namespace

Status InferenceSession::Run(const RunOptions& run_options,
                             gsl::span<const std::string> feed_names, gsl::span<const OrtValue> feeds,
                             gsl::span<const std::string> output_names, std::vector<OrtValue>* p_fetches,
                             const std::vector<OrtDevice>* p_fetches_device_info) {
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.Start();
  }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingActivity<telemetry_provider_handle> ortrun_activity;
  ortrun_activity.SetRelatedActivity(session_activity);
  TraceLoggingWriteStart(ortrun_activity, "OrtRun");
#endif
  Status retval = Status::OK();
  const Env& env = Env::Default();

  // Increment/decrement concurrent_num_runs_ and control
  // session threads spinning as configured. Do nothing for graph replay except the counter.
  const bool control_spinning = use_per_session_threads_ &&
                                force_spinning_stop_between_runs_ &&
                                !cached_execution_provider_for_graph_replay_.IsGraphCaptured();
  auto* intra_tp = (control_spinning) ? thread_pool_.get() : nullptr;
  auto* inter_tp = (control_spinning) ? inter_op_thread_pool_.get() : nullptr;
  ThreadPoolSpinningSwitch runs_refcounter_and_tp_spin_control(intra_tp, inter_tp, current_num_runs_);

  // Check if this Run() is simply going to be a CUDA Graph replay.
  if (cached_execution_provider_for_graph_replay_.IsGraphCaptured()) {
    LOGS(*session_logger_, INFO) << "Replaying the captured "
                                 << cached_execution_provider_for_graph_replay_.Type()
                                 << " CUDA Graph for this model with tag: " << run_options.run_tag;
    ORT_RETURN_IF_ERROR_SESSIONID_(cached_execution_provider_for_graph_replay_.ReplayGraph());
  } else {
    InlinedVector<IExecutionProvider*> exec_providers_to_stop;
    exec_providers_to_stop.reserve(execution_providers_.NumProviders());

    InlinedVector<AllocatorPtr> arenas_to_shrink;

    ORT_TRY {
      if (!is_inited_) {
        LOGS(*session_logger_, ERROR) << "Session was not initialized";
        return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
      }

      // log evaluation start to trace logging provider
      env.GetTelemetryProvider().LogEvaluationStart();

      ORT_RETURN_IF_ERROR_SESSIONID_(ValidateInputs(feed_names, feeds));
      ORT_RETURN_IF_ERROR_SESSIONID_(ValidateOutputs(output_names, p_fetches));

      // shrink certain default memory arenas if the user has requested for it
      const std::string& shrink_memory_arenas =
          run_options.config_options.GetConfigOrDefault(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "");

      if (!shrink_memory_arenas.empty()) {
        ORT_RETURN_IF_ERROR_SESSIONID_(ValidateAndParseShrinkArenaString(shrink_memory_arenas, arenas_to_shrink));
      }

      FeedsFetchesInfo info(feed_names, output_names, session_state_->GetOrtValueNameIdxMap());
      FeedsFetchesManager feeds_fetches_manager{std::move(info)};

      if (p_fetches_device_info) {
        // populate the target device info. ignored if pre-allocated fetches are provided
        const auto& fetch_device_info = *p_fetches_device_info;
        auto& fetch_info = feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo();

        for (size_t i = 0, end = output_names.size(); i < end; ++i) {
          fetch_info[i].target_device = fetch_device_info[i];
        }
      }

      if (!run_options.run_tag.empty()) {
        LOGS(*session_logger_, INFO) << "Running with tag: " << run_options.run_tag;
      }

      // scope of owned_run_logger is just the call to Execute.
      // If Execute ever becomes async we need a different approach
      std::unique_ptr<logging::Logger> owned_run_logger;
      const auto& run_logger = CreateLoggerForRun(run_options, owned_run_logger);

      std::optional<std::lock_guard<OrtMutex>> sequential_run_lock;
      if (is_concurrent_run_supported_ == false) {
        sequential_run_lock.emplace(session_mutex_);
      }

      // info all execution providers InferenceSession:Run started
      // TODO: only call OnRunStart for all providers in-use
      for (auto& xp : execution_providers_) {
        // call OnRunStart and add to exec_providers_to_stop if successful
        auto start_func = [&xp, &exec_providers_to_stop]() {
          auto status = xp->OnRunStart();
          if (status.IsOK())
            exec_providers_to_stop.push_back(xp.get());

          return status;
        };

        ORT_CHECK_AND_SET_RETVAL(start_func());
      }

#ifdef ENABLE_TRAINING
      if (run_options.only_execute_path_to_fetches) {
        // TODO: this method is not thread safe, if multiple Run happened in parallel we might hit race condition issue.
        // currently it only used in training, there is no parallel run execution in training so it is ok.
        // but it is better we can fix it with a better solution.
        session_state_->UpdateToBeExecutedRange(feeds_fetches_manager.GetFeedsFetchesInfo().fetches_mlvalue_idxs);
      }
#endif

      // execute the graph
#ifdef DEBUG_NODE_INPUTS_OUTPUTS
      session_state_->IncrementGraphExecutionCounter();
#endif

#ifdef ORT_ENABLE_STREAM
      DeviceStreamCollectionHolder device_stream_collection_holder(session_state_.get());
#endif

      if (retval.IsOK()) {
        retval = utils::ExecuteGraph(*session_state_, feeds_fetches_manager, feeds, *p_fetches,
                                     session_options_.execution_mode,
                                     run_options,
#ifdef ORT_ENABLE_STREAM
                                     device_stream_collection_holder,
#endif
                                     run_logger);
      }

      // info all execution providers InferenceSession:Run ended
      for (auto* xp : exec_providers_to_stop) {
        bool synchronize_execution_providers = run_options.config_options.GetConfigOrDefault(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "0") == "0";
        auto status = xp->OnRunEnd(synchronize_execution_providers);
        ORT_CHECK_AND_SET_RETVAL(status);
      }

      // Move stream cleanup from ExecuteGraph to here for cuda graph capture.
      // Cleanup will call cudaStreamSyncronize, which is not allowed for graph capture.
      // Note that graph capture ends when we call xp->OnRunEnd() in the above code so it is safe here.
#ifdef ORT_ENABLE_STREAM
      DeviceStreamCollection* device_stream_collection = device_stream_collection_holder.p_.get();
      if (device_stream_collection) {
        bool sync_execution_provider = run_options.config_options.GetConfigOrDefault(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "0") == "0";
        ORT_CHECK_AND_SET_RETVAL(device_stream_collection->CleanUp(sync_execution_provider));
      }
#endif
    }
    ORT_CATCH(const std::exception& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        retval = Status(common::ONNXRUNTIME, common::FAIL, e.what());
      });
    }
    ORT_CATCH(...) {
      retval = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
    }

    if (!arenas_to_shrink.empty()) {
      ShrinkMemoryArenas(arenas_to_shrink);
    }
  }

  // keep track of telemetry
  ++telemetry_.total_runs_since_last_;
  telemetry_.total_run_duration_since_last_ += TimeDiffMicroSeconds(tp);

  // time to send telemetry?
  if (TimeDiffMicroSeconds(telemetry_.time_sent_last_) > Telemetry::kDurationBetweenSending) {
    // send the telemetry
    env.GetTelemetryProvider().LogRuntimePerf(session_id_, telemetry_.total_runs_since_last_,
                                              telemetry_.total_run_duration_since_last_);
    // reset counters
    telemetry_.time_sent_last_ = std::chrono::high_resolution_clock::now();
    telemetry_.total_runs_since_last_ = 0;
    telemetry_.total_run_duration_since_last_ = 0;
  }

  // log evaluation stop to trace logging provider
  env.GetTelemetryProvider().LogEvaluationStop();

  // send out profiling events (optional)
  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "model_run", tp);
  }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingWriteStop(ortrun_activity, "OrtRun");
#endif

  // As N+1 inference runs (N for memory allocation and 1 for graph capturing)
  // are needed before replaying the captured graph, here run N inference runs recursively until graph captured,
  // so that users just need one session run to capture the graph.
  // N is defined in min_num_runs_before_cuda_graph_capture_ for CUDA EP, and the value could be different for other EP.
  if (retval.IsOK() && cached_execution_provider_for_graph_replay_.IsGraphCaptureEnabled() &&
      !cached_execution_provider_for_graph_replay_.IsGraphCaptured()) {
    LOGS(*session_logger_, INFO) << "Start another run for necessary memory allocation or graph capture.";
    ORT_RETURN_IF_ERROR(Run(run_options, feed_names, feeds, output_names, p_fetches, p_fetches_device_info));
  }
  return retval;
}

Status InferenceSession::Run(const RunOptions& run_options,
                             gsl::span<const char* const> feed_names,
                             gsl::span<const OrtValue* const> feeds,
                             gsl::span<const char* const> fetch_names,
                             gsl::span<OrtValue*> fetches) {
  size_t num_feeds = feed_names.size();
  size_t num_fetches = fetch_names.size();
  InlinedVector<std::string> feed_name_vec;
  feed_name_vec.reserve(num_feeds);
  InlinedVector<OrtValue> feed_vec;
  feed_vec.reserve(num_feeds);

  for (size_t i = 0; i != num_feeds; ++i) {
    if (feed_names[i] == nullptr || feed_names[i][0] == '\0') {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input name cannot be empty");
    }

    if (!feeds[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, MakeString("NULL input supplied for input ", feed_names[i]).c_str());
    }

    feed_name_vec.emplace_back(feed_names[i]);
    feed_vec.emplace_back(*feeds[i]);
  }

  // Create output feed
  InlinedVector<std::string> fetch_name_vec;
  fetch_name_vec.reserve(num_fetches);
  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetch_names[i] == nullptr || fetch_names[i][0] == '\0') {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "output name cannot be empty");
    }
    fetch_name_vec.emplace_back(fetch_names[i]);
  }

  std::vector<OrtValue> fetch_vec;
  fetch_vec.reserve(num_fetches);
  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetches[i] != nullptr) {
      fetch_vec.emplace_back(*fetches[i]);
    } else {
      fetch_vec.emplace_back();
    }
  }

  Status status;
  status = Run(run_options, feed_name_vec, feed_vec, fetch_name_vec, &fetch_vec, nullptr);

  if (!status.IsOK())
    return status;

  // We do it in two loops to make sure copy __ctors does not throw
  InlinedVector<std::unique_ptr<OrtValue>> fetch_unique_ptrs;
  fetch_unique_ptrs.reserve(num_fetches);
  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetches[i] == nullptr) {
      fetch_unique_ptrs.emplace_back(std::make_unique<OrtValue>(fetch_vec[i]));
    } else {
      fetch_unique_ptrs.emplace_back();
    }
  }

  for (size_t i = 0; i != num_fetches; ++i) {
    if (fetches[i] == nullptr) {
      ORT_ENFORCE(fetch_unique_ptrs[i] != nullptr);
      fetches[i] = fetch_unique_ptrs[i].release();
    }
  }
  return Status::OK();
}

common::Status InferenceSession::RunAsync(const RunOptions* run_options,
                                          gsl::span<const char* const> feed_names,
                                          gsl::span<const OrtValue* const> feeds,
                                          gsl::span<const char* const> fetch_names,
                                          gsl::span<OrtValue*> fetches,
                                          RunAsyncCallbackFn callback,
                                          void* user_data) {
  size_t num_fetches = fetch_names.size();
  if (!thread_pool_.get() || concurrency::ThreadPool::DegreeOfParallelism(thread_pool_.get()) < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "intra op thread pool must have at least one thread for RunAsync");
  }
  std::function<void()> run_fn = [=]() {
    ORT_TRY {
      Status status;
      if (run_options) {
        status = Run(*run_options, feed_names, feeds, fetch_names, fetches);
      } else {
        RunOptions default_run_options;
        status = Run(default_run_options, feed_names, feeds, fetch_names, fetches);
      }
      if (status.IsOK()) {
        callback(user_data, fetches.data(), num_fetches, ToOrtStatus(status));
      } else {
        callback(user_data, {}, 0, ToOrtStatus(status));
      }
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([=]() {
        callback(user_data, {}, 0, ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, ex.what())));
      });
    }
    ORT_CATCH(...) {
      callback(user_data, {}, 0, ToOrtStatus(ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "unknown exception")));
    }
  };  // run_fn
  concurrency::ThreadPool::Schedule(thread_pool_.get(), run_fn);
  return Status::OK();
}

common::Status InferenceSession::Run(const NameMLValMap& feeds, gsl::span<const std::string> output_names,
                                     std::vector<OrtValue>* p_fetches) {
  return Run(RunOptions(), feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const RunOptions& run_options, const NameMLValMap& feeds_map,
                                     gsl::span<const std::string> output_names, std::vector<OrtValue>* p_fetches) {
  InlinedVector<std::string> feed_names;
  InlinedVector<OrtValue> feeds;

  const auto num_feeds = feeds_map.size();
  feed_names.reserve(num_feeds);
  feeds.reserve(num_feeds);

  for (auto& pair : feeds_map) {
    feed_names.push_back(pair.first);
    feeds.push_back(pair.second);
  }

  return Run(run_options, feed_names, feeds, output_names, p_fetches, nullptr);
}

std::pair<common::Status, const ModelMetadata*> InferenceSession::GetModelMetadata() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  return std::make_pair(common::Status::OK(), &model_metadata_);
}

std::pair<common::Status, const InputDefList*> InferenceSession::GetModelInputs() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  // return required inputs (excludes any inputs used for overriding initializers)
  return std::make_pair(common::Status::OK(), &model_->MainGraph().GetInputs());
}

std::pair<common::Status, const InputDefList*> InferenceSession::GetOverridableInitializers() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  // returns a list of initializers that can be overriden.
  return std::make_pair(common::Status::OK(), &model_->MainGraph().GetOverridableInitializers());
}

std::pair<common::Status, const OutputDefList*> InferenceSession::GetModelOutputs() const {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."), nullptr);
    }
  }

  return std::make_pair(common::Status::OK(), &output_def_list_);
}

common::Status InferenceSession::NewIOBinding(std::unique_ptr<IOBinding>* io_binding) {
  {
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }
  }

  *io_binding = std::make_unique<IOBinding>(*session_state_);
  return Status::OK();
}

common::Status InferenceSession::Run(const RunOptions& run_options, IOBinding& io_binding) {
  // TODO should Run() call io_binding.SynchronizeInputs() or should it let the callers do it?
  // io_binding.SynchronizeInputs();
  return Run(run_options, io_binding.GetInputNames(), io_binding.GetInputs(), io_binding.GetOutputNames(),
             &io_binding.GetOutputs(), &io_binding.GetOutputsDeviceInfo());
}

common::Status InferenceSession::Run(IOBinding& io_binding) {
  RunOptions run_options;
  return Run(run_options, io_binding);
}

template <typename T>
void InferenceSession::StartProfiling(const std::basic_string<T>& file_prefix) {
  std::basic_ostringstream<T> ss;
  ss << file_prefix << "_" << GetCurrentTimeString<T>() << ".json";
  session_profiler_.StartProfiling(ss.str());
}

void InferenceSession::StartProfiling(const std::string& file_prefix) {
  StartProfiling<char>(file_prefix);
}

#ifdef _WIN32
void InferenceSession::StartProfiling(const std::wstring& file_prefix) {
  StartProfiling<PATH_CHAR_TYPE>(file_prefix);
}
#endif

void InferenceSession::StartProfiling(const logging::Logger* logger_ptr) {
  session_profiler_.StartProfiling(logger_ptr);
}

std::string InferenceSession::EndProfiling() {
  if (is_model_loaded_) {
    if (session_profiler_.IsEnabled()) {
      return session_profiler_.EndProfiling();
    } else {
      LOGS(*session_logger_, VERBOSE) << "Profiler is disabled.";
      return std::string();
    }
  }
  LOGS(*session_logger_, ERROR) << "Could not write a profile because no model was loaded.";
  return std::string();
}

const profiling::Profiler& InferenceSession::GetProfiling() const {
  return session_profiler_;
}

#if !defined(ORT_MINIMAL_BUILD)
std::vector<TuningResults> InferenceSession::GetTuningResults() const {
  std::vector<TuningResults> ret;
  for (const auto& provider : execution_providers_) {
    const auto* tuning_ctx = provider->GetTuningContext();
    if (tuning_ctx != nullptr) {
      ret.emplace_back(tuning_ctx->GetTuningResults());
    }
  }
  return ret;
}

Status InferenceSession::SetTuningResults(
    const std::vector<TuningResults>& trs,
    bool error_on_invalid,
    bool auto_enable) {
  std::string msg;

  for (size_t i = 0; i < trs.size(); i++) {
    const auto& tr = trs[i];
    auto* provider = execution_providers_.Get(tr.ep);
    if (provider == nullptr) {
      msg = MakeString("Cannot find execution provider ", tr.ep);
      ORT_RETURN_IF(error_on_invalid, msg);
      LOGS(*session_logger_, WARNING) << msg;
      continue;
    }

    auto* tuning_ctx = provider->GetTuningContext();
    if (tuning_ctx == nullptr) {
      msg = MakeString("Invalid TuningResults (index=", i, "). ", tr.ep, " does not support TunableOp.");
      ORT_RETURN_IF(error_on_invalid, msg);
      LOGS(*session_logger_, WARNING) << msg;
      continue;
    }

    auto status = tuning_ctx->LoadTuningResults(tr);
    if (!status.IsOK()) {
      msg = MakeString("Failed to load TuningResults (index=", i, "). Reason: ", status.ErrorMessage());
      ORT_RETURN_IF(error_on_invalid, msg);
      LOGS(*session_logger_, WARNING) << msg;
      continue;
    }

    if (auto_enable) {
      LOGS(*session_logger_, INFO) << "Correctly set TuningResults for " << tr.ep << ", enable TunableOp for using";
      tuning_ctx->EnableTunableOp();
    }
  }
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

AllocatorPtr InferenceSession::GetAllocator(const OrtMemoryInfo& mem_info) const {
  return session_state_->GetAllocator(mem_info);
}

common::Status InferenceSession::ValidateAndParseShrinkArenaString(const std::string& ort_device_list,
                                                                   /*out*/ InlinedVector<AllocatorPtr>& arenas_to_shrink) const {
  arenas_to_shrink.reserve(5);  // Allocate some memory for the container (we are unlikely to see more than 5 memory arena shrink requests)

  std::istringstream ss_1(ort_device_list);
  std::string device_id_pair;

  // Process all device-id pair(s)
  while (std::getline(ss_1, device_id_pair, ';')) {
    std::istringstream ss_2(device_id_pair);
    std::string device_id_component;

    // default values
    OrtDevice::DeviceType device_type = -1;
    OrtDevice::MemoryType memory_type = OrtDevice::MemType::DEFAULT;
    OrtDevice::DeviceId device_id = 0;

    int iter = 0;
    // Process this device-id pair
    while (std::getline(ss_2, device_id_component, ':')) {
      if (iter == 0) {  // this component corresponds to device
        if (device_id_component == "cpu") {
          device_type = OrtDevice::CPU;
        } else if (device_id_component == "gpu") {
          device_type = OrtDevice::GPU;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported device specified in the memory arena shrink list: ",
                                 device_id_component);
        }
      } else if (iter == 1) {  // This component corresponds to device id
        if (!TryParseStringWithClassicLocale<OrtDevice::DeviceId>(device_id_component, device_id)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported device id in the memory arena shrink list: ",
                                 device_id_component);
        }
      }

      ++iter;
    }

    // Shrink if it is an arena based allocator
    auto alloc = session_state_->GetAllocator(OrtDevice(device_type, memory_type, device_id));

    if (alloc == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Did not find an arena based allocator registered for device-id ",
                             " combination in the memory arena shrink list: ", device_id_pair);
    }

    if (alloc->Info().alloc_type != OrtAllocatorType::OrtArenaAllocator) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The registered allocator for device-id ",
                             " combination is not an arena based allocator: ", device_id_pair);
    }

    arenas_to_shrink.push_back(std::move(alloc));
  }

  return Status::OK();
}

void InferenceSession::ShrinkMemoryArenas(gsl::span<const AllocatorPtr> arenas_to_shrink) {
  for (auto& alloc : arenas_to_shrink) {
    auto status = static_cast<BFCArena*>(alloc.get())->Shrink();

    if (!status.IsOK()) {
      LOGS(*session_logger_, WARNING) << "Unable to shrink arena: " << alloc->Info().ToString()
                                      << " error message: " << status.ErrorMessage();
    }
  }
}

#if !defined(ORT_MINIMAL_BUILD)
// assumes model has already been loaded before
common::Status InferenceSession::DoPostLoadProcessing(onnxruntime::Model& model) {
  // TODO add other post load processing here
  common::Status status = SaveModelMetadata(model);
  return status;
}
#endif

common::Status InferenceSession::SaveModelMetadata(const onnxruntime::Model& model) {
  VLOGS(*session_logger_, 1) << "Saving model metadata";
  const onnxruntime::Graph& graph = model.MainGraph();

  // save model metadata
  model_metadata_.producer_name = model.ProducerName();
  model_metadata_.description = model.DocString();
  model_metadata_.graph_description = model.GraphDocString();
  model_metadata_.domain = model.Domain();
  model_metadata_.version = model.ModelVersion();
  model_metadata_.custom_metadata_map = model.MetaData();
  model_metadata_.graph_name = graph.Name();

  required_inputs_.clear();
  for (auto input : graph.GetInputs()) {
    required_inputs_.insert(input->Name());
  }

  auto add_inputs = [this](const InputDefList& inputs) {
    input_def_map_.clear();
    input_def_map_.reserve(inputs.size());
    for (auto elem : inputs) {
      auto elem_type = utils::GetMLDataType(*elem);
      auto elem_shape_proto = elem->Shape();
      input_def_map_.insert(
          {elem->Name(),
           InputDefMetaData(
               elem, elem_type,
               elem_shape_proto ? utils::GetTensorShapeFromTensorShapeProto(*elem_shape_proto) : TensorShape())});
    }
  };

  if (graph.CanOverrideInitializer()) {
    // for IR 4 or higher it is optional to have a matching graph input for an initializer, and if one exists the
    // initializer is explicitly overridable.
    add_inputs(graph.GetInputsIncludingInitializers());
  } else {
    // for IR < 4 we don't allow overriding initializers so that they can be treated as constant. exclude them from
    // the list of valid inputs by just using the GetInputs() list.
    add_inputs(graph.GetInputs());
  }

  // save outputs
  const auto& outputs = graph.GetOutputs();
  output_def_list_ = outputs;  // A direct copy of outputs

  model_output_names_.clear();
  model_output_names_.reserve(outputs.size());
  for (const auto& elem : outputs) {
    model_output_names_.insert(elem->Name());
  }

  VLOGS(*session_logger_, 1) << "Done saving model metadata";
  return common::Status::OK();
}

// Create a Logger for a single execution if possible. Otherwise use the default logger.
// If a new logger is created, it will also be stored in new_run_logger,
// which must remain valid for the duration of the execution.
// If the default logger is used, new_run_logger will remain empty.
// The returned value should be used in the execution.
const logging::Logger& InferenceSession::CreateLoggerForRun(const RunOptions& run_options,
                                                            std::unique_ptr<logging::Logger>& new_run_logger) {
  const logging::Logger* run_logger;

  // create a per-run logger if we can
  if (logging_manager_ != nullptr) {
    std::string run_log_id{session_options_.session_logid};

    if (!session_options_.session_logid.empty() && !run_options.run_tag.empty()) {
      run_log_id += ":";
    }

    run_log_id += run_options.run_tag;

    logging::Severity severity = logging::Severity::kWARNING;
    if (run_options.run_log_severity_level == -1) {
      severity = session_logger_->GetSeverity();
    } else {
      ORT_ENFORCE(run_options.run_log_severity_level >= 0 &&
                      run_options.run_log_severity_level <= static_cast<int>(logging::Severity::kFATAL),
                  "Invalid run log severity level. Not a valid onnxruntime::logging::Severity value: ",
                  run_options.run_log_severity_level);
      severity = static_cast<logging::Severity>(run_options.run_log_severity_level);
    }

    new_run_logger = logging_manager_->CreateLogger(run_log_id, severity, false, run_options.run_log_verbosity_level);

    run_logger = new_run_logger.get();
    VLOGS(*run_logger, 1) << "Created logger for run with id of " << run_log_id;
  } else {
    // fallback to using default logger. this does NOT have any session or run specific id/tag in it
    run_logger = session_logger_;
    VLOGS(*run_logger, 1) << "Using default logger for run " << run_options.run_tag;
  }

  return *run_logger;
}

void InferenceSession::InitLogger(logging::LoggingManager* logging_manager) {
  // create logger for session, using provided logging manager if possible
  if (logging_manager != nullptr) {
    logging::Severity severity = logging::Severity::kWARNING;
    if (session_options_.session_log_severity_level == -1) {
      severity = logging::LoggingManager::DefaultLogger().GetSeverity();
    } else {
      ORT_ENFORCE(session_options_.session_log_severity_level >= 0 &&
                      session_options_.session_log_severity_level <= static_cast<int>(logging::Severity::kFATAL),
                  "Invalid session log severity level. Not a valid onnxruntime::logging::Severity value: ",
                  session_options_.session_log_severity_level);
      severity = static_cast<logging::Severity>(session_options_.session_log_severity_level);
    }

    owned_session_logger_ = logging_manager_->CreateLogger(session_options_.session_logid, severity, false,
                                                           session_options_.session_log_verbosity_level);
    session_logger_ = owned_session_logger_.get();
  } else {
    session_logger_ = &logging::LoggingManager::DefaultLogger();
  }
}

#if !defined(ORT_MINIMAL_BUILD)

// Registers all the predefined transformers with transformer manager
common::Status InferenceSession::AddPredefinedTransformers(
    GraphTransformerManager& transformer_manager,
    TransformerLevel graph_optimization_level,
    MinimalBuildOptimizationHandling minimal_build_optimization_handling,
    RecordRuntimeOptimizationProducedNodeOpSchemaFn record_runtime_optimization_produced_op_schema_fn) const {
  const auto& cpu_ep = *execution_providers_.Get(onnxruntime::kCpuExecutionProvider);
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if (graph_optimization_level >= level) {
      // Generate and register transformers for level
      auto transformers_to_register = [&]() {
        const bool use_full_build_optimizations =
            level == TransformerLevel::Level1 ||
            minimal_build_optimization_handling == MinimalBuildOptimizationHandling::ApplyFullBuildOptimizations;

        if (use_full_build_optimizations) {
          return optimizer_utils::GenerateTransformers(level, session_options_, cpu_ep,
                                                       optimizers_to_disable_);
        } else {
          const auto sat_context =
              minimal_build_optimization_handling ==
                      MinimalBuildOptimizationHandling::SaveMinimalBuildRuntimeOptimizations
                  ? SatApplyContextVariant{SatRuntimeOptimizationSaveContext{
                        record_runtime_optimization_produced_op_schema_fn}}
                  : SatApplyContextVariant{SatDirectApplicationContext{}};
          return optimizer_utils::GenerateTransformersForMinimalBuild(level, session_options_, sat_context, cpu_ep,
                                                                      optimizers_to_disable_);
        }
      }();

      for (auto& entry : transformers_to_register) {
        ORT_RETURN_IF_ERROR(transformer_manager.Register(std::move(entry), level));
      }
    }
  }
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

common::Status InferenceSession::WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms) {
  if (timeout_in_ms > 0) {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, "timeout_in_ms >0 is not supported");  // TODO
  }
  p_executor_done->Wait();

  return Status::OK();
}

SessionIOBinding::SessionIOBinding(InferenceSession* session) : sess_(session) {
  ORT_ENFORCE(session->NewIOBinding(&binding_).IsOK());
}

const InferenceSession* SessionIOBinding::GetInferenceSession() const {
  return sess_;
}

InferenceSession* SessionIOBinding::GetInferenceSession() {
  return sess_;
}

const IOBinding* SessionIOBinding::Get() const {
  return binding_.get();
}

IOBinding* SessionIOBinding::Get() {
  return binding_.get();
}

}  // namespace onnxruntime
