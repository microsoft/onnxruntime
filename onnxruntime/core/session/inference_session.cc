// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/session/inference_session.h"

#include <memory>
#include <sstream>
#include <unordered_set>
#include <list>
#include <string>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/platform/notification.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/threadpool.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/customregistry.h"
#include "core/session/environment.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_frame.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/session_state_initializer.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/utils.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#ifdef USE_DML  // TODO: This is necessary for the workaround in TransformGraph
#include "core/providers/dml/DmlExecutionProvider/src/GraphTransformer.h"
#endif
#include "core/session/IOBinding.h"
#include "core/session/custom_ops.h"
#include "core/util/protobuf_parsing_utils.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/util/thread_utils.h"

using namespace ONNX_NAMESPACE;

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

}  // namespace

std::atomic<uint32_t> InferenceSession::global_session_id_{1};

InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   logging::LoggingManager* logging_manager)
    : session_options_(session_options),
      graph_transformation_mgr_(session_options.max_num_graph_transformation_steps),
      logging_manager_(logging_manager),
      thread_pool_(concurrency::CreateThreadPool("intra_op_thread_pool",
                                                 session_options.intra_op_num_threads)),
      inter_op_thread_pool_(session_options.execution_mode == ExecutionMode::ORT_PARALLEL
                                ? concurrency::CreateThreadPool("inter_op_thread_pool",
                                                                session_options.inter_op_num_threads)
                                : nullptr),
      session_state_(execution_providers_,
                     session_options.enable_mem_pattern && session_options.execution_mode == ExecutionMode::ORT_SEQUENTIAL,
                     thread_pool_.get(),
                     inter_op_thread_pool_.get()),
      insert_cast_transformer_("CastFloat16Transformer") {
  ORT_ENFORCE(Environment::IsInitialized(),
              "Environment must be initialized before creating an InferenceSession.");
  InitLogger(logging_manager);

  session_state_.SetDataTransferMgr(&data_transfer_mgr_);
  session_profiler_.Initialize(session_logger_);
  session_state_.SetProfiler(session_profiler_);
  if (session_options.enable_profiling) {
    StartProfiling(session_options.profile_file_prefix);
  }

  // a monotonically increasing session id for use in telemetry
  session_id_ = global_session_id_.fetch_add(1);
}

InferenceSession::~InferenceSession() {
  if (session_options_.enable_profiling) {
    try {
      EndProfiling();
    } catch (std::exception& e) {
      // TODO: Currently we have no way to transport this error to the API user
      // Maybe this should be refactored, so that profiling must be explicitly
      // started and stopped via C-API functions.
      // And not like now a session option and therefore profiling must be started
      // and stopped implicitly.
      LOGS(*session_logger_, ERROR) << "Error during EndProfiling(): " << e.what();
    } catch (...) {
      LOGS(*session_logger_, ERROR) << "Unknown error during EndProfiling()";
    }
  }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  if (session_activity_started_) TraceLoggingWriteStop(session_activity, "OrtInferenceSessionActivity");
#endif
}

common::Status InferenceSession::RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) {
  if (p_exec_provider == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for exec provider");
  }

  const std::string& provider_type = p_exec_provider->Type();

  // DML's memory is not byte addressable and hence mem pattern doesn't work.
  if (provider_type == onnxruntime::kDmlExecutionProvider) {
    if (session_options_.enable_mem_pattern) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Memory pattern must be disabled before registering DMLExecutionProvider");
    }
    if (session_options_.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Sequential execution must be enabled before registering DMLExecutionProvider");
    }
  }

  VLOGS(*session_logger_, 1) << "Adding execution provider of type: " << provider_type;
  auto p_data_xfr = p_exec_provider->GetDataTransfer();
  if (p_data_xfr) {
    auto st = data_transfer_mgr_.RegisterDataTransfer(std::move(p_data_xfr));
    if (!st.IsOK()) {
      return st;
    }
  }
  execution_providers_.Add(provider_type, std::move(p_exec_provider));

  return Status::OK();
}

common::Status InferenceSession::RegisterGraphTransformer(
    std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer, TransformerLevel level) {
  if (p_graph_transformer == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for graph transformer");
  }
  return graph_transformation_mgr_.Register(std::move(p_graph_transformer), level);
}

common::Status InferenceSession::AddCustomTransformerList(const std::vector<std::string>& transformers_to_enable) {
  std::copy(transformers_to_enable.begin(), transformers_to_enable.end(), std::back_inserter(transformers_to_enable_));

  return Status::OK();
}

common::Status InferenceSession::AddCustomOpDomains(const std::vector<OrtCustomOpDomain*>& op_domains) {
  std::shared_ptr<CustomRegistry> custom_registry;
  ORT_RETURN_IF_ERROR_SESSIONID_(CreateCustomRegistry(op_domains, custom_registry));
  RegisterCustomRegistry(custom_registry);
  return Status::OK();
}

common::Status InferenceSession::RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
  if (custom_registry == nullptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Received nullptr for custom registry");
  }

  custom_registries_.push_back(custom_registry);

  // Insert session-level customized kernel registry.
  kernel_registry_manager_.RegisterKernelRegistry(custom_registry->GetKernelRegistry());
  //    if (custom_schema_registries_.empty())
  //      custom_schema_registries_.push_back();
  custom_schema_registries_.push_back(custom_registry->GetOpschemaRegistry());
  return Status::OK();
}

common::Status InferenceSession::Load(std::function<common::Status(std::shared_ptr<Model>&)> loader,
                                      const std::string& event_name) {
  Status status = Status::OK();
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.StartTime();
  }
  try {
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

    // and log telemetry
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogSessionCreation(session_id_, model_->IrVersion(), model_->ProducerName(), model_->ProducerVersion(),
                                                  model_->Domain(), model_->MainGraph().DomainToVersionMap(), model_->MainGraph().Name(),
                                                  model_->MetaData(), event_name, execution_providers_.GetIds());

  } catch (const std::exception& ex) {
    status = Status(common::ONNXRUNTIME, common::FAIL, "Exception during loading: " + std::string(ex.what()));
  } catch (...) {
    LOGS(*session_logger_, ERROR) << "Unknown exception in Load()";
    status = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Load()");
  }

  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, event_name, tp);
  }

  return status;
}

template <typename T>
common::Status InferenceSession::Load(const std::basic_string<T>& model_uri) {
  model_location_ = ToWideString(model_uri);
  auto loader = [this](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_location_, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      AddCustomOpDomains({domain.get()});
    }
#endif
    return onnxruntime::Model::Load(model_location_, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr);
  };

  common::Status st = Load(loader, "model_loading_uri");
  if (!st.IsOK()) {
    std::ostringstream oss;
    oss << "Load model from " << ToMBString(model_uri) << " failed:" << st.ErrorMessage();
    return common::Status(st.Category(), st.Code(), oss.str());
  }
  return Status::OK();
}

common::Status InferenceSession::Load(const std::string& model_uri) { return Load<char>(model_uri); }

#ifdef _WIN32
common::Status InferenceSession::Load(const std::wstring& model_uri) { return Load<PATH_CHAR_TYPE>(model_uri); }
#endif

common::Status InferenceSession::Load(const ModelProto& model_proto) {
  auto loader = [this, &model_proto](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      AddCustomOpDomains({domain.get()});
    }
#endif
    return onnxruntime::Model::Load(model_proto, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr);
  };

  return Load(loader, "model_loading_proto");
}

common::Status InferenceSession::Load(std::unique_ptr<ModelProto> p_model_proto) {
  auto loader = [this, &p_model_proto](std::shared_ptr<onnxruntime::Model>& model) {
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(*p_model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      AddCustomOpDomains({domain.get()});
    }
#endif
    return onnxruntime::Model::Load(std::move(p_model_proto), model,
                                    HasLocalSchema() ? &custom_schema_registries_ : nullptr);
  };

  return Load(loader, "model_loading_proto");
}

common::Status InferenceSession::Load(std::istream& model_istream) {
  auto loader = [this, &model_istream](std::shared_ptr<onnxruntime::Model>& model) {
    ModelProto model_proto;

    google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
    const bool result = model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
    if (!result) {
      return Status(common::ONNXRUNTIME, common::INVALID_PROTOBUF,
                    "Failed to load model because protobuf parsing failed.");
    }
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      AddCustomOpDomains({domain.get()});
    }
#endif
    return onnxruntime::Model::Load(model_proto, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr);
  };

  return Load(loader, "model_loading_istream");
}

common::Status InferenceSession::Load(const void* model_data, int model_data_len) {
  auto loader = [this, model_data, model_data_len](std::shared_ptr<onnxruntime::Model>& model) {
    ModelProto model_proto;

    const bool result = model_proto.ParseFromArray(model_data, model_data_len);
    if (!result) {
      return Status(common::ONNXRUNTIME, common::INVALID_PROTOBUF,
                    "Failed to load model because protobuf parsing failed.");
    }
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
    LoadInterOp(model_proto, interop_domains_, [&](const char* msg) { LOGS(*session_logger_, WARNING) << msg; });
    for (const auto& domain : interop_domains_) {
      AddCustomOpDomains({domain.get()});
    }
#endif

    return onnxruntime::Model::Load(model_proto, model, HasLocalSchema() ? &custom_schema_registries_ : nullptr);
  };

  return Load(loader, "model_loading_array");
}

common::Status InferenceSession::TransformGraph(onnxruntime::Graph& graph,
                                                const onnxruntime::GraphTransformerManager& graph_transformer_mgr,
                                                const ExecutionProviders& providers,
                                                KernelRegistryManager& kernel_registry_manager,
                                                const InsertCastTransformer& insert_cast_transformer,
                                                SessionState& session_state) {
  // The transformer order:
  // 1. built-in graph rewriter
  // 2. each execution provider's transformer
  // 3. do node placement according to kernel definition
  // 4. insert copy nodes
  // 5. insert cast nodes.

  // first apply global(execution provider independent),  level 1(default/system/basic) graph to graph optimizations
  ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr.ApplyTransformers(graph, TransformerLevel::Level1));

#ifdef USE_DML
  // TODO: this is a temporary workaround to apply the DML EP's custom graph transformer prior to partitioning. This
  // transformer applies DML-specific fusions that go beyond what ORT offers by default. Ideally the DML EP should
  // apply these transforms during partitioning, but the full mutable Graph object isn't exposed to
  // IExecutionProvider::GetCapability, which is necessary for the DML EP's transforms.
  //
  // To prevent this from interfering with other EPs, we only apply this transform if the DML EP is the only one that's
  // registered (aside from the CPU EP, which is always registered by default.)
  if (execution_providers_.Get(kDmlExecutionProvider) && execution_providers_.NumProviders() <= 2) {
    auto dml_registry = execution_providers_.Get(kDmlExecutionProvider)->GetKernelRegistry();
    Dml::GraphTransformer dml_transformer(onnxruntime::kDmlExecutionProvider, std::move(dml_registry));

    bool modified = false;
    dml_transformer.Apply(graph, modified);
  }
#endif

  // Do partitioning based on execution providers' capability.
  GraphPartitioner partitioner(kernel_registry_manager, providers);
  ORT_RETURN_IF_ERROR_SESSIONID_(partitioner.Partition(graph, session_state.ExportDll(), session_state.GetMutableFuncMgr()));

  // apply transformers except default transformers
  // Default transformers are required for correctness and they are owned and run by inference session
  for (int i = static_cast<int>(TransformerLevel::Level1); i < static_cast<int>(TransformerLevel::MaxTransformerLevel); i++) {
    ORT_RETURN_IF_ERROR_SESSIONID_(graph_transformer_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i)));
  }

  bool modified = false;
  // Insert cast node/s.
  ORT_RETURN_IF_ERROR_SESSIONID_(insert_cast_transformer.Apply(graph, modified));

  // Now every node should be already assigned to an execution provider
  std::unordered_map<std::string, std::vector<std::string>> node_placements;
  bool is_verbose_mode = session_logger_->GetSeverity() == logging::Severity::kVERBOSE;
  for (auto& node : graph.Nodes()) {
    const auto& node_provider = node.GetExecutionProviderType();
    if (node_provider.empty()) {
      std::ostringstream oss;
      oss << "Could not find an implementation for the node ";
      if (!node.Name().empty()) oss << node.Name() << ":";
      oss << node.OpType();
      if (node.Op()) {
        oss << "(" << node.Op()->since_version() << ")";
      }
      return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, oss.str());
    } else {
      if (is_verbose_mode) {  // TODO: should we disable this if the number of nodes are above a certain threshold?
        std::string node_str = node.OpType();
        node_str += " (";
        node_str += node.Name();
        node_str += ")";
        node_placements[node_provider].push_back(node_str);
      }
    }
  }

  // print placement info
  if (is_verbose_mode) {
    LOGS(*session_logger_, VERBOSE) << "Node placements";
    if (node_placements.size() == 1) {
      LOGS(*session_logger_, VERBOSE) << "All nodes have been placed on [" << node_placements.begin()->first << "].";
    } else {
      for (const auto& pr : node_placements) {
        std::ostringstream all_nodes_str;
        std::copy(pr.second.begin(), pr.second.end(), std::ostream_iterator<std::string>(all_nodes_str, ", "));
        LOGS(*session_logger_, VERBOSE) << " Provider: [" << pr.first << "]"
                                        << ": [" << all_nodes_str.str() << "]";
      }
    }
  }

  std::vector<std::string> provider_types;
  for (auto& provider_ptr : providers) {
    provider_types.push_back(provider_ptr->Type());
  }

  // Insert copy node/s.
  MemcpyTransformer copy_transformer{provider_types, kernel_registry_manager};
  ORT_RETURN_IF_ERROR_SESSIONID_(copy_transformer.Apply(graph, modified));

  return common::Status::OK();
}

/// Create SessionState instance for each subgraph as we need that for the GraphPartitioner
/// This will be initialized by InitializeSubgraphSessions.
common::Status InferenceSession::CreateSubgraphSessionState(Graph& graph, SessionState& session_state) {
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& name = entry.first;
      Graph* subgraph = entry.second;
      ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

      auto subgraph_session_state = onnxruntime::make_unique<SessionState>(execution_providers_,
                                                                           session_state.GetEnableMemoryPattern(),
                                                                           session_state.GetThreadPool(),
                                                                           session_state.GetInterOpThreadPool());
      subgraph_session_state->SetProfiler(session_profiler_);
      subgraph_session_state->SetLogger(*session_logger_);
      // Pass data transfer manager to subgraph.
      subgraph_session_state->SetDataTransferMgr(&session_state.GetDataTransferMgr());
      // Pass fused function manager to subgraph
      subgraph_session_state->GetMutableFuncMgr().SetFusedFuncs(session_state.GetFuncMgr());

      // recurse
      ORT_RETURN_IF_ERROR_SESSIONID_(CreateSubgraphSessionState(*subgraph, *subgraph_session_state));

      // add the subgraph SessionState instance to the parent graph SessionState so it can be retrieved
      // by Compute() via OpKernelContextInternal.
      session_state.AddSubgraphSessionState(node.Index(), name, std::move(subgraph_session_state));
    }
  }

  return Status::OK();
}

/// iterate nodes in graph looking for ones with graph attribute/s
/// @param graph The graph to iterate
/// @param session_state The SessionState instance for 'graph'.
/// @remarks We pass in graph and session_state so we can handled nested subgraphs in the future
common::Status InferenceSession::InitializeSubgraphSessions(Graph& graph, SessionState& session_state) {
  for (auto& node : graph.Nodes()) {
    // We only need subgraph session state for control flow nodes being handled by the CPU execution provider.
    // Remove it if it's not needed.
    if (node.ContainsSubgraph() && node.GetExecutionProviderType() != kCpuExecutionProvider) {
      session_state.RemoveSubgraphSessionState(node.Index());
      continue;
    }

    for (const auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& name = entry.first;
      Graph& subgraph = *entry.second;

      SessionState* subgraph_session_state = session_state.GetMutableSubgraphSessionState(node.Index(), name);
      ORT_ENFORCE(subgraph_session_state, "CreateSubgraphSessionState should have created an entry earlier.");

      // setup everything required to execute the subgraph and save it in subgraph_session_state
      SessionStateInitializer initializer(session_options_.enable_mem_pattern, model_location_, subgraph,
                                          *subgraph_session_state, execution_providers_, kernel_registry_manager_);

      const auto implicit_inputs = node.ImplicitInputDefs();
      ORT_RETURN_IF_ERROR_SESSIONID_(initializer.CreatePlan(&node, &implicit_inputs,
                                                            session_options_.execution_mode));
      // LOGS(*session_logger_, VERBOSE) << std::make_pair(subgraph_info.session_state->GetExecutionPlan(),
      //                                                   &*subgraph_info.session_state);

      // setup all the info for handling the feeds and fetches used in subraph execution
      auto* p_op_kernel = session_state.GetMutableKernel(node.Index());
      ORT_ENFORCE(p_op_kernel);
      auto& control_flow_kernel = dynamic_cast<controlflow::IControlFlowKernel&>(*p_op_kernel);
      ORT_RETURN_IF_ERROR_SESSIONID_(control_flow_kernel.SetupSubgraphExecutionInfo(session_state, name, *subgraph_session_state));

      // recurse
      ORT_RETURN_IF_ERROR_SESSIONID_(InitializeSubgraphSessions(subgraph, *subgraph_session_state));
    }
  }

  return Status::OK();
}

common::Status InferenceSession::Initialize() {
  Status status = Status::OK();
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.StartTime();
  }

  try {
    LOGS(*session_logger_, INFO) << "Initializing session.";
    std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded.");
    }
    if (is_inited_) {  // already initialized
      LOGS(*session_logger_, INFO) << "Session has already been initialized.";
      return common::Status::OK();
    }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
    TraceLoggingWriteStart(session_activity, "OrtInferenceSessionActivity");
    session_activity_started_ = true;
#endif
    // Register default CPUExecutionProvider if user didn't provide it through the Register() calls
    if (!execution_providers_.Get(onnxruntime::kCpuExecutionProvider)) {
      LOGS(*session_logger_, INFO) << "Adding default CPU execution provider.";
      CPUExecutionProviderInfo epi{session_options_.enable_cpu_mem_arena};
      auto p_cpu_exec_provider = onnxruntime::make_unique<CPUExecutionProvider>(epi);
      ORT_RETURN_IF_ERROR_SESSIONID_(RegisterExecutionProvider(std::move(p_cpu_exec_provider)));
    }

    if (session_options_.execution_mode == ExecutionMode::ORT_PARALLEL &&
        execution_providers_.Get(onnxruntime::kCudaExecutionProvider)) {
      LOGS(*session_logger_, ERROR) << "Parallel execution is currently not supported "
                                       "for the registered CUDA Execution Provider.";
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "Parallel execution is currently not supported "
                            "for the registered CUDA Execution Provider.");
    }

    // add predefined transformers
    AddPredefinedTransformers(graph_transformation_mgr_, session_options_.graph_optimization_level,
                              transformers_to_enable_);

    onnxruntime::Graph& graph = model_->MainGraph();

    // Collect the kernel registries from execution provider instances;
    // There are 2 kinds of kernel registries with priority from high to low as below,
    // 1. Custom execution provider type specific kernel registries.
    // 2. common execution provider type specific kernel registries.
    // The 1st and 2nd ones are shared across sessions.
    // The 1st ones should have already been registered via session-level API into KernelRegistryManager.
    //
    // Register 2nd registries into KernelRegistryManager.
    ORT_RETURN_IF_ERROR_SESSIONID_(kernel_registry_manager_.RegisterKernels(execution_providers_));

    SessionStateInitializer session_initializer(session_options_.enable_mem_pattern, model_location_, graph,
                                                session_state_, execution_providers_, kernel_registry_manager_);

    // create SessionState for subgraphs as it's needed by the transformers
    ORT_RETURN_IF_ERROR_SESSIONID_(CreateSubgraphSessionState(graph, session_state_));

    // apply any transformations to the main graph and any subgraphs
    ORT_RETURN_IF_ERROR_SESSIONID_(TransformGraph(graph, graph_transformation_mgr_,
                                                  execution_providers_, kernel_registry_manager_,
                                                  insert_cast_transformer_,
                                                  session_state_));

    // now that all the transforms are done, call Resolve on the main graph. this will recurse into the subgraphs.
    ORT_RETURN_IF_ERROR_SESSIONID_(graph.Resolve());

    if (!session_options_.optimized_model_filepath.empty()) {
      // Serialize optimized ONNX model.
      ORT_RETURN_IF_ERROR_SESSIONID_(Model::Save(*model_, session_options_.optimized_model_filepath));
      if (session_options_.graph_optimization_level >= TransformerLevel::Level3) {
        LOGS(*session_logger_, WARNING) << "Serializing Optimized ONNX model with Graph Optimization"
                                           " level greater than ORT_ENABLE_EXTENDED. The generated"
                                           " model may contain hardware and execution provider specific"
                                           " optimizations, and should only be used in the same environment"
                                           " the model was optimized for.";
      }
    }

    ORT_RETURN_IF_ERROR_SESSIONID_(session_initializer.CreatePlan(nullptr, nullptr, session_options_.execution_mode));

    // handle any subgraphs
    ORT_RETURN_IF_ERROR_SESSIONID_(InitializeSubgraphSessions(graph, session_state_));
    is_inited_ = true;
    LOGS(*session_logger_, INFO) << "Session successfully initialized.";
  } catch (const NotImplementedException& ex) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Exception during initialization: ", ex.what());
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
  } catch (const std::exception& ex) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Exception during initialization: ", ex.what());
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
  } catch (...) {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Encountered unknown exception in Initialize()");
    LOGS(*session_logger_, ERROR) << status.ErrorMessage();
  }

  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "session_initialization", tp);
  }
  return status;
}

int InferenceSession::GetCurrentNumRuns() const { return current_num_runs_.load(); }

const std::vector<std::string>& InferenceSession::GetRegisteredProviderTypes() const {
  return execution_providers_.GetIds();
}

const SessionOptions& InferenceSession::GetSessionOptions() const {
  return session_options_;
}

common::Status InferenceSession::CheckShapes(const std::string& input_name,
                                             const TensorShape& input_shape,
                                             const TensorShape& expected_shape) const {
  auto input_shape_sz = input_shape.NumDimensions();
  auto expected_shape_sz = expected_shape.NumDimensions();
  if (input_shape_sz != expected_shape_sz) {
    std::ostringstream ostr;
    ostr << "Invalid rank for input: " << input_name << " Got: " << input_shape_sz << " Expected: " << expected_shape_sz
         << " Please fix either the inputs or the model.";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, ostr.str());
  }

  std::vector<int> invalid_dim_indices;
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
      int idx = invalid_dim_indices[i];
      ostr << " index: " << idx << " Got: " << input_shape[idx] << " Expected: " << expected_shape[idx] << "\n";
    }
    ostr << " Please fix either the inputs or the model.";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, ostr.str());
  }

  return Status::OK();
}

static common::Status CheckTypes(MLDataType actual, MLDataType expected) {
  if (actual == expected) {
    return Status::OK();
  }
  auto actual_name = std::string(typeid(*actual).name());
  auto expected_name = std::string(typeid(*expected).name());
  return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                "Unexpected input data type. Actual: (" + actual_name + ") , expected: (" + expected_name + ")");
}

common::Status InferenceSession::ValidateInputs(const std::vector<std::string>& feed_names,
                                                const std::vector<OrtValue>& feeds) const {
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
    auto& input_ml_value = feeds.at(i);
    if (input_ml_value.IsTensor()) {
      // check for type
      if (!expected_type->IsTensorType()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input with name: ", feed_name,
                               " is not expected to be of type tensor.");
      }

      auto expected_element_type = expected_type->AsTensorType()->GetElementType();
      auto input_element_type = input_ml_value.Get<Tensor>().DataType();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_element_type, expected_element_type));

      // check for shape
      const auto& expected_shape = iter->second.tensor_shape;
      if (expected_shape.NumDimensions() > 0) {
        const auto& input_shape = input_ml_value.Get<Tensor>().Shape();
        ORT_RETURN_IF_ERROR_SESSIONID_(CheckShapes(feed_name, input_shape, expected_shape));
      }
    } else {
      auto input_type = input_ml_value.Type();
      ORT_RETURN_IF_ERROR_SESSIONID_(CheckTypes(input_type, expected_type));
    }
  }

  return Status::OK();
}

common::Status InferenceSession::ValidateOutputs(const std::vector<std::string>& output_names,
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

Status InferenceSession::Run(const RunOptions& run_options, const std::vector<std::string>& feed_names,
                             const std::vector<OrtValue>& feeds, const std::vector<std::string>& output_names,
                             std::vector<OrtValue>* p_fetches) {
  TimePoint tp;
  if (session_profiler_.IsEnabled()) {
    tp = session_profiler_.StartTime();
  }

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingActivity<telemetry_provider_handle> ortrun_activity;
  ortrun_activity.SetRelatedActivity(session_activity);
  TraceLoggingWriteStart(ortrun_activity, "OrtRun");
#endif
  Status retval = Status::OK();

  try {
    if (!is_inited_) {
      LOGS(*session_logger_, ERROR) << "Session was not initialized";
      return Status(common::ONNXRUNTIME, common::FAIL, "Session not initialized.");
    }

    ORT_RETURN_IF_ERROR_SESSIONID_(ValidateInputs(feed_names, feeds));
    ORT_RETURN_IF_ERROR_SESSIONID_(ValidateOutputs(output_names, p_fetches));

    FeedsFetchesInfo info(feed_names, output_names, session_state_.GetOrtValueNameIdxMap());
    FeedsFetchesManager feeds_fetches_manager{std::move(info)};

    if (!run_options.run_tag.empty()) {
      LOGS(*session_logger_, INFO) << "Running with tag: " << run_options.run_tag;
    }

    ++current_num_runs_;

    // TODO should we add this exec to the list of executors? i guess its not needed now?

    // scope of owned_run_logger is just the call to Execute.
    // If Execute ever becomes async we need a different approach
    std::unique_ptr<logging::Logger> owned_run_logger;
    auto run_logger = CreateLoggerForRun(run_options, owned_run_logger);

    // info all execution providers InferenceSession:Run started
    // TODO: only call OnRunStart for all providers in-use
    for (auto& xp : execution_providers_) {
      ORT_CHECK_AND_SET_RETVAL(xp->OnRunStart());
    }

    // execute the graph
    ORT_CHECK_AND_SET_RETVAL(
        utils::ExecuteGraph(session_state_, feeds_fetches_manager, feeds, *p_fetches,
                            session_options_.execution_mode,
                            run_options.terminate, run_logger));

  } catch (const std::exception& e) {
    retval = Status(common::ONNXRUNTIME, common::FAIL, e.what());
  } catch (...) {
    retval = Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
  }

  // info all execution providers InferenceSession:Run ended
  for (auto& xp : execution_providers_) {
    ORT_CHECK_AND_SET_RETVAL(xp->OnRunEnd());
  }

  --current_num_runs_;

  // keep track of telemetry
  ++total_runs_since_last_;
  total_run_duration_since_last_ += TimeDiffMicroSeconds(tp);

  // time to send telemetry?
  if (TimeDiffMicroSeconds(time_sent_last_) > kDurationBetweenSending) {
    // send the telemetry
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogRuntimePerf(session_id_, total_runs_since_last_, total_run_duration_since_last_);
    // reset counters
    time_sent_last_ = std::chrono::high_resolution_clock::now();
    total_runs_since_last_ = 0;
    total_run_duration_since_last_ = 0;
  }

  // send out profiling events (optional)
  if (session_profiler_.IsEnabled()) {
    session_profiler_.EndTimeAndRecordEvent(profiling::SESSION_EVENT, "model_run", tp);
  }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  TraceLoggingWriteStop(ortrun_activity, "OrtRun");
#endif
  return retval;
}

common::Status InferenceSession::Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names,
                                     std::vector<OrtValue>* p_fetches) {
  return Run(RunOptions(), feeds, output_names, p_fetches);
}

common::Status InferenceSession::Run(const RunOptions& run_options, const NameMLValMap& feeds_map,
                                     const std::vector<std::string>& output_names, std::vector<OrtValue>* p_fetches) {
  std::vector<std::string> feed_names;
  std::vector<OrtValue> feeds;

  auto num_feeds = feeds_map.size();
  feed_names.reserve(num_feeds);
  feeds.reserve(num_feeds);

  for (auto& pair : feeds_map) {
    feed_names.push_back(pair.first);
    feeds.push_back(pair.second);
  }

  return Run(run_options, feed_names, feeds, output_names, p_fetches);
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
      return std::make_pair(common::Status(common::ONNXRUNTIME, common::FAIL, "Model was not loaded."),
                            nullptr);
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

  // private constructor, can't use make_unique
  *io_binding = std::unique_ptr<IOBinding>(new IOBinding(session_state_));
  return Status::OK();
}

common::Status InferenceSession::Run(const RunOptions& run_options, IOBinding& io_binding) {
  // TODO should Run() call io_binding.SynchronizeInputs() or should it let the callers do it?
  // io_binding.SynchronizeInputs();
  return Run(run_options, io_binding.GetInputNames(), io_binding.GetInputs(),
             io_binding.GetOutputNames(), &io_binding.GetOutputs());
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

void InferenceSession::StartProfiling(const std::string& file_prefix) { StartProfiling<char>(file_prefix); }

#ifdef _WIN32
void InferenceSession::StartProfiling(const std::wstring& file_prefix) { StartProfiling<PATH_CHAR_TYPE>(file_prefix); }
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

// assumes model has already been loaded before
common::Status InferenceSession::DoPostLoadProcessing(onnxruntime::Model& model) {
  // TODO add other post load processing here
  common::Status status = SaveModelMetadata(model);
  return status;
}

common::Status InferenceSession::SaveModelMetadata(const onnxruntime::Model& model) {
  VLOGS(*session_logger_, 1) << "Saving model metadata";
  const onnxruntime::Graph& graph = model.MainGraph();

  // save model metadata
  model_metadata_.producer_name = model.ProducerName();
  model_metadata_.description = model.DocString();
  model_metadata_.domain = model.Domain();
  model_metadata_.version = model.ModelVersion();
  model_metadata_.custom_metadata_map = model.MetaData();
  model_metadata_.graph_name = graph.Name();

  for (auto input : graph.GetInputs()) {
    required_inputs_.insert(input->Name());
  }

  auto add_inputs = [this](const InputDefList& inputs) {
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

  session_state_.SetLogger(*session_logger_);
}

// Registers all the predefined transformers with transformer manager
void InferenceSession::AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                                 TransformerLevel graph_optimization_level,
                                                 const std::vector<std::string>& custom_list) {
  auto add_transformers = [&](TransformerLevel level) {
    // Generate and register transformers for level
    auto transformers_to_register = optimizer_utils::GenerateTransformers(level, session_options_.free_dimension_overrides, custom_list);
    for (auto& entry : transformers_to_register) {
      transformer_manager.Register(std::move(entry), level);
    }
  };

  ORT_ENFORCE(graph_optimization_level < TransformerLevel::MaxTransformerLevel,
              "Allowed values are 1 and 2. Current level is set to " +
                  std::to_string(static_cast<uint32_t>(graph_optimization_level)));

  if ((graph_optimization_level >= TransformerLevel::Level1) || !custom_list.empty()) {
    add_transformers(TransformerLevel::Level1);
  }

  if ((graph_optimization_level >= TransformerLevel::Level2) || !custom_list.empty()) {
    add_transformers(TransformerLevel::Level2);
  }

  if ((graph_optimization_level >= TransformerLevel::Level3) || !custom_list.empty()) {
    add_transformers(TransformerLevel::Level3);
  }
}

common::Status InferenceSession::WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms) {
  if (timeout_in_ms > 0) {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, "timeout_in_ms >0 is not supported");  // TODO
  }
  p_executor_done->WaitForNotification();

  return Status::OK();
}

}  // namespace onnxruntime
