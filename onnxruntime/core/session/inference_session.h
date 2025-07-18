// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <filesystem>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/path_string.h"
#include "core/common/profiler.h"
#include "core/common/status.h"
#include "core/framework/execution_providers.h"
#include "core/framework/framework_common.h"
#include "core/framework/iexecutor.h"
#include "core/framework/external_data_loader_manager.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/resource_accountant.h"
#include "core/framework/session_state.h"
#include "core/framework/tuning_results.h"
#include "core/framework/framework_provider_common.h"
#include "core/framework/session_options.h"
#include "core/graph/basic_types.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/insert_cast_transformer.h"
#include <mutex>
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
#include "core/language_interop_ops/language_interop_ops.h"
#endif
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include "core/platform/tracing.h"
#include <TraceLoggingActivity.h>
#endif
#ifdef _WIN32
#include "core/platform/windows/logging/etw_sink.h"
#include "core/platform/windows/telemetry.h"
#endif

namespace ONNX_NAMESPACE {
class ModelProto;
}  // namespace ONNX_NAMESPACE

// OrtModelEditorApi Model. Used to dynamically construct a model via C API at runtime.
struct OrtModel;

namespace onnxruntime {  // forward declarations
class CustomRegistry;
class Environment;
class GraphTransformer;
class IExecutionProvider;
class IOBinding;
struct Notification;

void reset_saturation_count();

#ifdef ENABLE_TRAINING
struct PartialGraphExecutionState;
using OrtValueCache = InlinedHashMap<std::string, OrtValue>;
using OrtValueCachePtr = std::shared_ptr<OrtValueCache>;
#endif

namespace logging {
class LoggingManager;
}

/**
 * Pre-defined and custom metadata about the model.
 */
struct ModelMetadata {
  ModelMetadata() = default;
  ModelMetadata(const ModelMetadata&) = default;
  ~ModelMetadata() = default;
  ModelMetadata& operator=(const ModelMetadata&) = delete;

  std::string producer_name;
  std::string producer_version;
  std::string graph_name;
  std::string domain;
  std::string description;
  std::string graph_description;
  int64_t version = 0;
  std::unordered_map<std::string, std::string> custom_metadata_map;
};

/**
 * @brief This is the main class used to Run a model.
 * Sample simple usage:
 *  CPUExecutionProviderInfo epi;
 *  ProviderOption po{"CPUExecutionProvider", epi};
 *  SessionOptions so(vector<ProviderOption>{po});
 *  string log_id = "Foo";
 *  auto logging_manager = std::make_unique<LoggingManager>
                (std::unique_ptr<ISink>{new CLogSink{}},
                                  static_cast<Severity>(lm_info.default_warning_level),
                                  false,
                                  LoggingManager::InstanceType::Default,
                                  &log_id)
 *  Environment::Create(std::move(logging_manager), env)
 *  InferenceSession session_object{so,env};
 *  common::Status status = session_object.Load(MODEL_URI);
 *  common::Status status = session_object.Initialize();
 *
 *  NameMLValMap feeds;
 *  feeds.insert({});
 *  ...
 *  std::vector<std::string> output_names;
 *  output_names.insert(...);
 *  ...
 *  std::vector<OrtValue> fetches;
 *  common::Status status = session_object.Run(run_options, feeds, output_names, &fetches);
 *  process the output here...
 */

class InferenceSession {
  struct InputOutputDefMetaData {
    InputOutputDefMetaData(const NodeArg* node_arg0, MLDataType ml_data_type0, TensorShape&& tensor_shape0)
        : node_arg(node_arg0), ml_data_type(ml_data_type0), tensor_shape(std::move(tensor_shape0)) {
    }

    InputOutputDefMetaData(const NodeArg* node_arg0, MLDataType ml_data_type0)
        : node_arg(node_arg0), ml_data_type(ml_data_type0) {
    }

    gsl::not_null<const NodeArg*> node_arg;
    MLDataType ml_data_type;
    std::optional<TensorShape> tensor_shape;  // not applicable if the input is non-tensor type
  };

  using InputOutputDefMetaMap = InlinedHashMap<std::string_view, InputOutputDefMetaData>;
  static std::map<uint32_t, InferenceSession*> active_sessions_;
#ifdef _WIN32
  static std::mutex active_sessions_mutex_;  // Protects access to active_sessions_
  static onnxruntime::WindowsTelemetry::EtwInternalCallback callback_ML_ORT_provider_;
  onnxruntime::logging::EtwRegistrationManager::EtwInternalCallback callback_ETWSink_provider_;
#endif

 public:
#if !defined(ORT_MINIMAL_BUILD)

  /**
   * How minimal build graph optimizations should be handled in a full build.
   * Note: These only apply to optimizations at the extended level or higher.
   */
  enum class MinimalBuildOptimizationHandling {
    /** Run full build optimizations. The default behavior. */
    ApplyFullBuildOptimizations,
    /** Save minimal build optimizations as runtime optimizations in an ORT format model. */
    SaveMinimalBuildRuntimeOptimizations,
    /** Only run minimal build optimizations. */
    OnlyApplyMinimalBuildOptimizations,
  };

  using RecordRuntimeOptimizationProducedNodeOpSchemaFn = std::function<Status(const ONNX_NAMESPACE::OpSchema&)>;

#endif

  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    */
  explicit InferenceSession(const SessionOptions& session_options,
                            const Environment& session_env);

  /**
    Create a new InferenceSession that accepts thread pools for intra and inter op thread execution.
    Used by WinML only!
    @param session_options Session options.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    @param external_intra_op_thread_pool This represents the intra op threadpool.
    @param external_inter_op_thread_pool This represents the inter op threadpool.
    */
  explicit InferenceSession(const SessionOptions& session_options,
                            const Environment& session_env,
                            onnxruntime::concurrency::ThreadPool* external_intra_op_thread_pool,
                            onnxruntime::concurrency::ThreadPool* external_inter_op_thread_pool);

#if !defined(ORT_MINIMAL_BUILD)
  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param model_uri absolute path of the model file.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    This ctor will throw on encountering model parsing issues.
    */
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   const PathString& model_uri);
#ifdef _WIN32
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   const std::string& model_uri);
#endif

  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param istream object of the model.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    This ctor will throw on encountering model parsing issues.
    */
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   std::istream& model_istream);

  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param model_data Model data buffer.
    @param model_data_len Model data buffer size.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    This ctor will throw on encountering model parsing issues.
    */
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   const void* model_data,
                   int model_data_len);

#endif  // !defined(ORT_MINIMAL_BUILD)

  virtual ~InferenceSession();

  /**
   * Register an execution provider. If you've one to register, call this before invoking Initialize().
   * The order of invocation indicates the preference order as well. In other words call this method
   * on your most preferred execution provider first followed by the less preferred ones.
   * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
   * @return OK if success.
   */
  [[nodiscard]] common::Status RegisterExecutionProvider(const std::shared_ptr<IExecutionProvider>& p_exec_provider);

#if !defined(ORT_MINIMAL_BUILD)
  /**
   * Register a graph transformer. If you've one to register, call this before invoking Initialize().
   * Calling this API is optional.
   * @param p_graph_transformer The graph transformer to register.
   * @param level Optional. Level to which this transformer should be registered. Default is set to 2.
   * @return OK if success.
   */
  [[nodiscard]] common::Status RegisterGraphTransformer(std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer,
                                                        TransformerLevel level = TransformerLevel::Level2);

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  /**
   * Filter the enabled optimizers (either transformer or rewrite rule) using optimizers_to_disable.
   * For an optimizer to be enabled, it must be allowed at the current optimization level (as specified in
   * session options), and NOT in optimizers_to_disable.
   * This allows finer grained control of the enabled/disabled optimizations.
   * Must be called before Initialize() to take effect.
   *
   * Calling this API is optional.
   * @return OK if success.
   */
  common::Status FilterEnabledOptimizers(InlinedHashSet<std::string>&& optimizers_to_disable);
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  /**
   * Add custom ops. This API is not thread safe.
   */
  [[nodiscard]] common::Status AddCustomOpDomains(gsl::span<OrtCustomOpDomain* const> ops);

  /**
   * Register a custom registry for operator schema and kernels.  If you've one to register,
   * call this before invoking Initialize().
   * The order of invocation indicates the reversed preference order: Register your most
   * preferred registry at the end.
   * Calling this API is optional.
   * This API is not thread safe.
   * @return OK if success.
   */
  [[nodiscard]] common::Status RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry);
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

  /**
   * Load an ONNX or ORT format model.
   *
   * Set SessionOptions session config value ORT_SESSION_OPTIONS_CONFIG_LOAD_MODEL_FORMAT to 'ORT' or 'ONNX' to
   * explicitly choose model format.
   *
   * If format is not explicitly specified and filename ends in '.ort' it will be inferred to be an ORT format model.
   * All other files are assumed to be in ONNX format.
   *
   * @param model_uri absolute path of the model file.
   * @return OK if success.
   */
  [[nodiscard]] common::Status Load(const PathString& model_uri);
#ifdef _WIN32
  [[nodiscard]] common::Status Load(const std::string& model_uri);
#endif
  /**
   * Load an ONNX or ORT format model.
   *
   * Set SessionOptions session config value ORT_SESSION_OPTIONS_CONFIG_LOAD_MODEL_FORMAT to 'ORT' or 'ONNX' to
   * explicitly choose model format.
   *
   * If format is not explicitly specified the model format will be inferred from the bytes, defaulting to ONNX.
   *
   * @param model_data Model data buffer
   * @param model_data_len Model data buffer size
   * @return OK if success.
   */
  [[nodiscard]] common::Status Load(const void* model_data, int model_data_len);

#if !defined(ORT_MINIMAL_BUILD)
  /**
   * Load an ONNX model.
   * @param istream object of the model.
   * @allow_released_opsets_only Set true if you would like to only allow released ONNX opsets only, set false otherwise.
   * @return OK if success.
   */
  [[nodiscard]] common::Status Load(std::istream& model_istream, bool allow_released_opsets_only = true);

  /**
   * Load an ONNX model from the member model_proto_.
   * To be called only in conjunction with a ctor that takes in a model path/ model stream/ model array
   * @return OK if success.
   */
  [[nodiscard]] common::Status Load();

  /**
   * Load an OrtModel that was dynamically constructed via OrtModelEditorApi.
   *
   * @param graph_api_model OrtModel from OrtModelEditorApi
   * @return OK if success.
   */
  [[nodiscard]] common::Status Load(const OrtModel& graph_api_model);

  /**
   * Apply updates from an OrtModel that was created via OrtModelEditorApi.
   * This can:
   *   - add nodes at the start and end of the model
   *   - add initializers
   *   - update the graph inputs/outputs
   *
   * @param graph_api_model OrtModel from OrtModelEditorApi
   * @return OK if success.
   */
  [[nodiscard]] common::Status ApplyUpdates(const OrtModel& graph_api_model);

#endif  // !defined(ORT_MINIMAL_BUILD)

  /**
   * Initializes a previously loaded ONNX model. Initialization includes but is not
   * limited to graph transformations, construction of kernels, EP policy decisions, etc.
   * This method assumes that a model has been loaded previously.
   * This API is thread-safe.
   * @return OK if success
   */
  [[nodiscard]] common::Status Initialize();

  [[nodiscard]] common::Status SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                   gsl::span<const char* const> values);

  [[nodiscard]] common::Status Run(const RunOptions& run_options, gsl::span<const std::string> feed_names,
                                   gsl::span<const OrtValue> feeds, gsl::span<const std::string> output_names,
                                   std::vector<OrtValue>* p_fetches,
                                   const std::vector<OrtDevice>* p_fetches_device_info = nullptr);

  [[nodiscard]] common::Status Run(const RunOptions& run_options,
                                   gsl::span<const char* const> feed_names,
                                   gsl::span<const OrtValue* const> feeds,
                                   gsl::span<const char* const> fetch_names,
                                   gsl::span<OrtValue*> fetches);

  [[nodiscard]] common::Status RunAsync(const RunOptions* run_options,
                                        gsl::span<const char* const> feed_names,
                                        gsl::span<const OrtValue* const> feeds,
                                        gsl::span<const char* const> fetch_names,
                                        gsl::span<OrtValue*> fetches,
                                        RunAsyncCallbackFn callback,
                                        void* user_data = nullptr);

  /**
   * Run a pre-loaded and pre-intialized model.
   * Multiple threads are allowed to run this function; hence its thread-safe.
   * @param feeds named inputs owned by client code and should not be changed during
   *        execution of this function.
   * @param output_names output names
   * @param p_fetches output values in the order specified by output_names.
   *        This should not be changed during execution of this function.
   * @return OK if success.
   */
  [[nodiscard]] common::Status Run(const NameMLValMap& feeds, gsl::span<const std::string> output_names,
                                   std::vector<OrtValue>* p_fetches);

  /**
   * See Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names, std::vector<OrtValue>* p_fetches)
   * for details.
   * @param run_options use this to tune the Run call to your needs.
   */
  [[nodiscard]] common::Status Run(const RunOptions& run_options, const NameMLValMap& feeds,
                                   gsl::span<const std::string> output_names,
                                   std::vector<OrtValue>* p_fetches);

  /**
   * Creates a new binding object for binding inputs and outputs.
   * @param provider_type specifies the location where the inputs need to be potentially copied.
   * See IOBinding class for more info.
   */
  [[nodiscard]] common::Status NewIOBinding(std::unique_ptr<IOBinding>* io_binding);

  [[nodiscard]] virtual common::Status Run(const RunOptions& run_options, IOBinding& io_binding);
  [[nodiscard]] common::Status Run(IOBinding& io_binding);

#ifdef ENABLE_TRAINING
  /**
   * Partially run a pre-loaded and pre-intialized model.
   * @param run_options run options.
   * @param mutable_feeds inputs owned by client code and will be released as long as the feeds be set in session states.
   * Then the feeds will purely managed in the session states.
   * @param fetches outputs produced after the execution of this function.
   * @param state State of the graph needed to resume partial graph run.
   * @param feeds_fetches_manager Contains feed/fetches name to internal indices mapping and information for device
   *                              copy/checks.
   * @param cache Contains node arg name to OrtValue map stashed from previous run
   *              for frontier tensors
   * @param partial_graph_index Index of the partial graph to run.
   */
  common::Status PartialRun(onnxruntime::RunOptions& run_options,
                            std::vector<OrtValue>& mutable_feeds,
                            std::vector<OrtValue>& fetches,
                            PartialGraphExecutionState& state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            const OrtValueCachePtr& cache,
                            int32_t partial_graph_index);
#endif

  /**
   * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
   * @note lifetime of the returned pointer is valid as long as the Session object is live.
   */
  std::pair<common::Status, const ModelMetadata*> GetModelMetadata() const;

  /**
   * Get all input definitions of the model. This does not include weights. Use this
   * to get the name/type/shapes of the inputs.
   * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
   * @note lifetime of the returned pointer is valid as long as the Session object is live.
   */
  std::pair<common::Status, const InputDefList*> GetModelInputs() const;

  /**
   * Get all definitions of the model for overridable initializers.
   * This does not include weights. Use this to get the name/type/shapes of the overridable initializers.
   * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
   * @note lifetime of the returned pointer is valid as long as the Session object is live.
   * @note for IR < 4 returned list will always be empty.
   */
  std::pair<common::Status, const InputDefList*> GetOverridableInitializers() const;

  /**
   * Get all output definitions of the model. Use this to get the name/type/shapes of the outputs.
   * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
   * @note lifetime of the returned pointer is valid as long as the Session object is live.
   */
  std::pair<common::Status, const OutputDefList*> GetModelOutputs() const;

  enum class SessionInputOutputType : uint8_t {
    kInput = 0,
    kOutput = 1,
    kOverridableInitializer = 2
  };

  /**
   * Get the OrtMemoryInfo for the inputs or outputs of the model.
   *
   * This is required for a user to know the location of the input/output when autoep selection is enabled.
   */
  common::Status GetInputOutputMemoryInfo(SessionInputOutputType type,
                                          InlinedVector<const OrtMemoryInfo*>& memory_info) const;
  /**
   * Get the OrtEpDevice (if available) for the inputs of the model.
   *
   * This is required for a user to know the location of the input/output when autoep selection is enabled.
   */
  common::Status GetEpDeviceForInputs(InlinedVector<const OrtEpDevice*>& memory_info) const;
  /**
   * Get the current number of in-progress concurrent Run calls.
   */
  int GetCurrentNumRuns() const;

  /**
   * Get the names of registered Execution Providers. The returned vector is ordered by Execution Provider
   * priority. The first provider in the vector has the highest priority.
   */
  const std::vector<std::string>& GetRegisteredProviderTypes() const;

  /*
   * Get the options this session was initialized with.
   */
  const SessionOptions& GetSessionOptions() const;

  /*
   * Get the options so auto-selected EPs can augment as they are added post-session creation.
   */
  SessionOptions& GetMutableSessionOptions();

  /*
   * Get the DataTransferManager associated with this session
   */
  const DataTransferManager& GetDataTransferManager() const;

  /*
   * Get the GetExternalDataLoaderManager associated with this session
   */
  const ExternalDataLoaderManager& GetExternalDataLoaderManager() const;

  /*
   * Get all the providers' options this session was initialized with.
   */
  const ProviderOptionsMap& GetAllProviderOptions() const;

  /**
   * Start profiling on this inference session. This simply turns on profiling events to be
   * recorded. A corresponding EndProfiling has to follow to write profiling data to a file.
   *@param file_prefix is the prefix of the profile file. It can include a directory path.
   */
  void StartProfiling(const std::string& file_prefix);
#ifdef _WIN32
  void StartProfiling(const std::wstring& file_prefix);
#endif
  /**
   * Start profiling on this inference session. This simply turns on profiling events to be
   * recorded. A corresponding EndProfiling has to follow to send profiling events through the logger's ISink.
   *@param logger_ptr is pointer to the logger where profiling events will be sent to.
   */
  void StartProfiling(const logging::Logger* logger_ptr);

  /**
    * Write captured profile events in chromium format.
    @return the name of the profile file.
    */
  std::string EndProfiling();
  /**
    * Return the profiler to access its attributes
    @return the profiler object
    */
  const profiling::Profiler& GetProfiling() const;

#if !defined(ORT_MINIMAL_BUILD)
  /**
   * Get the TuningResults of TunableOp for every execution providers.
   * @return The TuningResults of each execution provider.
   */
  std::vector<TuningResults> GetTuningResults() const;

  /**
   * Set the TuningResults back to each execution provider. Mainly for offline tuning.
   * @param trs is the list of TuningResults to be loaded.
   * @param error_on_invalid otherwise, validation faliure is not an error, only a warning log will be produced.
   * @param auto_enable if true, automatically enable tunable op usage (but not tuning) if the TuningResults is
                        correctly loaded
   * @return OK if success.
   */
  Status SetTuningResults(const std::vector<TuningResults>& trs, bool error_on_invalid = false,
                          bool auto_enable = false);
#endif

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  MemoryProfiler& GetMemoryProfiler() {
    return memory_profiler_;
  }
#endif

  /**
   * Search registered execution providers for an allocator that has characteristics
   * specified within mem_info
   * @param mem_info is a reference to OrtMemoryInfo that contains required specs
   * @return a ptr to the allocator or nullptr if not available
   */
  AllocatorPtr GetAllocator(const OrtMemoryInfo& mem_info) const;

  /**
   *Get InferenceSession logger.
   */
  const logging::Logger* GetLogger() const { return session_logger_; };

  const SessionState& GetSessionState() const {
    ORT_ENFORCE(session_state_ != nullptr, "Session must be initialized to create session state.");
    return *session_state_;
  }

  /**
   * Add a PrepackedWeightsContainer instance to the session so as to store the pre-packed weights
   *  of shared initializers to be shared across sessions.
   * @param prepacked_weights_container PrepackedWeightsContainer instance
   */
  Status AddPrePackedWeightsContainer(PrepackedWeightsContainer* prepacked_weights_container);

#if !defined(ORT_MINIMAL_BUILD)
  /**
   * CreateNodeStats recorder and enable collection of node statistics that is useful
   * for resource constrained partitioning and otherwise.
   *
   * @param node_stats_file - this file will be created at the same folder where the model file is present.
   */
  Status CreateNodeStatsRecorder(const std::filesystem::path& node_stats_file);

  /**
   * Returns true if collection is enabled
   */
  bool IsNodeStatsCollectionEnabled() const noexcept {
    return node_stats_recorder_.has_value();
  }

  /**
   * NodeStatsRecorder pointer. If not present, returns nullptr
   */
  NodeStatsRecorder* GetNodeStatsRecorder() noexcept {
    return node_stats_recorder_.has_value() ? &*node_stats_recorder_ : nullptr;
  }

#endif

  const Model& GetModel() const;
  const Environment& GetEnvironment() const;

  void SetWeightDataType(const std::string& type) {
    weight_data_type_ = type;
  }

  const std::string& GetWeightDataType() const {
    return weight_data_type_;
  }

  void SetGraphHash(const std::string& hash) {
    graph_hash_ = hash;
  }

  const std::string& GetGraphHash() const {
    return graph_hash_;
  }

  void SetWeightHash(const std::string& hash) {
    weight_hash_ = hash;
  }

  const std::string& GetWeightHash() const {
    return weight_hash_;
  }

  uint32_t GetCurrentSessionId() const {
    return session_id_;
  }

 protected:
#if !defined(ORT_MINIMAL_BUILD)

  /**
   * Load an ONNX model.
   * @param protobuf object corresponding to the model file. model_proto will be copied by the API.
   * @return OK if success.
   */
  [[nodiscard]] common::Status LoadOnnxModel(ONNX_NAMESPACE::ModelProto model_proto);

  /**
   * Load an ONNX model.
   * @param protobuf object corresponding to the model file. This is primarily to support large models.
   * @return OK if success.
   */
  [[nodiscard]] common::Status LoadOnnxModel(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto);

  [[nodiscard]] common::Status LoadWithLoader(std::function<common::Status(std::shared_ptr<Model>&)> loader,
                                              const std::string& event_name);

  [[nodiscard]] common::Status DoPostLoadProcessing(onnxruntime::Model& model);

#endif  // !defined(ORT_MINIMAL_BUILD)

  bool IsInitialized() const;

  // Use these 2 threadpool methods to get access to the threadpools since they rely on
  // specific flags in session options
  // These methods assume that session options have been finalized before the call.
  onnxruntime::concurrency::ThreadPool* GetIntraOpThreadPoolToUse() const {
    if (session_options_.use_per_session_threads) {
      if (external_intra_op_thread_pool_) {
        return external_intra_op_thread_pool_;
      } else {
        return thread_pool_.get();
      }
    } else {
      return intra_op_thread_pool_from_env_;
    }
  }

  onnxruntime::concurrency::ThreadPool* GetInterOpThreadPoolToUse() const {
    if (session_options_.use_per_session_threads) {
      if (external_inter_op_thread_pool_) {
        return external_inter_op_thread_pool_;
      } else {
        return inter_op_thread_pool_.get();
      }
    } else {
      return inter_op_thread_pool_from_env_;
    }
  }

  /// convenience pointer to logger. should always be the same as session_state_.Logger();
  const logging::Logger* session_logger_;

  // The list of execution providers.
  // This MUST be prior to model_ in case there are values in the model that were allocated using an allocator
  // provided by the EP. If that is the case the allocator's `free` implementation may depend on other parts of the
  // EP instance.
  ExecutionProviders execution_providers_;

  // The model served by this inference session instance.
  // Currently this has to be a shared ptr because the Model::Load method
  // returns a shared_ptr only. Ideally factory functions should always return
  // unique_ptr for maximum flexibility. Client can always upgrade it to shared_ptr
  // if they need.
  std::shared_ptr<onnxruntime::Model> model_;

  // The file path of where the model was loaded. e.g. /tmp/test_squeezenet/model.onnx
  PathString model_location_;

  // Input, Output and Weight tensor data types
  std::string input_data_type_;
  std::string output_data_type_;
  std::string weight_data_type_;

  // Graph hash of the model
  std::string graph_hash_;

  // Weight hash of the model
  std::string weight_hash_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(InferenceSession);
  void SetLoggingManager(const SessionOptions& session_options,
                         const Environment& session_env);
  void ConstructorCommon(const SessionOptions& session_options,
                         const Environment& session_env);
  [[nodiscard]] common::Status HasInvalidCombinationOfExecutionProviders() const;
  [[nodiscard]] common::Status SaveModelMetadata(const onnxruntime::Model& model);

#if !defined(ORT_MINIMAL_BUILD)

  [[nodiscard]] common::Status LoadOnnxModel(const PathString& model_uri);

  bool HasLocalSchema() const {
    return !custom_schema_registries_.empty();
  }

  common::Status SaveToOrtFormat(const std::filesystem::path& filepath) const;
#endif

  /**
   * Load an ORT format model.
   * @param model_uri absolute path of the model file.
   * @return OK if success.
   */
  [[nodiscard]] common::Status LoadOrtModel(const PathString& model_uri);

  /**
   * Load an ORT format model.
   * @param model_data Model data buffer
   * @param model_data_len Model data buffer size
   * @return OK if success.
   * @remarks TODO: Provide way to load from in-memory bytes without copying. InferenceSession would need to
   *                take ownership of the buffer passed in.
   */
  [[nodiscard]] common::Status LoadOrtModel(const void* model_data, int model_data_len);

  [[nodiscard]] common::Status LoadOrtModelWithLoader(std::function<Status()> load_ort_format_model_bytes);

  // Create a Logger for a single execution if possible. Otherwise use the default logger.
  // If a new logger is created, it will also be stored in new_run_logger,
  // which must remain valid for the duration of the execution.
  // If the default logger is used, new_run_logger will remain empty.
  // The returned value should be used in the execution.
  const logging::Logger& CreateLoggerForRun(const RunOptions& run_options,
                                            std::unique_ptr<logging::Logger>& new_run_logger);

  void InitLogger(logging::LoggingManager* logging_manager);

  static void TraceSessionOptions(const SessionOptions& session_options, bool captureState, const logging::Logger& logger);

  [[nodiscard]] common::Status CheckShapes(const std::string& input_name, const TensorShape& input_shape,
                                           const TensorShape& expected_shape, const char* input_output_moniker) const;

  [[nodiscard]] common::Status ValidateInputs(gsl::span<const std::string> feed_names,
                                              gsl::span<const OrtValue> feeds) const;

  [[nodiscard]] common::Status ValidateOutputs(gsl::span<const std::string> output_names,
                                               const std::vector<OrtValue>* p_fetches) const;

  [[nodiscard]] common::Status ValidateInputsOutputs(gsl::span<const std::string> feed_fetches_names,
                                                     gsl::span<const OrtValue> feeds_fetches,
                                                     const InputOutputDefMetaMap& input_output_meta_map,
                                                     ArgType arg_type) const;

  [[nodiscard]] common::Status WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms);

  template <typename T>
  void StartProfiling(const std::basic_string<T>& file_prefix);

  /*
   * Validate and parses the shrink arena request string from the user
   * List format: "device_0:device_id_0;device_1:device_id_1"
   * If we encounter an invalid request, we return an error
   * back to the user.
   */
  [[nodiscard]] common::Status ValidateAndParseShrinkArenaString(
      const std::string& ort_device_list,
      /*out*/ InlinedVector<AllocatorPtr>& arenas_to_shrink) const;

  /*
   * Performs the shrinkage of arenas requested to be shrunk by the user
   * The `arenas_to_shrink` parameter is got from ValidateAndParseShrinkArenaString()
   */
  void ShrinkMemoryArenas(gsl::span<const AllocatorPtr> arenas_to_shrink);

#ifdef _WIN32
  static void LogAllSessions();
#endif

#if !defined(ORT_MINIMAL_BUILD)
  virtual common::Status AddPredefinedTransformers(
      GraphTransformerManager& transformer_manager,
      TransformerLevel graph_optimization_level,
      MinimalBuildOptimizationHandling minimal_build_optimization_handling,
      RecordRuntimeOptimizationProducedNodeOpSchemaFn record_runtime_optimization_produced_op_schema_fn,
      const logging::Logger& logger) const;

  common::Status TransformGraph(onnxruntime::Graph& graph, bool saving_model_in_ort_format);

  onnxruntime::GraphTransformerManager graph_transformer_mgr_;

  InlinedHashSet<gsl::not_null<const ONNX_NAMESPACE::OpSchema*>> saved_runtime_optimization_produced_node_op_schemas_;
#endif
  // Any GraphTransformer/RewriteRule name in this set will not be enabled.
  InlinedHashSet<std::string> optimizers_to_disable_;

  // session_options_ must be declared *before* session_state_ in order to guarantee that session_options_ is destroyed
  // *after* the session_state_. This destruction order ensures that the custom operator library handles stored within
  // the session options are released after the individual operators are destroyed.
  SessionOptions session_options_;

  CheckLoadCancellationFn check_load_cancellation_fn_ = [this]() {
    return session_options_.IsLoadCancellationFlagSet();
  };

  /// Logging manager if provided.
  logging::LoggingManager* logging_manager_;

  /// User specified logging mgr; logging_manager_ is simply the ptr in this unique_ptr when available
  std::unique_ptr<logging::LoggingManager> user_logging_manager_;

  /// Logger for this session. WARNING: Will contain nullptr if logging_manager_ is nullptr.
  std::unique_ptr<logging::Logger> owned_session_logger_ = nullptr;

  // Profiler for this session.
  profiling::Profiler session_profiler_;

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  MemoryProfiler memory_profiler_;
#endif

  // Immutable state for each op in the model. Shared by all executors.
  // It has a dependency on execution_providers_.
  std::unique_ptr<SessionState> session_state_;

  // Threadpools per session. These are initialized and used for the entire duration of the session
  // when use_per_session_threads is true.
  std::basic_string<ORTCHAR_T> thread_pool_name_;
  std::basic_string<ORTCHAR_T> inter_thread_pool_name_;

  // This option allows to decrease CPU usage between infrequent
  // requests and forces any TP threads spinning stop immediately when the last of
  // concurrent ExecuteGraph() call returns.
  // Spinning is restarted on the next Run()
  bool force_spinning_stop_between_runs_ = false;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;

  // Global threadpools. These are intialized and used when use_per_session_threads is false *and*
  // the environment is created with create_global_thread_pools = true.
  onnxruntime::concurrency::ThreadPool* intra_op_thread_pool_from_env_{};
  onnxruntime::concurrency::ThreadPool* inter_op_thread_pool_from_env_{};

  // External threadpools.
  onnxruntime::concurrency::ThreadPool* external_intra_op_thread_pool_{};
  onnxruntime::concurrency::ThreadPool* external_inter_op_thread_pool_{};

  // initialized from session options
  // Determines which threadpools will be intialized and used for the duration of this session.
  // If true, use the per session ones, or else the global threadpools.
  bool use_per_session_threads_;

  KernelRegistryManager kernel_registry_manager_;

#if !defined(ORT_MINIMAL_BUILD)
  std::list<std::shared_ptr<onnxruntime::IOnnxRuntimeOpSchemaCollection>> custom_schema_registries_;
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

  // CustomRegistry objects own the corresponding KernelRegistry and OnnxRuntimeOpSchemaRegistry objects.
  // So its lifetime should be same as its constituents. This vector is to extend the lifetime of the owner.
  std::vector<std::shared_ptr<CustomRegistry>> custom_registries_;
#endif

  ModelMetadata model_metadata_;

  InputOutputDefMetaMap input_def_map_;
  InputOutputDefMetaMap output_def_map_;

  // Data transfer manager.
  DataTransferManager data_transfer_mgr_;

  // External data loader manager.
  ExternalDataLoaderManager external_data_loader_mgr_;

  // Number of concurrently running executors
  std::atomic<int> current_num_runs_ = 0;

  mutable std::mutex session_mutex_;         // to ensure only one thread can invoke Load/Initialize
  bool is_model_loaded_ = false;             // GUARDED_BY(session_mutex_)
  bool is_inited_ = false;                   // GUARDED_BY(session_mutex_)
  bool is_concurrent_run_supported_ = true;  // Graph execution in Run is GUARDED_BY(session_mutex_) if false

#ifdef ENABLE_LANGUAGE_INTEROP_OPS
  InterOpDomains interop_domains_;
#endif
  // used to support platform telemetry
  static std::atomic<uint32_t> global_session_id_;  // a monotonically increasing session id
  uint32_t session_id_;                             // the current session's id

  struct Telemetry {
    Telemetry() : time_sent_last_() {}
    uint32_t total_runs_since_last_ = 0;                              // the total number of Run() calls since the last report
    long long total_run_duration_since_last_ = 0;                     // the total duration (us) of Run() calls since the last report
    std::string event_name_;                                          // where the model is loaded from: ["model_loading_uri", "model_loading_proto", "model_loading_istream"]
    std::unordered_map<int64_t, long long> duration_per_batch_size_;  // the duration (us) of Run() calls per batch size since the last report

    TimePoint time_sent_last_;  // the TimePoint of the last report
    // Event Rate per provider < 20 peak events per second
    constexpr static long long kDurationBetweenSending = 1000 * 1000 * 60 * 10;  // duration in (us).  send a report every 10 mins
  } telemetry_;

  mutable std::mutex telemetry_mutex_;  // to ensure thread-safe access to telemetry data

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  bool session_activity_started_ = false;
  TraceLoggingActivity<telemetry_provider_handle> session_activity;
#endif

  // used to hold the ModelProto parsed in an applicable ctor to be used while calling parameter-less Load()
  ONNX_NAMESPACE::ModelProto model_proto_;

  // Flag indicating if ModelProto has been parsed in an applicable ctor
  bool is_model_proto_parsed_ = false;
  const Environment& environment_;

  // View of the bytes from an ORT format model.
  // If the session is started with an input byte array contains model data, and the caller
  // specifies that ORT should use the model bytes directly by setting the session config option
  // "session.use_ort_model_bytes_directly" to "1"
  //   We use the the byte array directly without copy to reduce peak memory usage
  //   (Short term) This will require the user to guarantee the life time of the model data
  //   until the session is created.
  //   (Longer term) If we are going to use the memory offsets directly for initializers, the model data
  //   should be alive until the InferenceSession goes away.
  // If the session is started with an input byte array contains model data, and the caller does not
  // specify ORT should use the model bytes directly
  // Or the session is started with a model_uri
  //   We store them currently in the ort_format_model_bytes_data_holder_ to make the Load + Initialize
  //   behave the same way as for an ONNX model, as we need some of the bytes for the Load (create the Model)
  //   and some for the Initialize (create SessionState).
  // Short term we free them after Initialize.
  // Longer term we may want to directly refer to offsets in this buffer for initializers so we don't need to copy
  // those into new OrtValue instances, at which point we won't free them until the InferenceSession goes away.
  gsl::span<const uint8_t> ort_format_model_bytes_;

  // This holds the actual model data
  // In case if the session is started with an input byte array contains model data, and the caller
  // specifies that ORT should use the model bytes directly by setting the session config option
  // "session.use_ort_model_bytes_directly" to "1", this will be empty
  std::vector<uint8_t> ort_format_model_bytes_data_holder_;

  bool using_ort_model_bytes_for_initializers_{false};

  // Container to store pre-packed weights to share between sessions.
  // The life-cycle of the cache itself is maintained by the user and the user will ensure
  // the cache is valid until any session reliant on it is still in scope.
  PrepackedWeightsContainer* prepacked_weights_container_ = nullptr;

  // Cache the EP instance if the user has configured the EP to capture a graph
  // for the model and all the necessary criteria for graph capture has been met.
  // At Run() time, if this member is not nullptr and the captured graph is ready
  // to replay, simply invoke ReplayGraph().
  struct CachedExecutionProviderForGraphReplay {
    CachedExecutionProviderForGraphReplay() = default;

    CachedExecutionProviderForGraphReplay(IExecutionProvider* execution_provider) : cached_execution_provider_for_graph_replay_(execution_provider) {}

    void SetExecutionProvider(IExecutionProvider* execution_provider) {
      cached_execution_provider_for_graph_replay_ = execution_provider;
    }

    bool IsGraphCaptureEnabled() const {
      return cached_execution_provider_for_graph_replay_ != nullptr && cached_execution_provider_for_graph_replay_->IsGraphCaptureEnabled();
    }

    bool IsGraphCaptured(int graph_annotation_id) const {
      return cached_execution_provider_for_graph_replay_ != nullptr && cached_execution_provider_for_graph_replay_->IsGraphCaptured(graph_annotation_id);
    }

    bool AllowGraphCaptureOnRun(int graph_annotation_id) const {
      return cached_execution_provider_for_graph_replay_ != nullptr && graph_annotation_id != kGraphAnnotationSkip;
    }

    Status ReplayGraph(int graph_annotation_id) {
      if (cached_execution_provider_for_graph_replay_) {
        return cached_execution_provider_for_graph_replay_->ReplayGraph(graph_annotation_id);
      }
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cached EP instance for graph replay is not set yet before calling ReplayGraph()");
    }

    const std::string& Type() const {
      return cached_execution_provider_for_graph_replay_->Type();
    }

    IExecutionProvider* cached_execution_provider_for_graph_replay_ = nullptr;
    // TODO(wy): Same as kCudaGraphAnnotationSkip in cuda_graph.h. Move to a common place.
    constexpr static int kGraphAnnotationSkip = -1;
  };

  CachedExecutionProviderForGraphReplay cached_execution_provider_for_graph_replay_;

#if !defined(ORT_MINIMAL_BUILD)
  // Enable nodestats collection
  std::optional<NodeStatsRecorder> node_stats_recorder_;
#endif
};

struct SessionIOBinding {
 public:
  SessionIOBinding(InferenceSession* session);

  const IOBinding* Get() const;
  IOBinding* Get();
  const InferenceSession* GetInferenceSession() const;
  InferenceSession* GetInferenceSession();

 private:
  InferenceSession* sess_;
  std::unique_ptr<IOBinding> binding_;
};

}  // namespace onnxruntime
