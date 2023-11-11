// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/framework/config_options.h"
#include "core/framework/ort_value.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/util/thread_utils.h"

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#include "core/framework/library_handles.h"
#endif

#if !defined(ORT_MINIMAL_BUILD)
#include "core/framework/sharding_spec.h"
#endif

namespace onnxruntime {

enum class ExecutionOrder {
  DEFAULT = 0,        // default topological sort
  PRIORITY_BASED = 1  // priority-based topological sort
};

enum class FreeDimensionOverrideType {
  Invalid = 0,
  Denotation = 1,
  Name = 2
};

enum class ExecutionPriority : int {
  GLOBAL_HIGHT = -100,
  LOCAL_HIGH = -10,
  DEFAULT = 0,
  LOCAL_LOW = 10,
  GLOBAL_LOW = 100
};

struct FreeDimensionOverride {
  std::string dim_identifier;
  FreeDimensionOverrideType dim_identifer_type;
  int64_t dim_value;
};

/**
 * Configuration information for a session.
 */
struct SessionOptions {
  ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL;

  // set the execution order of the graph
  ExecutionOrder execution_order = ExecutionOrder::DEFAULT;

  // enable profiling for this session.
  bool enable_profiling = false;

  // Non empty filepath enables serialization of the transformed optimized model to the specified filepath.
  //
  // Set session config value for ORT_SESSION_OPTIONS_CONFIG_SAVE_MODEL_FORMAT to 'ORT' or 'ONNX' to explicitly
  // specify the format.
  //
  // If session config value is not set, it will be assumed to be ONNX
  // unless the filepath ends in '.ort' (case insensitive).
  std::basic_string<ORTCHAR_T> optimized_model_filepath;

  // enable the memory pattern optimization.
  // The idea is if the input shapes are the same, we could trace the internal memory allocation
  // and generate a memory pattern for future request. So next time we could just do one allocation
  // with a big chunk for all the internal memory allocation.
  // See class 'OrtValuePatternPlanner'.
  bool enable_mem_pattern = true;

  // Enable memory resue in memory planning. Allows to reuse tensor buffer between tensors if they are of
  // the same size. The issue with this is it can lead to memory being held for longer than needed and
  // can impact peak memory consumption.
  bool enable_mem_reuse = true;

  // enable the memory arena on CPU
  // Arena may pre-allocate memory for future usage.
  // set this option to false if you don't want it.
  bool enable_cpu_mem_arena = true;

  // the prefix of the profile file. The current time will be appended to the file name.
  std::basic_string<ORTCHAR_T> profile_file_prefix = ORT_TSTR("onnxruntime_profile_");

  std::string session_logid;  ///< logger id to use for session output

  /// Log severity for the inference session. Applies to session load, initialization, etc.
  /// See https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
  /// Default = -1 (use default logger severity)
  int session_log_severity_level = -1;
  int session_log_verbosity_level = 0;  ///< VLOG level if debug build and session_log_severity_level is 0 (VERBOSE).

  unsigned max_num_graph_transformation_steps = 10;  // TODO choose a good default here?

  // set graph optimization level
  TransformerLevel graph_optimization_level = TransformerLevel::Level3;

  // controls the size of the thread pool used to parallelize the execution of tasks within individual nodes (ops)
  OrtThreadPoolParams intra_op_param;

  // controls the size of the thread pool used to parallelize the execution of nodes (ops)
  // configuring this makes sense only when you're using parallel executor
  OrtThreadPoolParams inter_op_param;

  // For models with symbolic input dimensions (most commonly batch size), specifies a set of values to override those
  // symbolic dimensions with, keyed by dimension parameters.
  std::vector<FreeDimensionOverride> free_dimension_overrides;

  // By default the session uses its own set of threadpools, unless this is set to false.
  // Use this in conjunction with the CreateEnvWithGlobalThreadPools API.
  bool use_per_session_threads = true;
  bool thread_pool_allow_spinning = true;

  // Deterministic compute is likely not as performant. This option is default to false.
  bool use_deterministic_compute = false;

  // Stores the configurations for this session
  // To add an configuration to this session, call OrtApis::AddSessionConfigEntry
  // The configuration keys and value formats are defined in
  // /include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
  ConfigOptions config_options;
  std::unordered_map<std::string, const OrtValue*> initializers_to_share_map;

  // See onnxruntime_c_api.h for detailed documentation.
  Status AddInitializer(_In_z_ const char* name, _In_ const OrtValue* val);

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
  // Customer supplied pre-processed data for external initializers
  InlinedHashMap<std::string, OrtValue> external_initializers;
  Status AddExternalInitializers(gsl::span<const std::string> names, gsl::span<const OrtValue> values);
#endif
#if !defined(ORT_MINIMAL_BUILD)
  // TODO: use InlinedHashMap<std::string, TensorPartitionSpec> tensor_partition_specs;
  std::unordered_map<std::string, distributed::TensorPartitionSpec> tensor_partition_specs;
  Status AddTensorPartitionSpec(
    const std::string& name,
    const std::string& spec,
    const std::vector<int64_t>& device_mesh_shape,
    const std::vector<int64_t>& device_mesh_elements
  );
#endif

  // custom function callback to create a thread
  OrtCustomCreateThreadFn custom_create_thread_fn = nullptr;

  // custom options to pass to custom_create_thread_fn
  void* custom_thread_creation_options = nullptr;

  // custom function callback to join a thread
  OrtCustomJoinThreadFn custom_join_thread_fn = nullptr;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Store handles to custom op libraries so that their lifetimes extend the lifetime of the session options object.
  // Lazily initialized by the first call to SessionOptions::AddCustomOpLibraryHandle().
  std::shared_ptr<LibraryHandles> custom_op_libs;
  void AddCustomOpLibraryHandle(PathString library_name, void* library_handle);
#endif

  // User specified logging func and param
  OrtLoggingFunction user_logging_function = nullptr;
  void* user_logging_param = nullptr;
};

}  // namespace onnxruntime
