// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
 * This file defines SessionOptions Config Keys and format of the Config Values.
 *
 * The Naming Convention for a SessionOptions Config Key,
 * "[Area][.[SubArea1].[SubArea2]...].[Keyname]"
 * Such as "ep.cuda.use_arena"
 * The Config Key cannot be empty
 * The maximum length of the Config Key is 128
 *
 * The string format of a SessionOptions Config Value is defined individually for each Config.
 * The maximum length of the Config Value is 1024
 */

// Key for disable PrePacking,
// If the config value is set to "1" then the prepacking is disabled, otherwise prepacking is enabled (default value)
static const char* const kOrtSessionOptionsConfigDisablePrepacking = "session.disable_prepacking";

// A value of "1" means allocators registered in the env will be used. "0" means the allocators created in the session
// will be used. Use this to override the usage of env allocators on a per session level.
static const char* const kOrtSessionOptionsConfigUseEnvAllocators = "session.use_env_allocators";

// Set to 'ORT' (case sensitive) to load an ORT format model.
// If unset, model type will default to ONNX unless inferred from filename ('.ort' == ORT format) or bytes to be ORT
static const char* const kOrtSessionOptionsConfigLoadModelFormat = "session.load_model_format";

// Set to 'ORT' (case sensitive) to save optimized model in ORT format when SessionOptions.optimized_model_path is set.
// If unset, format will default to ONNX unless optimized_model_filepath ends in '.ort'.
static const char* const kOrtSessionOptionsConfigSaveModelFormat = "session.save_model_format";

// If a value is "1", flush-to-zero and denormal-as-zero are applied. The default is "0".
// When multiple sessions are created, a main thread doesn't override changes from succeeding session options,
// but threads in session thread pools follow option changes.
// When ORT runs with OpenMP, the same rule is applied, i.e. the first session option to flush-to-zero and
// denormal-as-zero is only applied to global OpenMP thread pool, which doesn't support per-session thread pool.
// Note that an alternative way not using this option at runtime is to train and export a model without denormals
// and that's recommended because turning this option on may hurt model accuracy.
static const char* const kOrtSessionOptionsConfigSetDenormalAsZero = "session.set_denormal_as_zero";

// It controls to run quantization model in QDQ (QuantizelinearDeQuantizelinear) format or not.
// "0": enable. ORT does fusion logic for QDQ format.
// "1": disable. ORT doesn't do fusion logic for QDQ format.
// Its default value is "0"
static const char* const kOrtSessionOptionsDisableQuantQDQ = "session.disable_quant_qdq";

// If set to "1", enables the removal of QuantizeLinear/DequantizeLinear node pairs once all QDQ handling has been
// completed. e.g. If after all QDQ handling has completed and we have -> FloatOp -> Q -> DQ -> FloatOp -> the
// Q -> DQ could potentially be removed. This will provide a performance benefit by avoiding going from float to
// 8-bit and back to float, but could impact accuracy. The impact on accuracy will be model specific and depend on
// other factors like whether the model was created using Quantization Aware Training or Post Training Quantization.
// As such, it's best to test to determine if enabling this works well for your scenario.
// The default value is "0"
// Available since version 1.11.
static const char* const kOrtSessionOptionsEnableQuantQDQCleanup = "session.enable_quant_qdq_cleanup";

// Enable or disable gelu approximation in graph optimization. "0": disable; "1": enable. The default is "0".
// GeluApproximation has side effects which may change the inference results. It is disabled by default due to this.
static const char* const kOrtSessionOptionsEnableGeluApproximation = "optimization.enable_gelu_approximation";

// Enable or disable using device allocator for allocating initialized tensor memory. "1": enable; "0": disable. The default is "0".
// Using device allocators means the memory allocation is made using malloc/new.
static const char* const kOrtSessionOptionsUseDeviceAllocatorForInitializers = "session.use_device_allocator_for_initializers";

// Configure whether to allow the inter_op/intra_op threads spinning a number of times before blocking
// "0": thread will block if found no job to run
// "1": default, thread will spin a number of times before blocking
static const char* const kOrtSessionOptionsConfigAllowInterOpSpinning = "session.inter_op.allow_spinning";
static const char* const kOrtSessionOptionsConfigAllowIntraOpSpinning = "session.intra_op.allow_spinning";

// Key for using model bytes directly for ORT format
// If a session is created using an input byte array contains the ORT format model data,
// By default we will copy the model bytes at the time of session creation to ensure the model bytes
// buffer is valid.
// Setting this option to "1" will disable copy the model bytes, and use the model bytes directly. The caller
// has to guarantee that the model bytes are valid until the ORT session using the model bytes is destroyed.
static const char* const kOrtSessionOptionsConfigUseORTModelBytesDirectly = "session.use_ort_model_bytes_directly";

// This should only be specified when exporting an ORT format model for use on a different platform.
// If the ORT format model will be used on ARM platforms set to "1". For other platforms set to "0"
// Available since version 1.11.
static const char* const kOrtSessionOptionsQDQIsInt8Allowed = "session.qdqisint8allowed";

// Specifies how minimal build graph optimizations are handled in a full build.
// These optimizations are at the extended level or higher.
// Possible values and their effects are:
// "save": Save runtime optimizations when saving an ORT format model.
// "apply": Only apply optimizations available in a minimal build.
// ""/<unspecified>: Apply optimizations available in a full build.
// Available since version 1.11.
static const char* const kOrtSessionOptionsConfigMinimalBuildOptimizations =
    "optimization.minimal_build_optimizations";

// Note: The options specific to an EP should be specified prior to appending that EP to the session options object in
// order for them to take effect.

// Specifies a list of stop op types. Nodes of a type in the stop op types and nodes downstream from them will not be
// run by the NNAPI EP.
// The value should be a ","-delimited list of op types. For example, "Add,Sub".
// If not specified, the default set of stop ops is used. To specify an empty stop ops types list and disable stop op
// exclusion, set the value to "".
static const char* const kOrtSessionOptionsConfigNnapiEpPartitioningStopOps = "ep.nnapi.partitioning_stop_ops";

// Enabling dynamic block-sizing for multithreading.
// With a positive value, thread pool will split a task of N iterations to blocks of size starting from:
// N / (num_of_threads * dynamic_block_base)
// As execution progresses, the size will decrease according to the diminishing residual of N,
// meaning the task will be distributed in smaller granularity for better parallelism.
// For some models, it helps to reduce the variance of E2E inference latency and boost performance.
// The feature will not function by default, specify any positive integer, e.g. "4", to enable it.
// Available since version 1.11.
static const char* const kOrtSessionOptionsConfigDynamicBlockBase = "session.dynamic_block_base";

// This option allows to decrease CPU usage between infrequent
// requests and forces any TP threads spinning stop immediately when the last of
// concurrent Run() call returns.
// Spinning is restarted on the next Run() call.
// Applies only to internal thread-pools
static const char* const kOrtSessionOptionsConfigForceSpinningStop = "session.force_spinning_stop";

// "1": all inconsistencies encountered during shape and type inference
// will result in failures.
// "0": in some cases warnings will be logged but processing will continue. The default.
// May be useful to expose bugs in models.
static const char* const kOrtSessionOptionsConfigStrictShapeTypeInference = "session.strict_shape_type_inference";
