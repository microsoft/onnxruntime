// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// NOTE: This file contains declarations of exported functions as WebAssembly API.
// Unlike a normal C-API, the purpose of this API is to make emcc to generate correct exports for the WebAssembly. The
// macro "EMSCRIPTEN_KEEPALIVE" helps the compiler to mark the function as an exported function of the WebAssembly
// module. Users are expected to consume those functions from JavaScript side.

#pragma once

#include <emscripten.h>

#ifdef USE_WEBGPU
#include <webgpu/webgpu.h>
#endif

#include <stddef.h>

struct OrtSession;
using ort_session_handle_t = OrtSession*;

struct OrtIoBinding;
using ort_io_binding_handle_t = OrtIoBinding*;

struct OrtSessionOptions;
using ort_session_options_handle_t = OrtSessionOptions*;

struct OrtRunOptions;
using ort_run_options_handle_t = OrtRunOptions*;

struct OrtValue;
using ort_tensor_handle_t = OrtValue*;

struct OrtLoraAdapter;
using ort_lora_adapter_handle_t = OrtLoraAdapter*;
struct OrtEpDevice;
using ort_ep_device_handle_t = const OrtEpDevice*;

extern "C" {

/**
 * perform global initialization. should be called only once.
 * @param num_threads number of total threads to use.
 * @param logging_level default logging level.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtInit(int num_threads, int logging_level);

/**
 * get the last error.
 * @param error_code [out] a pointer to accept the error code.
 * @param error_message [out] a pointer to accept the error message. The message buffer is only available before any ORT API is called.
 */
int EMSCRIPTEN_KEEPALIVE OrtGetLastError(int* error_code, const char** error_message);

/**
 * create an instance of ORT session options.
 * assume that all enum type parameters, such as graph_optimization_level, execution_mode, and log_severity_level,
 * are checked and set properly at JavaScript.
 * @param graph_optimization_level disabled, basic, extended, layout, or enable all
 * @param enable_cpu_mem_arena enable or disable cpu memory arena
 * @param enable_mem_pattern enable or disable memory pattern
 * @param execution_mode sequential or parallel execution mode
 * @param enable_profiling enable or disable profiling.
 * @param profile_file_prefix file prefix for profiling data. it's a no-op and for a future use.
 * @param log_id logger id for session output
 * @param log_severity_level verbose, info, warning, error or fatal
 * @param log_verbosity_level vlog level
 * @param optimized_model_filepath filepath of the optimized model to dump.
 * @returns a session option handle. Caller must release it after use by calling OrtReleaseSessionOptions().
 */
ort_session_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSessionOptions(size_t graph_optimization_level,
                                                                          bool enable_cpu_mem_arena,
                                                                          bool enable_mem_pattern,
                                                                          size_t execution_mode,
                                                                          bool enable_profiling,
                                                                          const char* profile_file_prefix,
                                                                          const char* log_id,
                                                                          size_t log_severity_level,
                                                                          size_t log_verbosity_level,
                                                                          const char* optimized_model_filepath);

/**
 * append an execution provider for a session.
 * @param name the name of the execution provider
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtAppendExecutionProvider(ort_session_options_handle_t session_options,
                                                    const char* name,
                                                    const char* const* provider_options_keys,
                                                    const char* const* provider_options_values,
                                                    size_t num_keys);

/**
 * get the execution provider devices registered with the environment.
 *
 * The array is owned by the environment and remains valid for its lifetime. The caller must not free it.
 *
 * @param ep_devices receives a pointer to the array of ort_ep_device_handle_t
 * @param num_ep_devices receives the number of entries in the array
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtGetEpDevices(const ort_ep_device_handle_t** ep_devices, size_t* num_ep_devices);

/**
 * get the execution provider name that an execution provider device is registered under.
 *
 * More than one device may share an EP name, e.g. when an EP reports one device per GPU.
 *
 * @returns the name, owned by the environment. It remains valid for the lifetime of the environment.
 */
const char* EMSCRIPTEN_KEEPALIVE OrtEpDevice_EpName(ort_ep_device_handle_t ep_device);

/**
 * append an execution provider for a session, selecting it by execution provider device.
 *
 * Unlike OrtAppendExecutionProvider, which resolves an EP name against ORT's built-in EP table, this selects
 * from the devices registered with the environment, which is how a plugin EP is selected. Use OrtGetEpDevices
 * and OrtEpDevice_EpName to choose the devices to pass in.
 *
 * @param ep_devices the devices to select, all of which must belong to the same execution provider
 * @param num_ep_devices the number of entries in ep_devices
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtAppendExecutionProviderV2(ort_session_options_handle_t session_options,
                                                      ort_ep_device_handle_t* ep_devices,
                                                      size_t num_ep_devices,
                                                      const char* const* provider_options_keys,
                                                      const char* const* provider_options_values,
                                                      size_t num_keys);

/**
 * add a free dimension override for one dimension of a session's input.
 */
int EMSCRIPTEN_KEEPALIVE OrtAddFreeDimensionOverride(ort_session_options_handle_t session_options,
                                                     const char* dim_param_name,
                                                     int dim_value);

/**
 * store configurations for a session.
 * @param session_options a handle to session options created by OrtCreateSessionOptions
 * @param config_key configuration keys and value formats are defined in
 *                   include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
 * @param config_value value for config_key
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtAddSessionConfigEntry(ort_session_options_handle_t session_options,
                                                  const char* config_key,
                                                  const char* config_value);

/**
 * release the specified ORT session options.
 */
int EMSCRIPTEN_KEEPALIVE OrtReleaseSessionOptions(ort_session_options_handle_t session_options);

/**
 * create an instance of ORT session.
 * @param data a pointer to a buffer that contains the ONNX or ORT format model.
 * @param data_length the size of the buffer in bytes.
 * @returns an ORT session handle. Caller must release it after use by calling OrtReleaseSession().
 */
ort_session_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSession(void* data,
                                                           size_t data_length,
                                                           ort_session_options_handle_t session_options);

/**
 * release the specified ORT session.
 */
int EMSCRIPTEN_KEEPALIVE OrtReleaseSession(ort_session_handle_t session);

/**
 * get model's input count and output count.
 * @param session handle of the specified session
 * @param input_count [out] a pointer to a size_t variable to accept input_count.
 * @param output_count [out] a pointer to a size_t variable to accept output_count.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtGetInputOutputCount(ort_session_handle_t session,
                                                size_t* input_count,
                                                size_t* output_count);

/**
 * get the metadata of the specified input or output of the model.
 * @param session handle of the specified session
 * @param index the input index or output index. index should be in range [0, input_count + output_count). if the index
 * is in range [0, input_count), it's an input index. otherwise, it's an output index.
 * @param name_cstr_ptr [out] a pointer to a buffer which contains C-style string of the name of the input or output. Caller must release the C style string after use by
 * calling OrtFree().
 * @param type_info_ptr [out] a pointer to a buffer which contains the type information of the input or output. Caller must release the buffer after use by
 * calling OrtFree().
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtGetInputOutputMetadata(ort_session_handle_t session, size_t index, char** name_cstr_ptr, void** type_info_ptr);

/**
 * free the specified buffer.
 * @param ptr a pointer to the buffer.
 */
int EMSCRIPTEN_KEEPALIVE OrtFree(void* ptr);

/**
 * create an instance of ORT tensor.
 * @param data_type data type defined in enum ONNXTensorElementDataType.
 * @param data for numeric tensor: a pointer to the tensor data buffer. for string tensor: a pointer to a C-Style null terminated string array.
 * @param data_length size of the buffer 'data' in bytes.
 * @param dims a pointer to an array of dims. the array should contain (dims_length) element(s).
 * @param dims_length the length of the tensor's dimension
 * @param data_location specify the memory location of the tensor data. 0 for CPU, 1 for GPU buffer.
 * @returns a tensor handle. Caller must release it after use by calling OrtReleaseTensor().
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateTensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length, int data_location);

/**
 * get type, shape info and data of the specified tensor.
 * @param tensor handle of the tensor.
 * @param data_type [out] specify the memory to write data type
 * @param data [out] specify the memory to write the tensor data. for string tensor: an array of C-Style null terminated string.
 * @param dims [out] specify the memory to write address of the buffer containing value of each dimension.
 * @param dims_length [out] specify the memory to write dims length
 * @remarks following temporary buffers are allocated during the call. Caller must release the buffers after use by calling OrtFree():
 *           'dims' (for all types of tensor), 'data' (only for string tensor)
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtGetTensorData(ort_tensor_handle_t tensor, size_t* data_type, void** data, size_t** dims, size_t* dims_length);

/**
 * release the specified tensor.
 */
int EMSCRIPTEN_KEEPALIVE OrtReleaseTensor(ort_tensor_handle_t tensor);

/**
 * create an instance of ORT LoRA adapter from a buffer in the LoRA adapter format (.onnx_adapter).
 * @param data a pointer to a buffer that contains the LoRA adapter data. The data is copied.
 * @param data_length the size of the buffer in bytes
 * @returns a LoRA adapter handle. Caller must release it after use by calling OrtReleaseLoraAdapter().
 */
ort_lora_adapter_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateLoraAdapter(const void* data, size_t data_length);

/**
 * release the specified ORT LoRA adapter.
 */
int EMSCRIPTEN_KEEPALIVE OrtReleaseLoraAdapter(ort_lora_adapter_handle_t adapter);

/**
 * create an instance of ORT run options.
 * @param log_severity_level verbose, info, warning, error or fatal
 * @param log_verbosity_level vlog level
 * @param terminate if true, all incomplete OrtRun calls will exit as soon as possible
 * @param tag tag for this run
 * @returns a run option handle. Caller must release it after use by calling OrtReleaseRunOptions().
 */
ort_run_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateRunOptions(size_t log_severity_level,
                                                                  size_t log_verbosity_level,
                                                                  bool terminate,
                                                                  const char* tag);

/**
 * set a single run configuration entry
 * @param run_options a handle to run options created by OrtCreateRunOptions
 * @param config_key configuration keys and value formats are defined in
 *                   include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h
 * @param config_value value for config_key
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtAddRunConfigEntry(ort_run_options_handle_t run_options,
                                              const char* config_key,
                                              const char* config_value);

/**
 * add a LoRA adapter to the list of active adapters of the run options.
 * @param run_options a handle to run options created by OrtCreateRunOptions
 * @param adapter a handle to a LoRA adapter created by OrtCreateLoraAdapter. It must not be released while the run
 *                options are in use.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtRunOptionsAddActiveLoraAdapter(ort_run_options_handle_t run_options,
                                                           ort_lora_adapter_handle_t adapter);

/**
 * release the specified ORT run options.
 */
int EMSCRIPTEN_KEEPALIVE OrtReleaseRunOptions(ort_run_options_handle_t run_options);

/**
 * create an instance of ORT IO binding.
 */
ort_io_binding_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateBinding(ort_session_handle_t session);

/**
 * bind an input tensor to the IO binding instance. A cross device copy will be performed if necessary.
 * @param io_binding handle of the IO binding
 * @param name name of the input
 * @param input handle of the input tensor
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtBindInput(ort_io_binding_handle_t io_binding,
                                      const char* name,
                                      ort_tensor_handle_t input);

/**
 * bind an output tensor or location to the IO binding instance.
 * @param io_binding handle of the IO binding
 * @param name name of the output
 * @param output handle of the output tensor. nullptr for output location binding.
 * @param output_location specify the memory location of the output tensor data.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtBindOutput(ort_io_binding_handle_t io_binding,
                                       const char* name,
                                       ort_tensor_handle_t output,
                                       int output_location);

/**
 * clear all bound outputs.
 */
int EMSCRIPTEN_KEEPALIVE OrtClearBoundOutputs(ort_io_binding_handle_t io_binding);

/**
 * release the specified ORT IO binding.
 */
int EMSCRIPTEN_KEEPALIVE OrtReleaseBinding(ort_io_binding_handle_t io_binding);

/**
 * inference the model.
 * @param session handle of the specified session
 * @param io_binding handle of the IO binding
 * @param run_options handle of the run options
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtRunWithBinding(ort_session_handle_t session,
                                           ort_io_binding_handle_t io_binding,
                                           size_t output_count,
                                           ort_tensor_handle_t* outputs,
                                           ort_run_options_handle_t run_options);

/**
 * inference the model.
 * @param session handle of the specified session
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtRun(ort_session_handle_t session,
                                const char** input_names,
                                const ort_tensor_handle_t* inputs,
                                size_t input_count,
                                const char** output_names,
                                size_t output_count,
                                ort_tensor_handle_t* outputs,
                                ort_run_options_handle_t run_options);

/**
 * end profiling.
 * @param session handle of the specified session
 * @returns a pointer to a buffer which contains C-style string of profile filename.
 * Caller must release the C style string after use by calling OrtFree().
 */
char* EMSCRIPTEN_KEEPALIVE OrtEndProfiling(ort_session_handle_t session);

// WebGPU API Section

#ifdef USE_WEBGPU

/**
 * create a WebGPU instance that supports the wait behavior required by WebGPU EP.
 *
 * @returns a WGPUInstance handle.
 */
WGPUInstance EMSCRIPTEN_KEEPALIVE OrtCreateWebGpuInstance();

/**
 * get the GPU Device by device ID.
 *
 * This function is only available after the GPU Device is initialized in WebGpuContextFactory.
 *
 * @returns a WGPUDevice handle.
 */
WGPUDevice EMSCRIPTEN_KEEPALIVE OrtGetWebGpuDevice(int device_id);

#endif

// Training API Section
};
