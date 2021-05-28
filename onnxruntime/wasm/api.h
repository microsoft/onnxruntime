// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// NOTE: This file contains declarations of exported functions as WebAssembly API.
// Unlike a normal C-API, the purpose of this API is to make emcc to generate correct exports for the WebAssembly. The
// macro "EMSCRIPTEN_KEEPALIVE" helps the compiler to mark the function as an exported funtion of the WebAssembly
// module. Users are expected to consume those functions from JavaScript side.

#pragma once

#include <emscripten.h>

#include <stddef.h>

struct OrtSession;
using ort_session_handle_t = OrtSession*;

struct OrtSessionOptions;
using ort_session_options_handle_t = OrtSessionOptions*;

struct OrtRunOptions;
using ort_run_options_handle_t = OrtRunOptions*;

struct OrtValue;
using ort_tensor_handle_t = OrtValue*;

extern "C" {

/**
 * perform global initialization. should be called only once.
 * @param num_threads number of total threads to use.
 * @param logging_level default logging level.
 */
int EMSCRIPTEN_KEEPALIVE OrtInit(int num_threads, int logging_level);

/**
 * create an instance of ORT session options.
 * assume that all enum type parameters, such as graph_optimization_level, execution_mode, and log_severity_level,
 * are checked and set properly at JavaScript.
 * @param graph_optimization_level disabled, basic, extended, or enable all
 * @param enable_cpu_mem_arena enable or disable cpu memory arena
 * @param enable_mem_pattern enable or disable memory pattern
 * @param execution_mode sequential or parallel execution mode
 * @param enable_profiling enable or disable profiling. it's a no-op and for a future use.
 * @param profile_file_prefix file prefix for profiling data. it's a no-op and for a future use.
 * @param log_id logger id for session output
 * @param log_severity_level verbose, info, warning, error or fatal
 * @param log_verbosity_level vlog level
 * @returns a pointer to a session option handle and must be freed by calling OrtReleaseSessionOptions().
 */
ort_session_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSessionOptions(size_t graph_optimization_level,
                                                                          bool enable_cpu_mem_arena,
                                                                          bool enable_mem_pattern,
                                                                          size_t execution_mode,
                                                                          bool enable_profiling,
                                                                          const char* profile_file_prefix,
                                                                          const char* log_id,
                                                                          size_t log_severity_level,
                                                                          size_t log_verbosity_level);

/**
 * store configurations for a session.
 * @param session_options a handle to session options created by OrtCreateSessionOptions
 * @param config_key configuration keys and value formats are defined in
 *                   include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
 * @param config_value value for config_key
 */
int EMSCRIPTEN_KEEPALIVE OrtAddSessionConfigEntry(ort_session_options_handle_t session_options,
                                                  const char* config_key,
                                                  const char* config_value);

/**
 * release the specified ORT session options.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseSessionOptions(ort_session_options_handle_t session_options);

/**
 * create an instance of ORT session.
 * @param data a pointer to a buffer that contains the ONNX or ORT format model.
 * @param data_length the size of the buffer in bytes.
 * @returns a handle of the ORT session.
 */
ort_session_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSession(void* data,
                                                           size_t data_length,
                                                           ort_session_options_handle_t session_options);

/**
 * release the specified ORT session.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseSession(ort_session_handle_t session);

size_t EMSCRIPTEN_KEEPALIVE OrtGetInputCount(ort_session_handle_t session);
size_t EMSCRIPTEN_KEEPALIVE OrtGetOutputCount(ort_session_handle_t session);

/**
 * get the model's input name.
 * @param session handle of the specified session
 * @param index the input index
 * @returns a pointer to a buffer which contains C-style string. Caller must release the C style string after use by
 * calling OrtFree().
 */
char* EMSCRIPTEN_KEEPALIVE OrtGetInputName(ort_session_handle_t session, size_t index);
/**
 * get the model's output name.
 * @param session handle of the specified session
 * @param index the output index
 * @returns a pointer to a buffer which contains C-style string. Caller must release the C style string after use by
 * calling OrtFree().
 */
char* EMSCRIPTEN_KEEPALIVE OrtGetOutputName(ort_session_handle_t session, size_t index);

/**
 * free the specified buffer.
 * @param ptr a pointer to the buffer.
 */
void EMSCRIPTEN_KEEPALIVE OrtFree(void* ptr);

/**
 * create an instance of ORT tensor.
 * @param data_type data type defined in enum ONNXTensorElementDataType.
 * @param data a pointer to the tensor data.
 * @param data_length size of the tensor data in bytes.
 * @param dims a pointer to an array of dims. the array should contain (dims_length) element(s).
 * @param dims_length the length of the tensor's dimension
 * @returns a handle of the tensor.
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateTensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length);

/**
 * get type, shape info and data of the specified tensor.
 * @param tensor handle of the tensor.
 * @param data_type [out] specify the memory to write data type
 * @param data [out] specify the memory to write the tensor data
 * @param dims [out] specify the memory to write address of the buffer containing value of each dimension.
 * @param dims_length [out] specify the memory to write dims length
 * @remarks a temporary buffer 'dims' is allocated during the call. Caller must release the buffer after use by calling OrtFree().
 */
int EMSCRIPTEN_KEEPALIVE OrtGetTensorData(ort_tensor_handle_t tensor, int* data_type, void** data, size_t** dims, size_t* dims_length);

/**
 * release the specified tensor.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseTensor(ort_tensor_handle_t tensor);

/**
 * create an instance of ORT run options.
 * @param log_severity_level verbose, info, warning, error or fatal
 * @param log_verbosity_level vlog level
 * @param terminate if true, all incomplete OrtRun calls will exit as soon as possible
 * @param tag tag for this run
 * @returns a pointer to a run option handle and must be freed by calling OrtReleaseRunOptions().
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
 */
int EMSCRIPTEN_KEEPALIVE OrtAddRunConfigEntry(ort_run_options_handle_t run_options,
                                              const char* config_key,
                                              const char* config_value);

/**
 * release the specified ORT run options.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseRunOptions(ort_run_options_handle_t run_options);

/**
 * inference the model.
 * @param session handle of the specified session
 * @returns error code defined in enum OrtErrorCode
 */
int EMSCRIPTEN_KEEPALIVE OrtRun(ort_session_handle_t session,
                                const char** input_names,
                                const ort_tensor_handle_t* inputs,
                                size_t input_count,
                                const char** output_names,
                                size_t output_count,
                                ort_tensor_handle_t* outputs,
                                ort_run_options_handle_t run_options);
};
